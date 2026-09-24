"""Adapted from ProPos by Zhizhong Huang; see THIRD_PARTY.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.gather_layer import GatherLayer
from .momentum_base import MomentumEncoderBase

class BYOLWrapper(MomentumEncoderBase):
    """
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
    https://github.com/lucidrains/byol-pytorch/tree/master/byol_pytorch
    """

    def __init__(self, encoder, num_cluster, in_dim, temperature, hidden_size=4096, fea_dim=256, byol_momentum=0.999, symmetric=True, shuffling_bn=True, latent_std=0.001, queue_size=0):
        nn.Module.__init__(self)
        self.symmetric = symmetric
        self.m = byol_momentum
        self.shuffling_bn = shuffling_bn
        self.num_cluster = num_cluster
        self.temperature = temperature
        self.fea_dim = fea_dim
        self.latent_std = latent_std
        self.queue_size = queue_size
        self.encoder_q = encoder
        self.projector_q = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, fea_dim))
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)
        self.predictor = nn.Sequential(nn.Linear(fea_dim, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, fea_dim))
        self.q_params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        self.k_params = list(self.encoder_k.parameters()) + list(self.projector_k.parameters())
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.encoder = nn.Sequential(self.encoder_k, self.projector_k)
        if self.queue_size > 0:
            self.register_buffer('queue', torch.randn(queue_size, fea_dim))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('queue_ind', torch.zeros(queue_size, dtype=torch.long))
        else:
            self.register_buffer('queue', None)
            self.register_buffer('queue_ptr', None)
            self.register_buffer('queue_ind', None)

    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.num_cluster, n_samples).to(x)
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def compute_cluster_loss(self, q_centers, k_centers, temperature, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k
        zero_classes = torch.arange(self.num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels), self.num_cluster), dim=0) == 0]
        mask = torch.zeros((self.num_cluster, self.num_cluster), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_cluster, self.num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_cluster - 1)
        loss = -pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.0
        loss = loss.sum() / (self.num_cluster - len(zero_classes))
        return loss

    def forward_k(self, im_k, psedo_labels, unlabeled=True):
        with torch.no_grad():
            if self.shuffling_bn:
                im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k_)
                k = k.float()
                k = self.projector_k(k)
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
                k = self.encoder_k(im_k)
                k = self.projector_k(k)
                k = nn.functional.normalize(k, dim=1)
            k = k.detach_()
            all_k = self.concat_all_gather(k)
            if unlabeled:
                return (k, None, all_k)
            if self.queue_size > 0:
                k_centers = self.compute_centers(torch.cat([all_k, self.queue], dim=0), torch.cat([psedo_labels, self.psedo_labels[self.queue_ind]], dim=0))
            else:
                k_centers = self.compute_centers(all_k, psedo_labels)
        return (k, k_centers, all_k)

    def forward_loss(self, im_q, im_k, psedo_labels: torch.Tensor, unlabeled=False):
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        batch_psedo_labels = psedo_labels
        batch_all_psedo_labels = self.concat_all_gather(batch_psedo_labels)
        k, k_centers, all_k = self.forward_k(im_k, batch_all_psedo_labels, unlabeled=unlabeled)
        noise_q = q + torch.randn_like(q) * self.latent_std
        contrastive_loss = -2 * F.cosine_similarity(self.predictor(noise_q), k).mean()
        all_q = F.normalize(torch.cat(GatherLayer.apply(q), dim=0), dim=1)
        if not unlabeled:
            if self.queue_size > 0:
                queue_labels = torch.cat([batch_all_psedo_labels, self.psedo_labels[self.queue_ind]])
                q_centers = self.compute_centers(torch.cat([all_q, self.queue]), queue_labels)
                cluster_loss_batch = self.compute_cluster_loss(q_centers, k_centers, self.temperature, queue_labels)
            else:
                q_centers = self.compute_centers(all_q, batch_all_psedo_labels)
                cluster_loss_batch = self.compute_cluster_loss(q_centers, k_centers, self.temperature, batch_psedo_labels)
        else:
            cluster_loss_batch = None
        return (contrastive_loss, cluster_loss_batch, all_q, all_k)

    def forward(self, im_q, im_k, indices, momentum_update=True, v2=False, unlabeled=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if v2:
            return self.forward_v2(im_q, im_k, indices, momentum_update=momentum_update, unlabeled=unlabeled)
        psedo_labels = self.psedo_labels[indices]
        if self.symmetric:
            contrastive_loss1, cluster_loss_batch1, q1, k1 = self.forward_loss(im_q, im_k, psedo_labels, unlabeled)
            contrastive_loss2, cluster_loss_batch2, q2, k2 = self.forward_loss(im_k, im_q, psedo_labels, unlabeled)
            contrastive_loss = 0.5 * (contrastive_loss1 + contrastive_loss2)
            if not unlabeled:
                cluster_loss_batch = 0.5 * (cluster_loss_batch1 + cluster_loss_batch2)
            else:
                cluster_loss_batch = None
            q = torch.cat([q1, q2], dim=0)
            k = torch.cat([k1, k2], dim=0)
        else:
            contrastive_loss, cluster_loss_batch, q, k = self.forward_loss(im_q, im_k, psedo_labels, unlabeled)
        if momentum_update:
            with torch.no_grad():
                self._momentum_update_key_encoder()
        if self.queue_size > 0:
            indices = self.concat_all_gather(indices)
            if self.symmetric:
                indices = indices.repeat(2)
            self._dequeue_and_enqueue(k, indices)
        return (contrastive_loss, cluster_loss_batch, q)

    def forward_v2(self, im_q_, im_k_, indices, momentum_update=True, unlabeled=False):
        if momentum_update:
            with torch.no_grad():
                self._momentum_update_key_encoder()
        im_q = torch.cat([im_q_, im_k_], dim=0)
        im_k = torch.cat([im_k_, im_q_], dim=0)
        psedo_labels = self.psedo_labels[indices]
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        batch_psedo_labels = psedo_labels
        batch_all_psedo_labels = self.concat_all_gather(batch_psedo_labels)
        k, _, all_k = self.forward_k(im_k, batch_all_psedo_labels.repeat(2), unlabeled=unlabeled)
        noise_q = q + torch.randn_like(q) * self.latent_std
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()
        if unlabeled:
            return (contrastive_loss, None, None)
        k1, k2 = all_k.chunk(2, dim=0)
        k_centers_1 = self.compute_centers(k1, batch_all_psedo_labels)
        k_centers_2 = self.compute_centers(k2, batch_all_psedo_labels)
        all_q = F.normalize(torch.cat(GatherLayer.apply(q), dim=0), dim=1)
        q1, q2 = all_q.chunk(2, dim=0)
        q_centers_1 = self.compute_centers(q1, batch_all_psedo_labels)
        q_centers_2 = self.compute_centers(q2, batch_all_psedo_labels)
        cluster_loss_batch = (self.compute_cluster_loss(q_centers_1, k_centers_1, self.temperature, batch_psedo_labels) + self.compute_cluster_loss(q_centers_2, k_centers_2, self.temperature, batch_psedo_labels)) / 2.0
        return (contrastive_loss, cluster_loss_batch, all_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, indices):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        self.queue[ptr:ptr + batch_size] = keys
        self.queue_ind[ptr:ptr + batch_size] = indices
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
