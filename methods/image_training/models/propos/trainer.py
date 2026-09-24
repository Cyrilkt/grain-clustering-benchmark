"""Grain training loop adapted from ProPos (Zhizhong Huang)."""
import torch
import numpy as np
from utils.ops import convert_to_ddp
from utils.ops import convert_to_cuda
from .model import BYOLWrapper
from models.base import TrainTask
from network import backbone_dict
from torchvision import transforms
import tqdm

class GrainResize:
    """Native deterministic resize: PIL default interpolation, no center crop."""

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, image):
        return image if image.size == self.size else image.resize(self.size)


def scattering_is_active(n_iter, iterations_per_epoch, warmup_epochs):
    """The sole scattering gate is the ordinary optimization warm-up."""
    return (n_iter - 1) / iterations_per_epoch > warmup_epochs

class BYOL(TrainTask):
    __BYOLWrapper__ = BYOLWrapper

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        byol = self.__BYOLWrapper__(encoder, in_dim=dim_in, num_cluster=self.num_cluster, temperature=opt.temperature, hidden_size=opt.hidden_size, fea_dim=opt.feat_dim, byol_momentum=opt.momentum_base, symmetric=opt.symmetric, shuffling_bn=opt.shuffling_bn, latent_std=opt.latent_std, queue_size=opt.queue_size)
        if opt.syncbn:
            if opt.shuffling_bn:
                byol.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.encoder_q)
                byol.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.projector_q)
                byol.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.predictor)
            else:
                byol = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol)
        optimizer = torch.optim.SGD(params=self.collect_params(byol, exclude_bias_and_bn=opt.exclude_bias_and_bn), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        self.feature_extractor_copy = None
        byol = byol.cuda()
        self.feature_extractor = byol.encoder
        byol = convert_to_ddp(byol)
        self.byol = byol
        self.optimizer = optimizer

    def set_loader(self):
        opt = self.opt
        normalize = self.normalize(opt.dataset)
        train_transform = self.train_transform(normalize)
        self.logger.msg_str('set transforms...')
        self.logger.msg_str(train_transform)
        self.logger.msg_str('set train and unlabeled dataloaders...')
        train_loader, labels, train_sampler = self.build_dataloader(dataset_name=opt.dataset, transform=train_transform, batch_size=opt.batch_size, shuffle=True, drop_last=True, sampler=True, train=True)
        self.logger.msg_str(f'set train_loader with {len(train_loader)} iterations...')
        unlabeled_loader, _, unlabeled_sampler = self.build_dataloader(dataset_name=opt.dataset, transform=train_transform, batch_size=opt.batch_size, shuffle=True, drop_last=True, sampler=True, train=False, unlabeled=True)
        self.logger.msg_str(f'set unlabeled_loader with {len(unlabeled_loader)} iterations...')
        test_transform = transforms.Compose([GrainResize(opt.img_size), transforms.ToTensor(), normalize])
        self.logger.msg_str('set memory dataloaders...')
        memory_loader = self.build_dataloader(opt.dataset, test_transform, train=False, batch_size=opt.batch_size, sampler=True)[0]
        self.logger.msg_str(f'set memory_loader with {len(memory_loader)} iterations...')
        self.logger.msg_str(test_transform)
        self.train_loader = train_loader
        self.memory_loader = memory_loader
        self.unlabeled_loader = unlabeled_loader
        self.train_sampler = train_sampler
        self.unlabeled_sampler = unlabeled_sampler
        self.iter_per_epoch = len(train_loader) + len(unlabeled_loader)
        self.num_samples = len(labels)
        self.indices = torch.zeros(len(self.train_sampler), dtype=torch.long).cuda()
        self.num_cluster = int(opt.num_cluster)
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()
        self.logger.msg_str('load {} images...'.format(self.num_samples))

    def fit(self):
        opt = self.opt
        self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs, disable=not self.verbose)
        n_iter = self.iter_per_epoch * opt.resume_epoch + 1
        self.progress_bar.update(n_iter)
        i = 0
        while True:
            epoch = int(n_iter // self.iter_per_epoch + 1)
            self.train_sampler.set_epoch(epoch)
            self.unlabeled_sampler.set_epoch(epoch)
            for inputs in self.unlabeled_loader:
                inputs = convert_to_cuda(inputs)
                self.train_unlabeled(inputs, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1
            apply_kmeans = epoch % opt.reassign == 0
            if apply_kmeans:
                self.psedo_labeling(n_iter)
            self.indices.copy_(torch.Tensor(list(iter(self.train_sampler))))
            for inputs in self.train_loader:
                inputs = convert_to_cuda(inputs)
                self.adjust_learning_rate(n_iter)
                self.train(inputs, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                n_iter += 1
            if i < len(opt.epochs_cluster_analysis) and opt.clusternet:
                if self.cur_epoch == opt.epochs_cluster_analysis[i]:
                    self.get_new_k(split=opt.clusternet_training_data, index=i)
                    i += 1
            self.cur_epoch += 1
            if self.cur_epoch > opt.epochs:
                self.progress_bar.close()
                self.export_final_artifacts()
                break

    def train(self, inputs, n_iter):
        opt = self.opt
        images, _ = inputs
        self.byol.train()
        im_q = images[0][0]
        im_k = images[0][1]
        _start = ((n_iter - 1) % self.iter_per_epoch - len(self.unlabeled_loader)) * opt.batch_size
        indices = self.indices[_start:_start + opt.batch_size]
        self.byol.module.psedo_labels = self.psedo_labels
        self.byol.module.num_cluster = self.num_cluster
        is_warmup = not self.cur_epoch > opt.warmup_epochs
        self.byol.module.latent_std = opt.latent_std * float(not is_warmup)
        with torch.autocast('cuda', enabled=opt.amp):
            contrastive_loss, cluster_loss_batch, q = self.byol(im_q, im_k, indices, True, opt.v2)
        loss = contrastive_loss
        if scattering_is_active(n_iter, self.iter_per_epoch, opt.warmup_epochs):
            loss += cluster_loss_batch * opt.cluster_loss_weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            q_std = torch.std(q.detach(), dim=0).mean()
        outputs = [contrastive_loss, cluster_loss_batch, q_std]
        self.logger.msg(outputs, n_iter)

    def train_unlabeled(self, inputs, n_iter):
        opt = self.opt
        images, _ = inputs
        self.byol.train()
        im_q, im_k = images
        _start = ((n_iter - 1) % self.iter_per_epoch - len(self.unlabeled_loader)) * opt.batch_size
        indices = self.indices[_start:_start + opt.batch_size]
        self.byol.module.psedo_labels = self.psedo_labels
        self.byol.module.num_cluster = self.num_cluster
        with torch.autocast('cuda', enabled=opt.amp):
            unlabeled_contrastive_loss, _, q = self.byol(im_q, im_k, indices, True, opt.v2, unlabeled=True)
        self.optimizer.zero_grad()
        unlabeled_contrastive_loss.backward()
        self.optimizer.step()
        outputs = [unlabeled_contrastive_loss]
        self.logger.msg(outputs, n_iter)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        lr = self.cosine_annealing_LR(n_iter)
        if opt.fix_predictor_lr:
            predictor_lr = opt.learning_rate
        else:
            predictor_lr = lr * opt.lambda_predictor_lr
        flag = False
        for param_group in self.optimizer.param_groups:
            if 'predictor' in param_group['name']:
                flag = True
                param_group['lr'] = predictor_lr
            else:
                param_group['lr'] = lr
        assert flag
        ema_momentum = opt.momentum_base
        if opt.momentum_increase:
            ema_momentum = opt.momentum_max - (opt.momentum_max - ema_momentum) * (np.cos(np.pi * n_iter / (opt.epochs * self.iter_per_epoch)) + 1) / 2
        self.byol.module.m = ema_momentum
        self.logger.msg([lr, predictor_lr, ema_momentum], n_iter)
