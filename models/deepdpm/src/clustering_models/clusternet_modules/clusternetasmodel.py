# Copyright (c) 2022 Meitar Ronen
import numpy as np
import torch
from torch import optim
import pytorch_lightning as pl
from torch.distributions.constraints import positive_definite
from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.clustering_utils.priors import Priors
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import init_mus_and_covs, compute_data_covs_hard_assignment
from src.clustering_models.clusternet_modules.utils.clustering_utils.split_merge_operations import update_models_parameters_split, split_step, merge_step, update_models_parameters_merge
from src.clustering_models.clusternet_modules.models.Classifiers import MLP_Classifier, Subclustering_net

def ensure_positive_definite(cov, initial_jitter=1e-05, max_iter=10, epsilon=1e-06):
    """
    Adjusts a covariance matrix to be positive definite.
    First attempts iterative jittering (with symmetry enforcement),
    then falls back to eigenvalue clipping if needed.
    
    Args:
        cov (torch.Tensor): The covariance matrix.
        initial_jitter (float): Starting jitter value for the diagonal.
        max_iter (int): Maximum iterations for jitter.
        epsilon (float): Minimum eigenvalue threshold for clipping.
        
    Returns:
        torch.Tensor: A positive definite covariance matrix.
        
    Raises:
        ValueError: If adjustments fail to produce a positive definite matrix.
    """
    print(' ensure positive definite loop')
    cov = (cov + cov.T) / 2.0
    jitter = initial_jitter
    I = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    for i in range(max_iter):
        if positive_definite.check(cov):
            return cov
        else:
            cov = cov + jitter * I
            jitter *= 10
    print('Jittering did not succeed. Falling back to eigenvalue clipping.')
    eigvals, eigvecs = torch.linalg.eigh(cov)
    print('Eigenvalues before clipping:', eigvals)
    eigvals_clipped = torch.clamp(eigvals, min=epsilon)
    cov_clipped = eigvecs * eigvals_clipped @ eigvecs.T
    cov_clipped = (cov_clipped + cov_clipped.T) / 2.0
    if positive_definite.check(cov_clipped):
        return cov_clipped
    else:
        raise ValueError('Covariance matrix could not be made positive definite after jittering and eigenvalue clipping.')

class ClusterNetModel(pl.LightningModule):

    def __init__(self, hparams, input_dim, init_k, feature_extractor=None, n_sub=2, centers=None, init_num=0):
        """The main class of the unsupervised clustering scheme.
        Performs all the training steps.

        Args:
            hparams ([namespace]): model-specific hyperparameters
            input_dim (int): the shape of the input data
            train_dl (DataLoader): The dataloader to train on
            init_k (int): The initial K to start the net with
            feature_extractor (nn.Module): The feature extractor to get codes with
            n_sub (int, optional): Number of subclusters per cluster. Defaults to 2.

        """
        super().__init__()
        self.hparams = hparams
        self.K = init_k
        self.n_sub = n_sub
        self.codes_dim = input_dim
        self.split_performed = False
        self.merge_performed = False
        self.feature_extractor = feature_extractor
        self.centers = centers
        if self.hparams.seed:
            pl.utilities.seed.seed_everything(self.hparams.seed)
        self.cluster_net = MLP_Classifier(hparams, k=self.K, codes_dim=self.codes_dim)
        if not self.hparams.ignore_subclusters:
            self.subclustering_net = Subclustering_net(hparams, codes_dim=self.codes_dim, k=self.K)
        else:
            self.subclustering_net = None
        self.last_key = self.K - 1
        self.training_utils = training_utils(hparams)
        self.init_num = init_num
        self.prior_sigma_scale = self.hparams.prior_sigma_scale
        if self.init_num > 0 and self.hparams.prior_sigma_scale_step != 0:
            self.prior_sigma_scale = self.hparams.prior_sigma_scale / (self.init_num * self.hparams.prior_sigma_scale_step)
        self.use_priors = self.hparams.use_priors
        self.prior = Priors(hparams, K=self.K, codes_dim=self.codes_dim, prior_sigma_scale=self.prior_sigma_scale)
        self.mus_inds_to_merge = None
        self.mus_ind_to_split = None

    def forward(self, x):
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(self.feature_extractor(x.view(x.size()[0], -1), latent=True)).to(device=self.device)
                print('x.view(x.size()[0], -1) :', x.view(x.size()[0], -1).size())
                print('FORWARD CLUSTERNET ', codes.size())
                print('x : ', x.size())
        else:
            codes = x
        return self.cluster_net(codes)

    def on_train_epoch_start(self):
        self.current_training_stage = 'gather_codes' if self.current_epoch == 0 and (not hasattr(self, 'mus')) else 'train_cluster_net'
        self.initialize_net_params(stage='train')
        if self.split_performed or self.merge_performed:
            self.split_performed = False
            self.merge_performed = False

    def on_validation_epoch_start(self):
        self.initialize_net_params(stage='val')
        return super().on_validation_epoch_start()

    def initialize_net_params(self, stage='train'):
        self.codes = []
        if stage == 'train':
            if self.current_epoch > 0:
                del self.train_resp, self.train_resp_sub, self.train_gt
            self.train_resp = []
            self.train_resp_sub = []
            self.train_gt = []
        else:
            if self.current_epoch > 0:
                del self.val_resp, self.val_resp_sub, self.val_gt
            self.val_resp = []
            self.val_resp_sub = []
            self.val_gt = []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(self.feature_extractor(x.view(x.size()[0], -1), latent=True)).to(device=self.device)
        else:
            codes = x
        if self.current_training_stage == 'gather_codes':
            return self.only_gather_codes(codes, y, optimizer_idx)
        elif self.current_training_stage == 'train_cluster_net':
            return self.cluster_net_pretraining(codes, y, optimizer_idx, x if batch_idx == 0 else None)
        else:
            raise NotImplementedError()

    def only_gather_codes(self, codes, y, optimizer_idx):
        """Only log codes for initialization

        Args:
            codes ([type]): The input data in the latent space
            y ([type]): The ground truth labels
            optimizer_idx ([type]): The optimizer index
        """
        if optimizer_idx == self.optimizers_dict_idx['cluster_net_opt']:
            self.codes, self.train_gt, _, _ = self.training_utils.log_codes_and_responses(model_codes=self.codes, model_gt=self.train_gt, model_resp=self.train_resp, model_resp_sub=self.train_resp_sub, codes=codes, y=y, logits=None)
        return None

    def cluster_net_pretraining(self, codes, y, optimizer_idx, x_for_vis=None):
        """Pretraining function for the clustering and subclustering nets.
        At this stage, the only loss is the cluster and subcluster loss. The autoencoder weights are held constant.

        Args:
            codes ([type]): The encoded data samples
            y: The ground truth labels
            optimizer_idx ([type]): The pytorch optimizer index
        """
        codes = codes.view(-1, self.codes_dim)
        logits = self.cluster_net(codes)
        if self.hparams.cluster_loss in ('diag_NIG', 'KL_GMM_2'):
            fixed_covs = []
            for cov in self.covs:
                if not positive_definite.check(cov):
                    cov = ensure_positive_definite(cov)
                fixed_covs.append(cov)
            covs_for_loss = torch.stack(fixed_covs)
        else:
            covs_for_loss = None
        cluster_loss = self.training_utils.cluster_loss_function(codes, logits, model_mus=self.mus, K=self.K, codes_dim=self.codes_dim, model_covs=covs_for_loss, pi=self.pi, logger=self.logger)
        self.log('cluster_net_train/train/cluster_loss', self.hparams.cluster_loss_weight * cluster_loss, on_step=True, on_epoch=False)
        loss = self.hparams.cluster_loss_weight * cluster_loss
        if not self.hparams.ignore_subclusters and optimizer_idx == self.optimizers_dict_idx['subcluster_net_opt']:
            logits = logits.detach()
            if self.hparams.start_sub_clustering <= self.current_epoch:
                sublogits = self.subcluster(codes, logits)
                subcluster_loss = self.training_utils.subcluster_loss_function_new(codes, logits, sublogits, self.K, self.n_sub, self.mus_sub, covs_sub=self.covs_sub if self.hparams.subcluster_loss in ('diag_NIG', 'KL_GMM_2') else None, pis_sub=self.pi_sub)
                self.log('cluster_net_train/train/subcluster_loss', self.hparams.subcluster_loss_weight * subcluster_loss, on_step=True, on_epoch=True)
                loss = self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                sublogits = None
                loss = None
        else:
            sublogits = None
        if optimizer_idx == len(self.optimizers_dict_idx) - 1:
            self.codes, self.train_gt, self.train_resp, self.train_resp_sub = self.training_utils.log_codes_and_responses(self.codes, self.train_gt, self.train_resp, self.train_resp_sub, codes, logits.detach(), y, sublogits=sublogits)
        if loss is not None:
            return loss
        else:
            return None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(self.feature_extractor(x.view(x.size()[0], -1), latent=True)).to(device=self.device)
        else:
            codes = x
        logits = self.cluster_net(codes)
        if batch_idx == 0 and (self.current_epoch < 5 or self.current_epoch % 50 == 0):
            pass
        if self.current_training_stage != 'gather_codes':
            cluster_loss = self.training_utils.cluster_loss_function(codes.view(-1, self.codes_dim), logits, model_mus=self.mus, K=self.K, codes_dim=self.codes_dim, model_covs=self.covs if self.hparams.cluster_loss in ('diag_NIG', 'KL_GMM_2') else None, pi=self.pi)
            loss = self.hparams.cluster_loss_weight * cluster_loss
            self.log('cluster_net_train/val/cluster_loss', loss)
            if self.current_epoch >= self.hparams.start_sub_clustering and (not self.hparams.ignore_subclusters):
                subclusters = self.subcluster(codes, logits)
                subcluster_loss = self.training_utils.subcluster_loss_function_new(codes.view(-1, self.codes_dim), logits, subclusters, self.K, self.n_sub, self.mus_sub, covs_sub=self.covs_sub if self.hparams.subcluster_loss in ('diag_NIG', 'KL_GMM_2') else None, pis_sub=self.pi_sub)
                self.log('cluster_net_train/val/subcluster_loss', subcluster_loss)
                loss += self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                subclusters = None
        else:
            loss = torch.tensor(1.0)
            subclusters = None
            logits = None
        self.codes, self.val_gt, self.val_resp, self.val_resp_sub = self.training_utils.log_codes_and_responses(self.codes, self.val_gt, self.val_resp, model_resp_sub=self.val_resp_sub, codes=codes, logits=logits, y=y, sublogits=subclusters, stage='val')
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """Perform logging operations and computes the clusters' and the subclusters' centers.
        Also perform split and merges steps

        Args:
            outputs ([type]): [description]
        """
        if self.current_training_stage == 'gather_codes':
            # Preserve the native NumPy RNG progression after removing visualization setup.
            np.random.rand(100, 3)
            self.prior.init_priors(self.codes.view(-1, self.codes_dim))
            if self.centers is not None:
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
            else:
                self.freeze_mus_after_init_until = 0
                self.mus, self.covs, self.pi, init_labels = init_mus_and_covs(codes=self.codes.view(-1, self.codes_dim), K=self.K, how_to_init_mu=self.hparams.how_to_init_mu, logits=self.train_resp, use_priors=self.hparams.use_priors, prior=self.prior, random_state=0, device=self.device)
        else:
            if not self.hparams.ignore_subclusters:
                clus_losses, subclus_losses = (outputs[0], outputs[1])
            else:
                clus_losses = outputs
            avg_clus_loss = torch.stack([x['loss'] for x in clus_losses]).mean()
            self.log('cluster_net_train/train/avg_cluster_loss', avg_clus_loss)
            if self.current_epoch >= self.hparams.start_sub_clustering and (not self.hparams.ignore_subclusters):
                avg_subclus_loss = torch.stack([x['loss'] for x in subclus_losses]).mean()
                self.log('cluster_net_train/train/avg_subcluster_loss', avg_subclus_loss)
            perform_split = self.training_utils.should_perform_split(self.current_epoch) and self.centers is None
            perform_merge = self.training_utils.should_perform_merge(self.current_epoch, self.split_performed) and self.centers is None
            if self.centers is not None:
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
            freeze_mus = self.training_utils.freeze_mus(self.current_epoch, self.split_performed) or self.current_epoch <= self.freeze_mus_after_init_until
            if not freeze_mus:
                self.pi, self.mus, self.covs = self.training_utils.comp_cluster_params(self.train_resp, self.codes.view(-1, self.codes_dim), self.pi, self.K, self.prior)
            if self.hparams.start_sub_clustering == self.current_epoch + 1 or (self.hparams.ignore_subclusters and (perform_split or perform_merge)):
                self.pi_sub, self.mus_sub, self.covs_sub = self.training_utils.init_subcluster_params(self.train_resp, self.train_resp_sub, self.codes.view(-1, self.codes_dim), self.K, self.n_sub, self.prior)
            elif self.hparams.start_sub_clustering <= self.current_epoch and (not freeze_mus) and (not self.hparams.ignore_subclusters):
                self.pi_sub, self.mus_sub, self.covs_sub = self.training_utils.comp_subcluster_params(self.train_resp, self.train_resp_sub, self.codes, self.K, self.n_sub, self.mus_sub, self.covs_sub, self.pi_sub, self.prior)
            if perform_split and (not freeze_mus):
                self.training_utils.last_performed = 'split'
                split_decisions = split_step(self.K, self.codes, self.train_resp, self.train_resp_sub, self.mus, self.mus_sub, self.hparams.cov_const, self.hparams.alpha, self.hparams.split_prob, self.prior, self.hparams.ignore_subclusters)
                if split_decisions.any():
                    self.split_performed = True
                    self.perform_split_operations(split_decisions)
            if perform_merge and (not freeze_mus):
                self.training_utils.last_performed = 'merge'
                mus_to_merge, highest_ll_mus = merge_step(self.mus, self.train_resp, self.codes, self.K, self.hparams.raise_merge_proposals, self.hparams.cov_const, self.hparams.alpha, self.hparams.merge_prob, prior=self.prior)
                if len(mus_to_merge) > 0:
                    self.merge_performed = True
                    self.perform_merge(mus_to_merge, highest_ll_mus)
            with torch.no_grad():
                pass
        if self.split_performed or self.merge_performed:
            self.update_params_split_merge()
            print('Current number of clusters: ', self.K)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('cluster_net_train/val/avg_val_loss', avg_loss)
        if self.current_epoch > self.hparams.start_sub_clustering and (self.current_epoch % 50 == 0 or self.current_epoch == self.hparams.train_cluster_net - 1):
            from pytorch_lightning.loggers.base import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                pass

    def subcluster(self, codes, logits, hard_assignment=True):
        sub_clus_resp = self.subclustering_net(codes)
        z = logits.argmax(-1)
        mask = torch.zeros_like(sub_clus_resp)
        mask[np.arange(len(z)), 2 * z] = 1.0
        mask[np.arange(len(z)), 2 * z + 1] = 1.0
        sub_clus_resp = torch.nn.functional.softmax(sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm, dim=1)
        return sub_clus_resp

    def update_subcluster_net_split(self, split_decisions):
        subclus_opt = self.optimizers()[self.optimizers_dict_idx['subcluster_net_opt']]
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_split(split_decisions, self.hparams.split_init_weights_sub)
        subclus_opt.param_groups[0]['params'] = list(self.subclustering_net.parameters())

    def perform_split_operations(self, split_decisions):
        print('perform_split_operations function clusternetasmodel :', self.current_epoch)
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx['cluster_net_opt']]
        else:
            clus_opt = self.optimizers()
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        self.cluster_net.update_K_split(split_decisions, self.hparams.init_new_weights, self.subclustering_net)
        clus_opt.param_groups[1]['params'] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        mus_ind_to_split = torch.nonzero(torch.tensor(split_decisions), as_tuple=False)
        self.mus_new, self.covs_new, self.pi_new, self.mus_sub_new, self.covs_sub_new, self.pi_sub_new = update_models_parameters_split(split_decisions, self.mus, self.covs, self.pi, mus_ind_to_split, self.mus_sub, self.covs_sub, self.pi_sub, self.codes, self.train_resp, self.train_resp_sub, self.n_sub, self.hparams.how_to_init_mu_sub, self.prior, use_priors=self.hparams.use_priors)
        self.K += len(mus_ind_to_split)
        if not self.hparams.ignore_subclusters:
            self.update_subcluster_net_split(split_decisions)
        self.mus_ind_to_split = mus_ind_to_split

    def update_subcluster_nets_merge(self, merge_decisions, pairs_to_merge, highest_ll):
        subclus_opt = self.optimizers()[self.optimizers_dict_idx['subcluster_net_opt']]
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_merge(merge_decisions, pairs_to_merge=pairs_to_merge, highest_ll=highest_ll, init_new_weights=self.hparams.merge_init_weights_sub)
        subclus_opt.param_groups[0]['params'] = list(self.subclustering_net.parameters())

    def perform_merge(self, mus_lists_to_merge, highest_ll_mus, use_priors=True):
        """A method that performs merges of clusters' centers

        Args:
            mus_lists_to_merge (list): a list of lists, each one contains 2 indices of mus that were chosen to be merged.
            highest_ll_mus ([type]): a list of the highest log likelihood index for each pair of mus
        """
        print(f'Merging clusters {mus_lists_to_merge}')
        mus_lists_to_merge = torch.tensor(mus_lists_to_merge)
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        inds_to_mask[mus_lists_to_merge.flatten()] = 1
        self.mus_new, self.covs_new, self.pi_new, self.mus_sub_new, self.covs_sub_new, self.pi_sub_new = update_models_parameters_merge(mus_lists_to_merge, inds_to_mask, self.K, self.mus, self.covs, self.pi, self.mus_sub, self.covs_sub, self.pi_sub, self.codes, self.train_resp, self.prior, use_priors=self.hparams.use_priors, n_sub=self.n_sub, how_to_init_mu_sub=self.hparams.how_to_init_mu_sub)
        self.K -= len(highest_ll_mus)
        if not self.hparams.ignore_subclusters:
            self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus)
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx['cluster_net_opt']]
        else:
            clus_opt = self.optimizers()
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        self.cluster_net.update_K_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus, init_new_weights=self.hparams.init_new_weights)
        clus_opt.param_groups[1]['params'] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        self.mus_inds_to_merge = mus_lists_to_merge

    def configure_optimizers(self):
        cluster_params = torch.nn.ParameterList([p for n, p in self.cluster_net.named_parameters() if 'class_fc2' not in n])
        cluster_net_opt = optim.Adam(cluster_params, lr=self.hparams.cluster_lr)
        cluster_net_opt.add_param_group({'params': self.cluster_net.class_fc2.parameters()})
        self.optimizers_dict_idx = {'cluster_net_opt': 0}
        if self.hparams.lr_scheduler == 'StepLR':
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
            print('StepLR')
        elif self.hparams.lr_scheduler == 'ReduceOnP':
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode='min', factor=0.5, patience=4)
            print('ReduceOnP')
        else:
            cluster_scheduler = None
            print('NO OPTIMIZER')
        if not self.hparams.ignore_subclusters:
            sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr)
            self.optimizers_dict_idx['subcluster_net_opt'] = 1
            return ({'optimizer': cluster_net_opt, 'scheduler': cluster_scheduler, 'monitor': 'cluster_net_train/val/cluster_loss'}, {'optimizer': sub_clus_opt})
        return {'optimizer': cluster_net_opt, 'scheduler': cluster_scheduler, 'monitor': 'cluster_net_train/val/cluster_loss'} if cluster_scheduler else cluster_net_opt

    def update_params_split_merge(self):
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.mus_sub = self.mus_sub_new
        self.covs_sub = self.covs_sub_new
        self.pi = self.pi_new
        self.pi_sub = self.pi_sub_new

    def init_covs_and_pis_given_mus(self):
        if self.hparams.use_priors_for_net_params_init:
            _, cov_prior = self.prior.init_priors(self.mus)
            self.covs = torch.stack([cov_prior for k in range(self.K)])
            p_counts = torch.ones(self.K) * 10
            self.pi = p_counts / float(self.K * 10)
        else:
            dis_mat = torch.empty((len(self.codes), self.K))
            for i in range(self.K):
                dis_mat[:, i] = torch.sqrt(((self.codes - self.mus[i]) ** 2).sum(axis=1))
            hard_assign = torch.argmin(dis_mat, dim=1)
            vals, counts = torch.unique(hard_assign, return_counts=True)
            if len(counts) < self.K:
                new_counts = []
                for k in range(self.K):
                    if k in vals:
                        new_counts.append(counts[vals == k])
                    else:
                        new_counts.append(0)
                counts = torch.tensor(new_counts)
            pi = counts / float(len(self.codes))
            data_covs = compute_data_covs_hard_assignment(hard_assign.numpy(), self.codes, self.K, self.mus.cpu(), self.prior)
            if self.use_priors:
                covs = []
                for k in range(self.K):
                    codes_k = self.codes[hard_assign == k]
                    cov_k = self.prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])
                    covs.append(cov_k)
                covs = torch.stack(covs)
            self.covs = covs
            self.pi = pi

