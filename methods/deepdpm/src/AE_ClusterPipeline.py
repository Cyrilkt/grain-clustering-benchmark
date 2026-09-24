# Copyright (c) 2022 Meitar Ronen
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from src.clustering_models.clusternet import ClusterNet
from src.feature_extractors.feature_extractor import FeatureExtractor

class AE_ClusterPipeline(pl.LightningModule):

    def __init__(self, logger, args, input_dim, data_to_emb, labels):
        super(AE_ClusterPipeline, self).__init__()
        self.args = args
        self.pretrain_logger = logger
        self.args = args
        self.beta = args.beta
        self.lambda_ = args.lambda_
        self.n_clusters = self.args.n_clusters
        self.device_name = f'cuda:{args.gpus}' if torch.cuda.is_available() and args.gpus is not None else 'cpu'
        self.data_to_emb = torch.as_tensor(data_to_emb).to(self.device_name)
        print('TRAINING DEVICE:', self.device_name)
        self.labels = labels
        if self.args.seed is not None:
            pl.utilities.seed.seed_everything(self.args.seed)
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))
        if not self.lambda_ > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lambda_))
        if len(self.args.hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')
        self.feature_extractor = FeatureExtractor(args, input_dim)
        self.args.latent_dim = self.feature_extractor.latent_dim
        self.criterion = nn.MSELoss(reduction='sum')
        self.clustering = ClusterNet(args, self)
        self.init_clusternet_num = 0
        # Preserve the native NumPy RNG progression after removing visualization setup.
        np.random.rand(100, 3)

    def configure_optimizers(self):
        """Configure the optimizers of the AE model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def _loss(self, X, latent_X, cluster_assignment):
        """Compute batch loss

        Args:
            X ([torch.tensor]): The current batch of data ([N, D])
            cluster_assignment ([torch.tensor]): (soft) cluster assignments for each data point

        Returns:
            Tuple: the loss (reconstruction + distance loss) and both losses seperately for logging
        """
        batch_size = X.size()[0]
        rec_X = self.feature_extractor.decode(latent_X)
        X = self.feature_extractor.extract_features(X) if self.feature_extractor.feature_extractor else X
        rec_loss = self.lambda_ * self.criterion(X, rec_X)
        if self.args.regularization == 'dist_loss':
            dist_loss = torch.tensor(0.0).to(self.device_name)
            clusters = torch.FloatTensor(self.clustering.clusters).to(self.device_name)
            for i in range(batch_size):
                diff_vec = latent_X[i] - clusters[cluster_assignment.argmax(-1)[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
                dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)
            reg_loss = dist_loss
        elif self.args.regularization == 'cluster_loss':
            mus, covs, pi, K = self.clustering.get_model_params()
            reg_loss = self.clustering.model.cluster_model.training_utils.cluster_loss_function(latent_X.detach(), cluster_assignment, model_mus=torch.from_numpy(mus), K=K, codes_dim=self.args.latent_dim, model_covs=torch.from_numpy(covs) if self.args.cluster_loss in ('diag_NIG', 'KL_GMM_2') else None, pi=torch.from_numpy(pi)) * batch_size
        return (rec_loss + reg_loss, rec_loss.detach(), reg_loss.detach())

    def _init_clusters(self, verbose=True, centers=None):
        if verbose:
            print(f'========== Alternation {self.init_clusternet_num}: Running DeepDPM clustering ==========\n')
        if self.args.clustering == 'cluster_net':
            self.clustering.init_cluster(self.train_dataloader(), self.val_dataloader(), logger=self.logger, centers=centers, init_num=self.init_clusternet_num)
            self.n_clusters = self.clustering.n_clusters
            if self.args.save_checkpoints:
                print('Checkpoint ...')
                save_dict = {}
                clustering_module = self.clustering.model.cluster_model
                if len(clustering_module.optimizers_dict_idx) > 1:
                    for key, value in clustering_module.optimizers_dict_idx.items():
                        save_dict[key] = clustering_module.optimizers()[value].state_dict()
                else:
                    save_dict['clusternet_opt'] = clustering_module.optimizers().state_dict()
                save_dict['model'] = (clustering_module.state_dict(),)
                save_dict['K'] = (clustering_module.K,)
                save_dict['epoch'] = clustering_module.current_epoch
                save_dict['alt_num'] = self.init_clusternet_num
                torch.save(save_dict, f'./saved_models/{self.args.dataset}/{self.args.exp_name}/alt_{self.init_clusternet_num}_checkpoint.pth.tar')
            self.init_clusternet_num += 1
            self.log('alt_num', self.init_clusternet_num)
        else:
            batch_X, batch_Y = ([], [])
            for data, labels in self.train_dataloader():
                batch_size = data.size()[0]
                data = data.to(self.device_name).view(batch_size, -1)
                latent_X = self.feature_extractor(data, latent=True)
                batch_X.append(latent_X.detach().cpu().numpy())
                batch_Y.append(labels)
            batch_X = np.vstack(batch_X)
            batch_Y = torch.cat(batch_Y)
            self.clustering.init_cluster(batch_X)
        if verbose:
            print('========== End initializing clusters ==========\n')

    def _comp_clusters(self, *args, **kwargs):
        raise RuntimeError('Additional AE/clustering alternations are disabled in the Grain benchmark')

    def _pre_step(self, x, y=None):
        if len(self.sampled_codes) < 10000:
            with torch.no_grad():
                latent_X = self.feature_extractor(x, latent=True)
                self.sampled_codes = torch.cat([self.sampled_codes, latent_X.detach().cpu()])
                self.sampled_gt = torch.cat([self.sampled_gt, y.cpu()])
        rec_X = self.feature_extractor(x)
        x = self.feature_extractor.extract_features(x) if self.feature_extractor.feature_extractor else x
        loss = self.criterion(x, rec_X)
        return loss

    def _step(self, x, y):
        """Implementing one optimization step.
        1. gets the latent features using a forward pass on the AE
        2. gets the cluster assignments for the latent features (the closest cluster id to each sample is chosen)
        3. updates the clusters centers in an online fashion (eta = 1 / (N_k + x_i) , (1-eta)*mu + eta*x_i)
        4. Compute and return loss

        Args:
            x (torch.tensor): batch [N, D]

        Returns:
            loss and losses for logging
        """
        latent_X, cluster_assign = self(x)
        if len(self.sampled_codes) < 10000:
            self.sampled_codes = torch.cat([self.sampled_codes, latent_X.detach().cpu()])
            self.sampled_gt = torch.cat([self.sampled_gt, y.cpu()])
        if self.args.update_clusters_params != 'False':
            self._update_clusters(latent_X, cluster_assign)
        loss, rec_loss, dist_loss = self._loss(x, latent_X, cluster_assign)
        return (loss, rec_loss, dist_loss)

    def _update_clusters(self, latent_X, cluster_assign):
        if self.args.update_clusters_params == 'only_centers':
            elem_count = cluster_assign.sum(axis=0)
            for k in range(self.n_clusters):
                if elem_count[k] == 0:
                    continue
                self.clustering.update_cluster_center(latent_X.detach().cpu().numpy(), k, cluster_assign.detach().cpu().numpy())

    def forward(self, x, latent=False):
        latent_X = self.feature_extractor(x, latent=True)
        if latent:
            return latent_X.detach().cpu().numpy()
        if len(latent_X.size()) > 2:
            latent_X = latent_X.view(latent_X.size(0), -1)
        if self.args.cluster_assignments != 'pseudo_label':
            return (latent_X, self.clustering.update_assign(latent_X, self.args.cluster_assignments))
        else:
            return 0

    def on_train_epoch_start(self):
        self.sampled_codes = torch.empty(0)
        self.sampled_gt = torch.empty(0)
        self.pretrain = True

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        if self.pretrain:
            stage = 'val_pretrain'
            loss = self._pre_step(x, y)
            rec_loss = loss
            dist_loss = torch.tensor([0.0])
        else:
            stage = 'val'
            loss, rec_loss, dist_loss = self._step(x, y)
        self.log(f'{stage}/loss', loss)
        self.log(f'{stage}/reconstruction_loss', rec_loss)
        self.log(f'{stage}/dist_loss', dist_loss)
        _, assign = self(x)
        y_pred = assign.argmax(-1).cpu().numpy()
        return {'loss': loss, 'y_gt': y, 'y_pred': y_pred}

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        if self.pretrain:
            stage = 'pretrain'
            if self.args.pretrain_noise_factor > 0:
                x = x + self.args.pretrain_noise_factor * torch.randn(*x.shape).to(device=self.device_name)
                x = np.clip(x.cpu(), 0.0, 1.0).to(device=self.device_name)
            loss = self._pre_step(x, y)
            rec_loss = loss
            dist_loss = torch.tensor([0.0])
        else:
            stage = 'train'
            loss, rec_loss, dist_loss = self._step(x, y)
        self.log(f'{stage}/loss', loss)
        self.log(f'{stage}/reconstruction_loss', rec_loss)
        self.log(f'{stage}/dist_loss', dist_loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        losses, y_gt, y_pred = ([], [], [])
        for out_dict in validation_step_outputs:
            losses.append(out_dict['loss'])
            y_gt += list(out_dict['y_gt'].cpu().numpy())
            y_pred += list(out_dict['y_pred'])
        avg_loss = torch.tensor(losses).mean()
        y_pred = np.vstack(y_pred).reshape(-1)

