"""DIVA VAE and Dirichlet-mixture updates for Grain features."""
from abc import abstractmethod
from typing import List, Any, TypeVar
from itertools import cycle
import os
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import bnpy
from bnpy.data.XData import XData
Tensor = TypeVar('torch.tensor')

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class PTTensorDataset(Dataset):
    """One features.npy matrix; dummy labels only, never reference labels.
    The legacy class name is retained to minimize changes in native callers.
    """

    def __init__(self, data_path: str, train: bool=True):
        from pathlib import Path
        path = Path(data_path) / 'features.npy'
        self.data = torch.from_numpy(np.load(path, allow_pickle=False)).float()
        self.labels = torch.zeros(len(self.data), dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = int(self.labels[idx])
        return (x, y)

class DIVA_MLP(BaseVAE):

    def __init__(self, input_dim: int, latent_dim: int, dpmm_param: dict, output_type: str='linear', **kwargs) -> None:
        super(DIVA_MLP, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.dpmm_param = dpmm_param
        self.output_type = output_type
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU())
        self.fc_mu = nn.Linear(2000, latent_dim)
        self.fc_log_var = nn.Linear(2000, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, self.input_dim))
        self.bnp_model = False
        self.bnp_info_dict = None
        pwd = os.getcwd()
        self.bnp_root = pwd + '/save/bn_model/'
        self.bnp_iterator = cycle(range(2))

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        if self.output_type == 'linear':
            pass
        elif self.output_type == 'sigmoid':
            result = torch.sigmoid(result)
        else:
            result = torch.tanh(result)
        return result

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        kld_weight = kwargs['M_N']
        if not self.bnp_model:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld_loss': kld_loss, 'z': z}
        else:
            prob_comps, comps = self.cluster_assignments(z)
            var = torch.exp(0.5 * log_var) ** 2
            dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(var))
            B, K = prob_comps.shape
            kld = torch.zeros(B).to(mu.device)
            for k in range(K):
                prob_k = prob_comps[:, k]
                dist_k = torch.distributions.MultivariateNormal(loc=self.comp_mu[k].to(mu.device), covariance_matrix=torch.diag_embed(self.comp_var[k]).to(mu.device))
                expanded_dist_k = dist_k.expand(dist.batch_shape)
                kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)
                kld += torch.from_numpy(prob_k).to(mu.device) * kld_k
            kld_loss = torch.mean(kld)
            loss = recons_loss + kld_weight * kld_loss
            loss = loss.to(input.device)
            return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld_loss': kld_loss, 'z': z, 'comps': comps}

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def fit_dpmm(self, z):
        z = XData(z.detach().cpu().numpy())
        if not self.bnp_model:
            print('Initialing DPMM model ...')
            self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', output_path=self.bnp_root + str(next(self.bnp_iterator)), initname='randexamples', K=1, gamma0=1.0, sF=self.dpmm_param['initial_sF'], ECovMat='eye', b_Kfresh=5, b_startLap=0, m_startLap=2, moves='birth,delete,merge,shuffle', nLap=2, b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'], b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'], b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'])
        else:
            self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', output_path=self.bnp_root + str(next(self.bnp_iterator)), initname=self.bnp_info_dict['task_output_path'], K=self.bnp_info_dict['K_history'][-1], gamma0=1.0, sF=self.dpmm_param['sF'], b_Kfresh=5, b_startLap=1, m_startLap=2, moves='birth,delete,merge,shuffle', nLap=2, b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'], b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'], b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'])
        self.calc_cluster_component_params()

    def calc_cluster_component_params(self):
        self.comp_mu = [torch.Tensor(self.bnp_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, self.bnp_model.obsModel.K)]
        self.comp_var = [torch.Tensor(np.sum(self.bnp_model.obsModel.get_covar_mat_for_comp(i), axis=0)) for i in np.arange(0, self.bnp_model.obsModel.K)]

    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.bnp_model.calc_local_params(z)
        resp = LP['resp']
        Z = resp.argmax(axis=1)
        return (resp, Z)

def data_loader(fn):

    def func_wrapper(self):
        return fn(self)
    return func_wrapper

class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.assignment = None
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.dpmm_init_epoch = 0

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'], optimizer_idx=optimizer_idx, batch_idx=batch_idx, device=self.curr_device)
        for name, metric in train_loss.items():
            if 'loss' in name:
                self.log('train_' + name, metric.mean().item(), on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch >= self.dpmm_init_epoch:
            z = torch.cat([out['z'] for out in outputs])
            self.model.fit_dpmm(z)
        if self.model.bnp_model:
            self.log('Number_of_DP_Comps', int(self.model.bnp_model.obsModel.K), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'], optimizer_idx=optimizer_idx, batch_idx=batch_idx, device=self.curr_device)
        for name, metric in val_loss.items():
            if 'loss' in name:
                self.log('val_' + name, metric.mean().item(), on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        try:
            if self.params.get('LR_2', None) is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(), lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass
        try:
            if self.params.get('scheduler_gamma', None) is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)
                try:
                    if self.params.get('scheduler_gamma_2', None) is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1], gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return (optims, scheds)
        except:
            return optims
        return optims

    @data_loader
    def train_dataloader(self):
        full_train_dataset = PTTensorDataset(data_path=self.params['data_path'], train=True)
        self.num_train_imgs = len(full_train_dataset)
        return DataLoader(full_train_dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)

    @data_loader
    def val_dataloader(self):
        full_test_dataset = PTTensorDataset(data_path=self.params['data_path'], train=False)
        self.sample_dataloader = DataLoader(full_test_dataset, batch_size=self.params['batch_size'], shuffle=False, drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader
