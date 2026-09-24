# Copyright (c) 2022 Meitar Ronen
import pytorch_lightning as pl
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

class ClusterNetTrainer:

    def __init__(self, args, init_k, latent_dim, feature_extractor, centers=None, init_num=0):
        self.args = args
        self.cluster_model = ClusterNetModel(hparams=args, input_dim=latent_dim, init_k=init_k, feature_extractor=feature_extractor, centers=centers, init_num=init_num)
        if self.args.seed is not None:
            pl.utilities.seed.seed_everything(self.args.seed)

    def fit(self, train_loader, val_loader, logger, n_epochs, centers=None):
        cluster_trainer = pl.Trainer(logger=logger, max_epochs=n_epochs, gpus=self.args.gpus, num_sanity_val_steps=0, checkpoint_callback=False)
        if self.args.seed is not None:
            pl.utilities.seed.seed_everything(self.args.seed)
        self.cluster_model.centers = centers
        cluster_trainer.fit(self.cluster_model, train_loader, val_loader)

    def get_current_K(self):
        return self.cluster_model.K

    def get_clusters_centers(self):
        return self.cluster_model.mus.cpu().numpy()

    def get_clusters_covs(self):
        return self.cluster_model.covs.cpu().numpy()

    def get_clusters_pis(self):
        return self.cluster_model.pi.cpu().numpy()
