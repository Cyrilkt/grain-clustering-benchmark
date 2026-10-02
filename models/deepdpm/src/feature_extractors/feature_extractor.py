# Copyright (c) 2022 Meitar Ronen
from src.feature_extractors.autoencoder import AutoEncoder, ConvAutoEncoder
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self, args, input_dim):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.feature_extractor = None
        self.autoencoder = None
        self.latent_dim = None
        self.input_dim = input_dim
        if self.args.dataset == 'usps':
            self.autoencoder = ConvAutoEncoder(args, input_dim=self.input_dim)
        else:
            self.autoencoder = AutoEncoder(args, input_dim=self.input_dim)
        self.latent_dim = self.autoencoder.latent_dim

    def forward(self, X, latent=False):
        if self.feature_extractor:
            X = self.feature_extractor(X)
        if self.autoencoder:
            output = self.autoencoder.encoder(X)
            if latent:
                return output
            return self.autoencoder.decoder(output)
        return X

    def decode(self, latent_X):
        return self.autoencoder.decoder(latent_X)
