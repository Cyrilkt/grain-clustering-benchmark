import numpy as np
from numpy import log, pi
from torch.utils.data import DataLoader

class InfiniteDataLoader(DataLoader):
    """
    Reload the dataset from start when meets an end.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def mvnrvs(mu, sig2):
    """multi-variant normal random variables"""
    sig = np.sqrt(sig2)
    d = len(mu)
    return np.random.randn(d) * sig + mu

def mvnlogpdf(x, mu, sig2):
    """multi-variant normal log probability density function"""
    d = len(mu)
    return -0.5 * d * log(2 * pi * sig2) - 0.5 * np.sum((x - mu) ** 2) / sig2
