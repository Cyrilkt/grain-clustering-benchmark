"""Adapted from ProPos by Zhizhong Huang; see THIRD_PARTY.md."""
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import tqdm
import torch.distributed as dist
from utils.ops import convert_to_cuda, is_root_worker

@torch.no_grad()
def extract_features(extractor, loader):
    extractor.eval()
    local_features = []
    local_labels = []
    for inputs in tqdm.tqdm(loader, disable=not is_root_worker()):
        images, labels = convert_to_cuda(inputs)
        local_labels.append(labels)
        local_features.append(extractor(images))
    local_features = torch.cat(local_features, dim=0)
    local_labels = torch.cat(local_labels, dim=0)
    indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()
    features = torch.zeros(len(loader.dataset), local_features.size(1)).cuda()
    all_labels = torch.zeros(len(loader.dataset)).cuda()
    counts = torch.zeros(len(loader.dataset)).cuda()
    features.index_add_(0, indices, local_features)
    all_labels.index_add_(0, indices, local_labels.float())
    counts[indices] = 1.0
    if dist.is_initialized():
        dist.all_reduce(features, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    labels = (all_labels / counts).long()
    features /= counts[:, None]
    return (features, labels)

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

    def __str__(self):
        return f'transform1 {str(self.transform1)} transform2 {str(self.transform2)}'
