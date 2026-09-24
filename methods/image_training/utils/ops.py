"""Adapted from ProPos by Zhizhong Huang; see THIRD_PARTY.md."""
from torch import nn
import torch
from typing import Union
import torch.distributed as dist
import collections.abc as container_abcs
import six
string_classes = six.string_types

def concat_all_gather(tensor):
    dtype = tensor.dtype
    tensor = tensor.float()
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    output = output.to(dtype)
    return output

class dataset_with_indices(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        outs = self.dataset[idx]
        return [outs, idx]

def convert_to_cuda(data):
    """Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and (not isinstance(data, string_classes)):
        return [convert_to_cuda(d) for d in data]
    else:
        return data

def is_root_worker():
    verbose = True
    if dist.is_initialized():
        if dist.get_rank() != 0:
            verbose = False
    return verbose

def convert_to_ddp(modules: Union[list, nn.Module], **kwargs):
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if dist.is_initialized():
        device = torch.cuda.current_device()
        if isinstance(modules, list):
            modules = [torch.nn.parallel.DistributedDataParallel(x, device_ids=[device], output_device=device, **kwargs) for x in modules]
        else:
            modules = torch.nn.parallel.DistributedDataParallel(modules, device_ids=[device], output_device=device, **kwargs)
    else:
        modules = torch.nn.DataParallel(modules)
    return modules
