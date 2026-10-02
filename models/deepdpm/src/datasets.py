"""Grain feature adapter; reference labels are intentionally not read here."""
import torch
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

def transform_embeddings(transform_type, data):
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    if transform_type == 'normalize':
        return torch.tensor(Normalizer().fit_transform(data_np), dtype=torch.float32)
    elif transform_type == 'min_max':
        return torch.tensor(MinMaxScaler().fit_transform(data_np), dtype=torch.float32)
    elif transform_type == 'standard':
        return torch.tensor(StandardScaler().fit_transform(data_np), dtype=torch.float32)
    elif transform_type == 'l2':
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data_np, dtype=torch.float32)
        return torch.nn.functional.normalize(data, p=2, dim=1)
    elif transform_type == 'l2_and_normalize':
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data_np, dtype=torch.float32)
        data = F.normalize(data, p=2, dim=1)
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        data = (data - mean) / (std + 1e-06)
        return data
    elif transform_type == 'None':
        return data
    else:
        raise NotImplementedError(f'Unknown transform_type: {transform_type}')

class GrainLoaders:

    def __init__(self, features, args):
        x = F.normalize(torch.from_numpy(features).float(), p=2, dim=1)
        if args.transform_input_data:
            x = transform_embeddings(args.transform_input_data, x)
        self.features = x
        self.dataset = TensorDataset(x, torch.zeros(len(x), dtype=torch.long))
        self.args = args

    def get_loaders(self):
        return (DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0), DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0))
