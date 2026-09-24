"""Train the Grain DDPM flow and mixture; evaluate the resulting partition."""
from pathlib import Path
import argparse
import sys
import time
import json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default=str(ROOT / 'data'))
    p.add_argument('--config', type=Path)
    p.add_argument('--regime', choices=('grain_only', 'grain_volcashdb'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--epochs', type=int)
    args = p.parse_args()
    from grain_benchmark.config import method_settings
    args.regime, settings = method_settings('ddpm', args.config, args.regime)
    import torch
    import numpy as np
    from torch.utils.data import Dataset
    from engine import init_dir_params, init_flow_model, dirichlet_clustering, train_flow
    from utils import InfiniteDataLoader
    from grain_benchmark.data import load_features
    from grain_benchmark.results import save_feature_result
    from grain_benchmark.cli import write_json
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    x, ids, entry = load_features(args.data_root, args.regime)
    args.epochs = args.epochs if args.epochs is not None else settings['epochs']
    if args.epochs < 1 or args.seed < 0:
        raise ValueError('Invalid epoch or seed')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.dim = x.shape[1]
    args.n_sample_load = len(x)
    args.lr = settings['learning_rate']
    args.nice_nlayers = settings['nice_layers']
    args.nice_units = settings['nice_units']
    args.logalpha = settings['logalpha']
    args.kappa0 = settings['kappa0']
    args.a0 = settings['a0']
    args.b0 = settings['b0']
    args.dmm_rebuild_freq = settings['rebuild_frequency']
    config = dict(settings, epochs=args.epochs, seed=args.seed, dim=args.dim, N=len(x), input_preprocessing='native_global_scalar_standardization_torch_std', status='new_run')
    write_json(args.out / 'config_resolved.json', config)
    features = torch.from_numpy(x).float()
    scalar_std = torch.std(features)
    if not torch.isfinite(scalar_std) or scalar_std <= 0:
        raise ValueError('Cannot apply native standardization to constant/non-finite input')
    features = (features - torch.mean(features)) / scalar_std

    class Features(Dataset):

        def __len__(self):
            return len(x)

        def __getitem__(self, i):
            z = features[i]
            return (z, z, 0, i)
    ds = Features()
    inf = iter(InfiniteDataLoader(ds, batch_size=settings['batch_size'], shuffle=True, pin_memory=args.device == 'cuda'))
    model, opt = init_flow_model(args)
    state = init_dir_params(args)

    def transform():
        model.to(args.device).eval()
        with torch.no_grad():
            return np.concatenate([model.f(row.view(-1).to(args.device))[0].cpu().numpy().reshape(1, -1) for row in features])
    step_dmm = step_flow = 0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        z = transform()
        state = dirichlet_clustering(epoch, state, z, None, settings['assignments_per_epoch'], args, log_step=step_dmm)
        step_dmm += settings['assignments_per_epoch'] + 1
        train_flow(epoch, model, opt, inf, settings['flow_steps_per_epoch'], state, args, log_step=step_flow, prev_z_repr=z)
        step_flow += settings['flow_steps_per_epoch'] + 1
        print(f'epoch={epoch + 1}/{args.epochs} K={state.K}', flush=True)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    result = save_feature_result(args.out, state.samples_k, ids, entry, data_root=args.data_root, regime=args.regime, method='ddpm', config=config, fit_seconds=duration)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
