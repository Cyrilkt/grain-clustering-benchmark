"""Train BYOL, fixed-K ProPos or AutoProPos on Grain images."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method', choices=('byol', 'propos', 'autopropos'), required=True)
    parser.add_argument('--grain-root', type=Path, required=True)
    parser.add_argument('--volcashdb-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--num-devices', type=int)
    parser.add_argument('--initial-k', type=int, help='Required fixed K for ProPos; optional initial K for AutoProPos')
    parser.add_argument('--epochs', type=int, help='Override the fixed training horizon')
    parser.add_argument('--check-config', action='store_true', help='Validate paths and settings, print the resolved configuration, then exit')
    return parser

def resolve_options(args):
    from grain_benchmark.config import method_settings
    from models.base import TrainTask
    regime, settings = method_settings(args.method, args.config)
    defaults = TrainTask.default_options()
    allowed = set(defaults) | {'batch_size', 'epochs', 'feat_dim', 'hidden_size', 'img_size', 'learning_rate', 'learning_eta_min', 'lambda_predictor_lr', 'warmup_epochs', 'weight_decay', 'momentum', 'momentum_base', 'momentum_max', 'momentum_increase', 'shuffling_bn', 'symmetric', 'temperature', 'reassign', 'use_gaussian_blur', 'autoencoder_latent_dim', 'cluster_range', 'n_candidate_autoencoder_clustering', 'autoencoder_training_epoch', 'n_candidate_k_mean', 'n_candidate_k_max', 'number_autoencoder', 'clusternet_training_data', 'encoder_name', 'kmeans_n_init', 'supervisor_kmeans_n_init', 'cluster_loss_weight', 'latent_std', 'clusternet', 'epochs_cluster_analysis', 'num_cluster'}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f'Unsupported image configuration keys: {sorted(unknown)}')
    if regime == 'grain_volcashdb' and args.volcashdb_root is None:
        raise ValueError('grain_volcashdb requires --volcashdb-root')
    if regime == 'grain_only' and args.volcashdb_root is not None:
        raise ValueError('grain_only must not include --volcashdb-root')
    for path in [args.grain_root] + ([args.volcashdb_root] if args.volcashdb_root else []):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.seed < 0:
        raise ValueError('The run seed must be nonnegative')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('run-name must be a single directory name')
    if args.method == 'propos' and args.initial_k is None:
        raise ValueError('Fixed-K ProPos requires an explicit --initial-k')
    opt = argparse.Namespace(**{**defaults, **settings})
    for key, value in (('num_cluster', args.initial_k), ('num_workers', args.num_workers), ('num_devices', args.num_devices), ('epochs', args.epochs)):
        if value is not None:
            setattr(opt, key, value)
    if opt.resume_epoch != 0:
        raise ValueError('This launcher starts a new run; resume is not supported')
    if opt.num_workers < 0 or opt.num_devices < 1 or opt.epochs < 1:
        raise ValueError('Invalid worker count, device count or training horizon')
    if opt.num_cluster < 2 or opt.batch_size < 2:
        raise ValueError('Initial K and per-device batch size must be at least 2')
    if opt.encoder_name != 'resnet50':
        raise ValueError('This benchmark uses the ResNet-50 image backbone')
    if args.method == 'byol' and (opt.clusternet or opt.cluster_loss_weight != 0):
        raise ValueError('BYOL requires no supervisor and zero prototype-scattering weight')
    if args.method == 'propos' and opt.clusternet:
        raise ValueError('Fixed-K ProPos requires the supervisor to be disabled')
    if args.method == 'autopropos' and (not opt.clusternet):
        raise ValueError('AutoProPos requires the supervisor')
    opt.method, opt.regime = (args.method, regime)
    opt.base_seed, opt.seed = (args.seed, args.seed)
    opt.output_root, opt.run_name = (str(args.output_root.resolve()), args.run_name)
    opt.debug_dir = str(Path(opt.output_root) / opt.run_name / 'supervisor')
    opt.data_folder = {'labeled': str(args.grain_root.resolve()), 'unlabeled': [str(args.grain_root.resolve())] + ([str(args.volcashdb_root.resolve())] if args.volcashdb_root else [])}
    return opt

def main():
    args = build_parser().parse_args()
    opt = resolve_options(args)
    if args.check_config:
        print(json.dumps(vars(opt), indent=2))
        return
    import torch
    import torch.distributed as dist
    from grain_benchmark.cli import write_json, provenance
    from models.propos.trainer import BYOL
    if not torch.cuda.is_available():
        raise RuntimeError('Image training requires CUDA and torchrun')
    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Launch image training with torchrun --standalone --nproc_per_node=...')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    try:
        if opt.num_devices != dist.get_world_size():
            raise ValueError('num_devices must match the torchrun process count')
        destination = Path(opt.output_root) / opt.run_name
        error = [None]
        if dist.get_rank() == 0:
            try:
                destination.mkdir(parents=True, exist_ok=False)
                (destination / 'supervisor').mkdir()
            except OSError as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f'Cannot create run directory: {error[0]}')
        opt.seed = opt.base_seed + dist.get_rank()
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        if dist.get_rank() == 0:
            write_json(destination / 'config_resolved.json', vars(opt))
            write_json(destination / 'provenance.json', provenance())
        dist.barrier()
        task = BYOL(opt)
        start = time.perf_counter()
        task.fit()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if dist.get_rank() == 0:
            write_json(destination / 'timing.json', {'training_and_final_export_seconds': elapsed, 'world_size': dist.get_world_size(), 'global_batch_size': opt.batch_size * dist.get_world_size()})
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
