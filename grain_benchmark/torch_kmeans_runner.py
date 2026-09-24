"""Run the supplied PyTorchKMeans; labels are opened only after fitting."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import numpy as np
from .data import load_labels
from .evaluation import clustering_metrics


def run(args, x, ids, entry):
    import torch
    import torch.distributed as dist
    from torch_clustering import PyTorchKMeans
    from .cli import provenance, write_json
    if args.n_init < 1 or args.max_iter < 1 or args.threads < 1 or args.tol < 0:
        raise ValueError('Invalid solver settings')
    if any(k < 2 or k > len(x) for k in args.k_values):
        raise ValueError('Require 2 <= K <= number of rows')
    if any(s < 0 or s >= 2**31 for s in args.seeds):
        raise ValueError('Seed outside supported range')
    if len(set(args.k_values)) != len(args.k_values) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Repeated K or seed')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly use --device cpu')
    initialized_here = False
    if args.distributed:
        if device.type == 'cuda':
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
            device = torch.device('cuda', torch.cuda.current_device())
        if not dist.is_initialized():
            dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
            initialized_here = True
    elif int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('torchrun requires --distributed (otherwise outputs would collide)')
    rank = dist.get_rank() if args.distributed else 0
    world = dist.get_world_size() if args.distributed else 1
    X = torch.from_numpy(x).to(device)
    try:
        for k in args.k_values:
            for seed in args.seeds:
                dest = args.out/args.regime/f'k{k}'/f'seed{seed}'
                # All ranks use the same filesystem in the supported single-node setup.
                if dest.exists():
                    raise FileExistsError(dest)
                config = dict(method='kmeans', implementation='provided_torch_clustering.PyTorchKMeans',
                    regime=args.regime, K_requested=k, seed=seed, metric=args.metric,
                    n_init_per_rank=args.n_init, world_size=world, total_restarts=args.n_init*world,
                    restart_seeds=list(range(seed, seed+args.n_init*world)),
                    max_iter=args.max_iter, tol=args.tol, init='k-means++',
                    device=str(device), threads=args.threads,
                    experiment_status='new_run',
                    features_sha256=entry['files']['features']['sha256'])
                if args.distributed:dist.barrier()
                if rank == 0:
                    dest.mkdir(parents=True)
                    write_json(dest/'config_resolved.json',config)
                    prov=provenance();prov['torch']=torch.__version__
                    write_json(dest/'provenance.json',prov)
                model=PyTorchKMeans(metric=args.metric, init='k-means++', random_state=seed,
                    n_clusters=k, n_init=args.n_init, max_iter=args.max_iter, tol=args.tol,
                    distributed=args.distributed, verbose=False)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                labels_tensor=model.fit_predict(X)
                if device.type=='cuda':torch.cuda.synchronize()
                fit_seconds=time.perf_counter()-start
                if rank==0:
                    pred=labels_tensor.cpu().numpy()
                    # No label access above this point, including restart selection.
                    y=load_labels(args.data_root,args.regime)
                    np.savez_compressed(dest/'predictions.npz',row_ids=ids,predictions=pred,
                        regime=np.asarray(args.regime),features_sha256=np.asarray(config['features_sha256']))
                    _,inertia=model.predict(X)
                    result=clustering_metrics(y,pred)
                    result.update(config,fit_seconds=fit_seconds,native_inertia=float(inertia.item()))
                    write_json(dest/'metrics.json',result)
                    print(json.dumps({'path':str(dest),**result}),flush=True)
                if args.distributed:dist.barrier()
    finally:
        if initialized_here:dist.destroy_process_group()
