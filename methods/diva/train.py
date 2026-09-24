"""Run the supplied DIVA model on one complete frozen Grain collection.
Requires the legacy Lightning hooks used by the supplied model (Lightning 1.x).
"""
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
    args.regime, settings = method_settings('diva', args.config, args.regime)
    args.epochs = args.epochs if args.epochs is not None else settings['epochs']
    import torch
    import numpy as np
    import pytorch_lightning as pl
    if int(pl.__version__.split('.')[0]) >= 2:
        raise RuntimeError('DIVA retains the original Lightning 1.x epoch hooks; use the source environment, not Lightning 2.x')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    if args.epochs < 1 or args.seed < 0:
        raise ValueError('Invalid epoch or seed')
    from grain_benchmark.data import load_features
    from grain_benchmark.results import save_feature_result
    from grain_benchmark.cli import write_json
    from model import DIVA_MLP, VAEXperiment
    from bnpy.data.XData import XData
    x, ids, entry = load_features(args.data_root, args.regime)
    args.out.mkdir(parents=True, exist_ok=False)
    data_dir = Path(args.data_root).resolve() / 'embeddings' / args.regime
    config = dict(method='diva', epochs=args.epochs, seed=args.seed, input_dim=x.shape[1], latent_dim=settings['latent_dim'], sF=settings['sF'], initial_sF=settings['initial_sF'], batch_size=settings['batch_size'], LR=settings['learning_rate'], weight_decay=settings['weight_decay'], kld_weight=settings['kld_weight'], min_component_samples=settings['min_component_samples'], final_inference_drop_last=False, status='new_run')
    write_json(args.out / 'config_resolved.json', config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = DIVA_MLP(input_dim=x.shape[1], latent_dim=config['latent_dim'], dpmm_param=dict(sF=config['sF'], initial_sF=config['initial_sF'], b_minNumAtomsForNewComp=config['min_component_samples'], b_minNumAtomsForTargetComp=config['min_component_samples'], b_minNumAtomsForRetainComp=config['min_component_samples']))
    model.bnp_root = str(args.out.resolve() / 'bn_model') + '/'
    experiment = VAEXperiment(model, dict(batch_size=config['batch_size'], LR=config['LR'], weight_decay=config['weight_decay'], kld_weight=config['kld_weight'], data_path=str(data_dir)))
    trainer = pl.Trainer(default_root_dir=str(args.out), logger=False, max_epochs=args.epochs, accelerator='gpu' if args.device == 'cuda' else 'cpu', devices=1, enable_checkpointing=False, num_sanity_val_steps=5, log_every_n_steps=50)
    start = time.perf_counter()
    trainer.fit(experiment)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    model = model.to(args.device).eval()
    pred = []
    with torch.no_grad():
        for chunk in torch.from_numpy(x).split(config['batch_size']):
            z = model(chunk.to(args.device))[4].cpu().numpy()
            pred.extend(model.bnp_model.calc_local_params(XData(z))['resp'].argmax(axis=1).tolist())
    result = save_feature_result(args.out, pred, ids, entry, data_root=args.data_root, regime=args.regime, method='diva', config=config, fit_seconds=duration)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
