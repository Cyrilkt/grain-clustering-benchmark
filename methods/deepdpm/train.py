"""Train a Grain autoencoder, then one DeepDPM split/merge clustering stage."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--regime', choices=('grain_only', 'grain_volcashdb'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--ae-epochs', type=int, default=None)
    parser.add_argument('--cluster-epochs', type=int, default=None)
    cli = parser.parse_args()
    from grain_benchmark.config import method_settings
    cli.regime, settings = method_settings('deepdpm', cli.config, cli.regime)
    cli.ae_epochs = cli.ae_epochs if cli.ae_epochs is not None else settings['ae_epochs']
    cli.cluster_epochs = cli.cluster_epochs if cli.cluster_epochs is not None else settings['cluster_epochs']
    if min(cli.ae_epochs, cli.cluster_epochs) < 1 or cli.seed < 0:
        parser.error('Epoch counts must be positive and seed nonnegative')
    import torch
    import numpy as np
    import pytorch_lightning as pl
    if int(pl.__version__.split('.')[0]) != 1:
        raise RuntimeError('The supplied DeepDPM core requires legacy Lightning 1.x; do not silently use the current Lightning 2.x environment. See environments/README.md.')
    if cli.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    from defaults import default_options
    from src.AE_ClusterPipeline import AE_ClusterPipeline
    from src.datasets import GrainLoaders
    from src.utils import check_args
    from pytorch_lightning.loggers.base import DummyLogger
    from grain_benchmark.data import load_features
    from grain_benchmark.results import save_feature_result
    from grain_benchmark.cli import write_json
    x, ids, entry = load_features(cli.data_root, cli.regime)
    args = default_options()
    args.dataset = 'custom'
    args.dir = str(Path(cli.data_root) / 'embeddings' / cli.regime)
    args.seed = cli.seed
    args.gpus = '0' if cli.device == 'cuda' else None
    args.device = cli.device
    args.data_dim = x.shape[1]
    args.features_dim = x.shape[1]
    args.latent_dim = settings['latent_dim']
    args.init_k = args.n_clusters = settings['initial_k']
    args.batch_size = settings['batch_size']
    args.lr = settings['learning_rate']
    args.pretrain = True
    args.pretrain_epochs = cli.ae_epochs
    args.train_cluster_net = cli.cluster_epochs
    args.pretrain_path = None
    args.alternate = False
    args.number_of_ae_alternations = 1
    args.save_checkpoints = False
    args.offline = True
    check_args(args, args.latent_dim)
    pl.seed_everything(cli.seed)
    loaders = GrainLoaders(x, args)
    train_loader, val_loader = loaders.get_loaders()
    cli.out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), stage_sequence=['ae_pretrain', 'first_clustering'], reference_labels='final_evaluation_only', input_preprocessing='native_row_L2_then_native_transform_input_data', status='new_run')
    write_json(cli.out / 'config_resolved.json', config)

    class GrainAE(AE_ClusterPipeline):

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader
    logger = DummyLogger()
    model = GrainAE(logger, args, x.shape[1], loaders.features.numpy(), np.zeros(len(x), dtype=np.int64))
    trainer = pl.Trainer(logger=logger, max_epochs=cli.ae_epochs, gpus=args.gpus, num_sanity_val_steps=0, checkpoint_callback=False)
    start = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    model.to(cli.device)
    model.pretrain = False
    model._init_clusters()
    head = model.clustering.model.cluster_model.to(cli.device)
    head.eval()
    with torch.no_grad():
        pred = np.concatenate([head(batch.to(cli.device)).argmax(-1).cpu().numpy() for batch in loaders.features.split(args.batch_size)])
    if cli.device == 'cuda':
        torch.cuda.synchronize()
    result = save_feature_result(cli.out, pred, ids, entry, data_root=cli.data_root, regime=cli.regime, method='deepdpm', config=config, fit_seconds=time.perf_counter() - start)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
