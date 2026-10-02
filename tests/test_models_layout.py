"""Regression checks for the explicit per-model entry points."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from models.common.image_training.main import build_parser

ROOT = Path(__file__).resolve().parents[1]
IMAGE_METHODS = ('byol', 'propos', 'autopropos')


def test_all_model_implementations_are_under_models():
    for name in ('diva', 'deepdpm', 'ddpm', 'kmeans', *IMAGE_METHODS):
        assert (ROOT/'models'/name/'train.py').is_file()
    assert (ROOT/'models/autopropos/supervisor.py').is_file()
    assert (ROOT/'models/kmeans/torch_clustering/kmeans/kmeans.py').is_file()
    assert (ROOT/'models/common/image_training/base.py').is_file()
    assert not (ROOT/'methods').exists()
    assert not (ROOT/'torch_clustering').exists()
    assert not (ROOT/'grain_benchmark/torch_kmeans_runner.py').exists()


@pytest.mark.parametrize('method', IMAGE_METHODS)
def test_explicit_image_entrypoint_cannot_select_another_method(method):
    parser = build_parser(method)
    argv = ['--config','configs/grain_only.yaml','--grain-root','/tmp/grain',
            '--run-name','test','--seed','0']
    assert parser.parse_args(argv).method == method
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, '--method', 'byol'])


@pytest.mark.parametrize('method', IMAGE_METHODS)
@pytest.mark.parametrize('regime', ('grain_only','grain_volcashdb'))
def test_explicit_image_entrypoint_resolves_configuration(method, regime, tmp_path):
    grain = tmp_path/'grain'
    ash = tmp_path/'ash'
    grain.mkdir()
    ash.mkdir()
    command = [sys.executable, str(ROOT/'models'/method/'train.py'),
               '--config', str(ROOT/'configs'/f'{regime}.yaml'),
               '--grain-root',str(grain), '--run-name','test', '--seed','7',
               '--check-config']
    if regime == 'grain_volcashdb':
        command.extend(['--volcashdb-root',str(ash)])
    if method == 'propos':
        command.extend(['--initial-k','21'])
    # Launch from another directory: the wrappers must resolve the repository.
    proc = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                          timeout=35, env={**os.environ,'OMP_NUM_THREADS':'1'})
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    opts = json.loads(proc.stdout)
    assert opts['method'] == method and opts['regime'] == regime
    assert opts['seed'] == 7 and opts['epochs'] == 1000
    assert bool(opts['clusternet']) == (method == 'autopropos')
    assert len(opts['data_folder']['unlabeled']) == (2 if regime == 'grain_volcashdb' else 1)
    if method == 'byol':
        assert opts['cluster_loss_weight'] == 0
    if method == 'autopropos':
        assert opts['epochs_cluster_analysis'] == [200,600,800]
