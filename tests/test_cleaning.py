"""Focused regression tests for the cleanup, not GPU training validation."""
from pathlib import Path
import hashlib
import json
import importlib.util
import subprocess
import sys
from argparse import Namespace
import numpy as np
import pytest
import torch
from grain_benchmark.final_export import write_final_artifacts
from grain_benchmark.evaluation import clustering_metrics
from torch_clustering import PyTorchKMeans

ROOT = Path(__file__).resolve().parents[1]


def test_export_is_independent_of_reference_labels(tmp_path):
    x = np.arange(48, dtype=np.float32).reshape(6,8)
    y = np.array([0,0,1,1,2,2])
    a = write_final_artifacts(tmp_path/'a',x,y,epoch=1000)
    b = write_final_artifacts(tmp_path/'b',x,y[::-1],epoch=1000)
    assert a['files']['features']['sha256'] == b['files']['features']['sha256']
    assert a['selection'] == 'final_epoch_after_last_optimizer_step'
    np.testing.assert_array_equal(np.load(tmp_path/'a'/'features.npy'),x)


def test_final_export_never_overwrites(tmp_path):
    x=np.ones((3,4),dtype=np.float32)
    write_final_artifacts(tmp_path/'final',x,epoch=1)
    with pytest.raises(FileExistsError):
        write_final_artifacts(tmp_path/'final',2*x,epoch=2)
    np.testing.assert_array_equal(np.load(tmp_path/'final'/'features.npy'),x)


@pytest.mark.parametrize('kwargs',[
    {'epoch':0}, {'epoch':True},
    {'epoch':1,'metadata':{'selection':'best'}},
    {'epoch':1,'row_ids':np.array([1,1,2])},
])
def test_export_rejects_ambiguous_metadata(tmp_path,kwargs):
    with pytest.raises(ValueError):
        write_final_artifacts(tmp_path/'bad',np.ones((3,4),dtype=np.float32),**kwargs)
    assert not (tmp_path/'bad').exists()


def test_best_acc_options_absent_from_active_image_code():
    for d in (ROOT/'methods/image_training',ROOT/'configs'):
        for p in d.rglob('*'):
            if p.suffix in ('.py','.yaml'):
                text=p.read_text()
                assert 'save_best_embeddings' not in text, p
                assert 'save_best_model' not in text, p
                assert 'best_acc' not in text, p


def test_kmeans_core_hashes_match_supplied_archive():
    manifest=json.loads((ROOT/'tests/kmeans_core_sha256.json').read_text())
    for rel,digest in manifest.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('metric',['cosine','euclidean'])
def test_native_kmeans_toy_partition(metric):
    torch.set_num_threads(1)
    x=torch.tensor([[1.,.01],[1.,-.01],[.01,1.],[-.01,1.]])
    model=PyTorchKMeans(n_clusters=2,metric=metric,init='k-means++',n_init=2,
                        random_state=7,max_iter=100,tol=1e-4,distributed=False,verbose=False)
    pred=model.fit_predict(x).cpu().numpy()
    assert clustering_metrics(np.array([0,0,1,1]),pred)['ACC']==1.


def test_native_deepdpm_autoencoder_forward():
    p=ROOT/'methods/deepdpm/src/feature_extractors/autoencoder.py'
    spec=importlib.util.spec_from_file_location('test_native_ae',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=Namespace(hidden_dims=[500,500,2000],latent_dim=20,n_clusters=3)
    ae=mod.AutoEncoder(args,256)
    x=torch.randn(4,256)
    assert ae(x).shape==(4,256)
    assert ae(x,latent=True).shape==(4,20)
    loss=ae(x).square().sum();loss.backward()
    assert all(p.grad is not None for p in ae.parameters())


def test_ddpm_core_cpu_smoke():
    code="""
from argparse import Namespace
import numpy as np,torch
from engine import init_flow_model,init_dir_params,dirichlet_clustering,train_flow
from torch.utils.data import TensorDataset
from utils import InfiniteDataLoader
np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
a=Namespace(dim=4,n_sample_load=32,device='cpu',nice_nlayers=2,nice_units=8,
 lr=1e-4,a0=.05,b0=.1,kappa0=5.,logalpha=1.,dmm_rebuild_freq=4)
x=torch.randn(32,4);x=(x-x.mean())/x.std()
m,opt=init_flow_model(a);state=init_dir_params(a)
z=torch.stack([m.f(row)[0] for row in x]).detach().numpy()
assert z.shape==(32,4)
state=dirichlet_clustering(0,state,z,None,32,a)
assert (state.samples_k>=0).all()
ds=TensorDataset(x,x,torch.zeros(32),torch.arange(32))
train_flow(0,m,opt,iter(InfiniteDataLoader(ds,batch_size=16,shuffle=True)),2,state,a)
assert all(torch.isfinite(p).all() for p in m.parameters())
"""
    proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT/'methods/ddpm',
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
