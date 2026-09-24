"""Repository-level contracts for the two-regime public interface."""
from pathlib import Path
import ast
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from grain_benchmark.config import METHODS, REGIMES, load_config, method_settings
from grain_benchmark.predictions import image_dataset_digest, read_image_predictions

ROOT = Path(__file__).resolve().parents[1]


def test_only_two_yaml_configurations():
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.suffix in {'.yaml','.yml'})
    assert files == ['configs/grain_only.yaml','configs/grain_volcashdb.yaml']


@pytest.mark.parametrize('regime', REGIMES)
def test_every_method_has_settings(regime):
    for method in METHODS:
        found, settings = method_settings(method, ROOT/'configs'/f'{regime}.yaml')
        assert found == regime and settings
    cfg = load_config(ROOT/'configs'/f'{regime}.yaml')
    assert cfg['diva']['epochs']==150
    assert cfg['deepdpm']['ae_epochs']==150 and cfg['deepdpm']['cluster_epochs']==500
    assert cfg['image']['warmup_epochs']==50
    assert cfg['autopropos']['epochs_cluster_analysis']==[200,600,800]


def test_conflicting_regime_rejected():
    with pytest.raises(ValueError):
        method_settings('diva',ROOT/'configs/grain_only.yaml','grain_volcashdb')


def test_no_private_external_metric_implementations():
    forbidden = {'cluster_acc','cluster_accuracy','best_cluster_fit','normalized_mutual_info_score',
                 'adjusted_rand_score','accuracy_score','calculate_acc','calculate_caa'}
    for p in (ROOT/'methods').rglob('*.py'):
        if 'bnpy' in p.parts:
            continue
        names={n.id for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Name)}
        names|={n.name for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert not names.intersection(forbidden),p


def test_image_config_and_scattering_contracts(tmp_path):
    grain=tmp_path/'grain';ash=tmp_path/'ash';grain.mkdir();ash.mkdir()
    code=r'''
import json,inspect,sys,ast
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from main import build_parser,resolve_options
from models.propos.trainer import scattering_is_active,GrainResize
from models.base import TrainTask
from supervisor import ClusterAnalysis
root=Path(sys.argv[1]);grain=Path(sys.argv[2]);ash=Path(sys.argv[3])
for regime in ('grain_only','grain_volcashdb'):
 for method in ('byol','propos','autopropos'):
  args=['--config',str(root/'configs'/f'{regime}.yaml'),'--method',method,
        '--grain-root',str(grain),'--run-name','test','--seed','7']
  if regime=='grain_volcashdb':args+=['--volcashdb-root',str(ash)]
  if method=='propos':args+=['--initial-k','21']
  opt=resolve_options(build_parser().parse_args(args))
  assert opt.regime==regime and opt.method==method and opt.base_seed==7
assert not scattering_is_active(50*100+1,100,50)
assert scattering_is_active(50*100+2,100,50)
assert scattering_is_active(100*100+1,100,50)
x=Image.fromarray(np.random.default_rng(4).integers(0,255,(33,21,3),dtype=np.uint8))
np.testing.assert_array_equal(np.asarray(GrainResize(24)(x)),np.asarray(x.resize((24,24))))
for node in ast.walk(ast.parse((root/'methods/image_training/models/base.py').read_text())):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ClusterAnalysis':
  inspect.signature(ClusterAnalysis).bind(**{kw.arg:None for kw in node.keywords})
# Unknown label-guided export options must not be silently accepted.
import yaml
cfg=yaml.safe_load((root/'configs/grain_only.yaml').read_text());cfg['image']['save_best_embeddings']=True
bad=grain/'bad.yaml';bad.write_text(yaml.safe_dump(cfg))
a=build_parser().parse_args(['--config',str(bad),'--method','byol','--grain-root',str(grain),'--run-name','bad','--seed','0'])
try:resolve_options(a)
except ValueError:pass
else:raise AssertionError('Removed model-selection option was accepted')
# Dataset membership: VolcAshDB augments SSL only, never the Grain partition.
for parent,count in [(grain,2),(ash,3)]:
 folder=parent/'class0';folder.mkdir()
 for i in range(count):Image.new('RGB',(12,12)).save(folder/f'{i}.png')
roots={'labeled':str(grain),'unlabeled':[str(grain),str(ash)]}
target,_=TrainTask.create_dataset(roots,'custom_big',True)
ssl,_=TrainTask.create_dataset(roots,'custom_big',False,unlabeled=True)
assert len(target)==2 and len(ssl)==5
print('IMAGE CONTRACTS PASS')
'''
    proc=subprocess.run([sys.executable,'-c',code,str(ROOT),str(grain),str(ash)],
        cwd=ROOT/'methods/image_training',env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=35)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr


def _image_bundle(path,seed=0):
    ids=np.arange(4,dtype=np.int64);y=np.array([0,0,1,1]);sample_ids=np.array(['a/1','a/2','b/1','b/2'])
    digest=image_dataset_digest(sample_ids,y)
    np.savez_compressed(path,row_ids=ids,labels=y,predictions=y+seed,sample_ids=sample_ids,
                        regime=np.asarray('grain_only'),dataset_sha256=np.asarray(digest))
    return digest


def test_image_evaluation_without_rgb_or_frozen_embeddings(tmp_path):
    path=tmp_path/'predictions.npz';_image_bundle(path)
    y,ids,pred,_=read_image_predictions(path,'grain_only')
    np.testing.assert_array_equal(y,pred)
    proc=subprocess.run([sys.executable,'-m','grain_benchmark','evaluate','--input-kind','images',
        '--regime','grain_only','--data-root',str(tmp_path/'absent-data'),'--predictions',str(path),
        '--out',str(tmp_path/'metrics.json')],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    assert json.loads((tmp_path/'metrics.json').read_text())['ACC']==1


def test_image_regime_and_integrity_checks(tmp_path):
    path=tmp_path/'predictions.npz';_image_bundle(path)
    with pytest.raises(ValueError):read_image_predictions(path,'grain_volcashdb')
    with np.load(path) as b:values={name:b[name] for name in b.files}
    values['labels']=np.array([1,1,0,0]);np.savez_compressed(path,**values)
    with pytest.raises(ValueError):read_image_predictions(path,'grain_only')


def test_image_structured_mean_without_rgb(tmp_path):
    paths=[]
    for seed in [0,1]:
        path=tmp_path/f'p{seed}.npz';_image_bundle(path,seed);paths.append(path)
    proc=subprocess.run([sys.executable,'-m','grain_benchmark','structured-mean','--input-kind','images',
        '--regime','grain_only','--predictions',*[str(p) for p in paths],'--expected-runs','2',
        '--out',str(tmp_path/'matrix')],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    with np.load(tmp_path/'matrix/structured_confusion.npz') as b:
        assert b['mean'].shape==(4,2)
        np.testing.assert_allclose(b['mean'].sum(axis=0),100.)


@pytest.mark.parametrize('entry',['methods/diva/train.py','methods/deepdpm/train.py','methods/ddpm/train.py','methods/image_training/main.py'])
def test_entrypoint_help(entry):
    p=subprocess.run([sys.executable,entry,'--help'],cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr
    assert '--config' in p.stdout and '--seed' in p.stdout

@pytest.mark.parametrize('different_config',[False,True])
def test_image_summary_keeps_training_settings_separate(tmp_path,different_config):
    from grain_benchmark.evaluation import clustering_metrics
    paths=[]
    for seed in [0,1]:
        r=clustering_metrics(np.array([0,0,1,1]),np.array([0,0,1,1]))
        r.update(method='autopropos',regime='grain_only',seed=seed,dataset_sha256='dataset1',
                 input_kind='images',config={'epochs':1000,'num_cluster':201 if different_config and seed else 200})
        path=tmp_path/f'm{seed}.json';path.write_text(json.dumps(r));paths.append(path)
    proc=subprocess.run([sys.executable,'-m','grain_benchmark.summarize','--metrics',
        *[str(p) for p in paths],'--expected-runs','2','--std-ddof','1','--out',str(tmp_path/'summary.json')],
        cwd=ROOT,capture_output=True,text=True,timeout=15)
    if different_config:
        assert proc.returncode!=0
        assert 'Run count mismatch' in proc.stderr
    else:
        assert proc.returncode==0,proc.stderr
        assert json.loads((tmp_path/'summary.json').read_text())['groups'][0]['seeds']==[0,1]
