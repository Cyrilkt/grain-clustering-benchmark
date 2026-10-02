"""Aggregate actual run outputs; never infer historical seeds or silently mix protocols."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .cli import write_json

GROUP_FIELDS = ('method', 'regime', 'K_requested', 'n_init', 'max_iter', 'tol',
                'algorithm', 'implementation', 'metric', 'n_init_per_rank', 'world_size',
                'total_restarts', 'input_kind', 'dataset_sha256', 'selection', 'device', 'init', 'threads', 'features_sha256',
                'acc_policy', 'nmi_average_method', 'experiment_status')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', nargs='+', type=Path, required=True)
    p.add_argument('--expected-runs', type=int, required=True)
    p.add_argument('--std-ddof', type=int, choices=[0,1], required=True,
                   help='Standard-deviation convention: 0 = population, 1 = sample')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.expected_runs <= a.std_ddof:
        p.error('Need more runs than std-ddof')
    if len(set(x.resolve() for x in a.metrics)) != len(a.metrics):
        p.error('Duplicate input file')
    groups = {}
    for path in a.metrics:
        r = json.loads(path.read_text())
        if not {'method','regime','seed','ACC','NMI','ARI','K_occupied','N'}.issubset(r):
            raise ValueError(f'{path}: missing required run metadata')
        if not ({'features_sha256', 'dataset_sha256'} & set(r)):
            raise ValueError(f'{path}: no input identity metadata')
        meta = {k:r.get(k) for k in GROUP_FIELDS}
        if 'config' in r:
            meta['native_config']={k:v for k,v in r['config'].items() if k != 'seed'}
        key = json.dumps(meta,sort_keys=True)
        groups.setdefault(key,[]).append((path,r))
    output = {'std_ddof':a.std_ddof,'expected_runs':a.expected_runs,'groups':[]}
    for key, runs in groups.items():
        if len(runs) != a.expected_runs:
            raise ValueError(f'Run count mismatch for {key}: {len(runs)}')
        if len(set(r['seed'] for _,r in runs)) != len(runs):
            raise ValueError('Duplicate seed within an experiment group')
        if len(set(r['N'] for _,r in runs)) != 1:
            raise ValueError('Evaluated sample counts differ')
        item = {'configuration':json.loads(key),'seeds':[r['seed'] for _,r in runs],
                'inputs':[str(path) for path,_ in runs],'N':runs[0][1]['N'],'summary':{}}
        for metric in ('ACC','NMI','ARI','K_occupied','fit_seconds'):
            if not all(metric in r for _,r in runs):
                continue
            values=np.array([r[metric] for _,r in runs],dtype=float)
            if not np.isfinite(values).all():raise ValueError(f'Non-finite {metric}')
            item['summary'][metric]={'values':values.tolist(),'mean':float(values.mean()),
                                     'std':float(values.std(ddof=a.std_ddof))}
        output['groups'].append(item)
    write_json(a.out,output)
    print(a.out)

if __name__=='__main__':main()
