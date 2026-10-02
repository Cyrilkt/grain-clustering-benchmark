"""Common Grain engine from demo_synthetic_gaussian_grain_umap.py.
The three *_umap* variants share these training functions. Input/output and plots
are separated; the distinct demo_ddpm_grain.py pretraining branch is not selected.
"""
from argparse import Namespace
import numpy as np
from numpy import log, exp, pi
import torch
import torch.optim as optim
from scipy.special import loggamma
from nice import NICE
from dircluster import sample_mu_lam
from utils import mvnlogpdf

def dirichlet_clustering(global_epoch, dir_params, samples, ys, n_iter_samples, args, log_step=0):
    hp = dir_params.hyper
    _mu0, _ka0, logalpha, _a0, _b0 = (hp.mu0, hp.ka0, hp.logalpha, hp.a0, hp.b0)
    K, lam_K, mu_K, nK, ks = (dir_params.K, dir_params.lam_K, dir_params.mu_K, dir_params.n_K, dir_params.samples_k)
    for iter_idx in range(n_iter_samples):
        global_log_step = iter_idx + log_step
        sample_idx = global_log_step % len(ks)
        xi = samples[sample_idx]
        old_k = ks[sample_idx]
        if old_k != -1:
            nK[old_k] -= 1
            if nK[old_k] == 0:
                idx = ks == K - 1
                ks[idx] = old_k
                nK[old_k], lam_K[old_k], mu_K[old_k] = (nK[K - 1], lam_K[K - 1], mu_K[K - 1])
                nK, lam_K, mu_K = (nK[:-1], lam_K[:-1], mu_K[:-1])
                K -= 1
        p_lst = []
        Kn = K
        if K == 0:
            chosen_k = 0
        else:
            Klst = np.random.choice(K, size=Kn, replace=False, p=np.asarray(nK) / np.sum(np.asarray(nK)))
            for k in Klst:
                pk = log(nK[k]) + mvnlogpdf(xi, mu_K[k], 1 / lam_K[k])
                p_lst.append(pk)
            _kan = _ka0 + 1
            _an = _a0 + 0.5 * args.dim
            _bn = _b0 + _ka0 * np.linalg.norm(xi - _mu0) ** 2 / (2 * (_ka0 + 1))
            logpk = log(logalpha) + loggamma(_an) - loggamma(_a0) + _a0 * log(_b0) - _an * log(_bn) + 0.5 * (log(_ka0) - log(_kan)) - args.dim / 2 * log(2 * pi)
            p_lst.append(logpk)
            maxpk = max(p_lst)
            p_lst = [exp(v - maxpk) for v in p_lst]
            chosen_k = np.random.choice(list(range(Kn + 1)), p=p_lst / sum(p_lst))
        if chosen_k == Kn:
            nK.append(1)
            ks[sample_idx] = K
            lam_k, mu_k = sample_mu_lam(samples, nK, ks, chosen_k, _mu0, _ka0, _a0, _b0)
            mu_K.append(mu_k)
            lam_K.append(lam_k)
            K += 1
        else:
            chosen_k = Klst[chosen_k]
            nK[chosen_k] += 1
            ks[sample_idx] = chosen_k
        if sample_idx % args.dmm_rebuild_freq == 0 or sample_idx == args.n_sample_load:
            for k in range(K):
                lam_K[k], mu_K[k] = sample_mu_lam(samples, nK, ks, k, _mu0, _ka0, _a0, _b0)
    dir_params.K, dir_params.lam_K, dir_params.mu_K, dir_params.n_K, dir_params.samples_k = (K, lam_K, mu_K, nK, ks)
    return dir_params

def train_flow(global_epoch, model_nice, opt, ds_inf_iter, n_iter, dir_params, args, log_step=0, prev_z_repr=None):
    if n_iter == 0:
        print('flow_opt=0, train ignored')
        return
    model_nice = model_nice.to(args.device)
    K, ks, ns, mu_K, lam_K = (dir_params.K, dir_params.samples_k, dir_params.n_K, dir_params.mu_K, dir_params.lam_K)
    for n_batch in range(n_iter):
        _, x, _, idx = next(ds_inf_iter)
        x = x.to(args.device)
        model_nice.train()
        opt.zero_grad()
        _, likelihood = model_nice(x, K, ks[idx.numpy()], ns, np.asarray(mu_K), np.asarray(lam_K))
        if isinstance(likelihood, int):
            continue
        loss = -torch.mean(likelihood)
        loss.backward()
        opt.step()

def init_dir_params(args):
    dir_params = Namespace(hyper=Namespace(a0=args.a0 * args.dim, b0=args.b0 * args.dim, mu0=np.zeros(args.dim), ka0=args.kappa0, logalpha=args.logalpha), K=0, n_K=[], lam_K=[], mu_K=[], samples_k=np.ones(args.n_sample_load, dtype=int) * -1)
    return dir_params

def init_flow_model(args):
    model_nice = NICE(data_dim=args.dim, num_coupling_layers=args.nice_nlayers, num_hidden_units=args.nice_units, device_name=args.device)
    opt = optim.Adam(model_nice.parameters(), args.lr)
    return (model_nice, opt)
