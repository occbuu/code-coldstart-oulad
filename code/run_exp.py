"""Paper 1 experiments. Usage: python code/run_exp.py main|ablation|stream|anytime [N]"""
import sys, os, time, copy, json, numpy as np, pandas as pd
import p1lib as L
from p1lib import *
if torch is not None: torch.set_num_threads(int(os.environ.get('NT', '2')))
MSET = os.environ.get('MSET', 'all')  # 'classical' | 'hybrid' | 'all'
EXP = sys.argv[1]; SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.makedirs('results', exist_ok=True); os.makedirs('models', exist_ok=True)
seq, st, meta, SN, STN = load(os.environ.get('DATA', '.'))
KS = [0, 25, 50, 100, 200, 'full']
TARGETS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']

def data(cutoff):
    X, S, ys, yw, m = snapshot(seq, st, meta, cutoff)
    F = np.concatenate([tabularise(X), S], 1)
    return X, S, ys, yw, m, F

def pretrain(X, S, ys, yw, src, tag, seed=0, steps=3000):
    path = f'models/pre_{tag}_s{seed}.pt'
    rng = np.random.default_rng(seed); idx = rng.permutation(src); nv = len(idx) // 10
    va, tr = idx[:nv], idx[nv:]
    norm = Norm(X[tr], S[tr]); Xn, Sn = norm(X, S)
    model = HybridMTL(X.shape[2], S.shape[1]).to(DEV)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEV)); return model, norm
    t = time.time()
    train_dl(model, Xn[tr], Sn[tr], ys[tr], yw[tr], steps=steps, lr=1e-3, seed=seed,
             val=(Xn[va], Sn[va], ys[va], yw[va]), patience=8, eval_every=100)
    torch.save(model.state_dict(), path); print(f'  pretrain {tag} {time.time()-t:.0f}s', flush=True)
    return model, norm

def ft_steps(k): return 300 if k != 'full' else 1500

def run_methods(X, S, ys, yw, F, src, tr, te, pre, norm, seed, methods):
    out = {}
    Xn, Sn = norm(X, S)
    for meth in methods:
        t = time.time()
        if meth in ('LR-T', 'LightGBM-T', 'XGBoost-T'):
            if len(tr) == 0: continue
            ps, pw = fit_predict_classical(meth[:-2], F[tr], ys[tr], yw[tr], F[te], seed=seed)
        elif meth == 'LightGBM-S':
            ps, pw = fit_predict_classical('LightGBM', F[src], ys[src], yw[src], F[te], seed=seed)
        elif meth == 'LightGBM-P':
            if len(tr) == 0: continue
            ii = np.concatenate([src, tr]); w = np.r_[np.ones(len(src)), np.full(len(tr), 5.0)]
            ps, pw = fit_predict_classical('LightGBM', F[ii], ys[ii], yw[ii], F[te], w=w, seed=seed)
        elif meth == 'Hybrid-S':
            ps, pw = predict_dl(pre, Xn[te], Sn[te])
        elif meth == 'Hybrid-T':
            if len(tr) == 0: continue
            nt = Norm(X[tr], S[tr]); a, b = nt(X, S)
            mdl = HybridMTL(X.shape[2], S.shape[1]).to(DEV)
            train_dl(mdl, a[tr], b[tr], ys[tr], yw[tr], steps=ft_steps(K), lr=1e-3, seed=seed)
            ps, pw = predict_dl(mdl, a[te], b[te])
        elif meth.startswith('Hybrid-FT') or meth == 'Hybrid-Head':
            if len(tr) == 0: continue
            mdl = copy.deepcopy(pre); ref = {k: v for k, v in pre.state_dict().items()}
            kw = dict(steps=ft_steps(K), lr=3e-4, seed=seed)
            if meth == 'Hybrid-Head': kw.update(trainable=['head'], lr=1e-3)
            if meth == 'Hybrid-FT-L2SP': kw.update(l2sp=1e-2, ref=ref)
            train_dl(mdl, Xn[tr], Sn[tr], ys[tr], yw[tr], **kw)
            ps, pw = predict_dl(mdl, Xn[te], Sn[te])
        elif meth == 'Hybrid-LPFT':  # linear probe then L2-SP fine-tune (Kumar et al., 2022)
            if len(tr) == 0: continue
            mdl = copy.deepcopy(pre)
            train_dl(mdl, Xn[tr], Sn[tr], ys[tr], yw[tr], steps=ft_steps(K), lr=1e-3, trainable=['head'], seed=seed)
            ref = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
            train_dl(mdl, Xn[tr], Sn[tr], ys[tr], yw[tr], steps=ft_steps(K), lr=1e-4, l2sp=1e-2, ref=ref, seed=seed)
            ps, pw = predict_dl(mdl, Xn[te], Sn[te])
        elif meth == 'Ens-Head+LGBM-P':  # probability average of head-only transfer and pooled LightGBM
            if len(tr) == 0: continue
            mdl = copy.deepcopy(pre)
            train_dl(mdl, Xn[tr], Sn[tr], ys[tr], yw[tr], steps=ft_steps(K), lr=1e-3, trainable=['head'], seed=seed)
            a, b = predict_dl(mdl, Xn[te], Sn[te])
            ii = np.concatenate([src, tr]); w = np.r_[np.ones(len(src)), np.full(len(tr), 5.0)]
            c, d = fit_predict_classical('LightGBM', F[ii], ys[ii], yw[ii], F[te], w=w, seed=seed)
            ps, pw = (a + c) / 2, (b + d) / 2
        r = evaluate(ys[te], yw[te], ps, pw); r['time_s'] = time.time() - t
        out[meth] = (r, ps, pw)
    return out

ALL = ['LR-T', 'XGBoost-T', 'LightGBM-T', 'LightGBM-S', 'LightGBM-P', 'Hybrid-S', 'Hybrid-T', 'Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP']

if EXP == 'main':
    X, S, ys, yw, m, F = data(70)
    rows = []; fn = f'results/main_{MSET}.csv'; preds = {}
    for tgt in TARGETS:
        src, pool, te = split_coldstart(m, tgt)
        pre, norm = pretrain(X, S, ys, yw, src, f'{tgt}_c70') if MSET != 'classical' else (None, lambda a, b: (a, b))
        for K in KS:
            for seed in range(SEEDS if K not in (0, 'full') else (1 if K == 0 else min(SEEDS, 3))):
                rng = np.random.default_rng(1000 + seed)
                tr = pool if K == 'full' else (rng.choice(pool, K, replace=False) if K else np.array([], int))
                meths = ['LightGBM-S', 'Hybrid-S'] if K == 0 else [x for x in ALL if x not in ('LightGBM-S', 'Hybrid-S')]
                if MSET == 'classical': meths = [x for x in meths if not x.startswith('Hybrid')]
                if MSET == 'hybrid': meths = [x for x in meths if x.startswith('Hybrid')]
                if MSET == 'extra': meths = [] if K == 0 else ['Hybrid-LPFT']
                res = run_methods(X, S, ys, yw, F, src, tr, te, pre, norm, seed, meths)
                for meth, (r, ps, pw) in res.items():
                    rows.append(dict(target=tgt, k=K, seed=seed, method=meth, n_train=len(tr), n_test=len(te), **r))
                    if seed == 0: preds[(tgt, str(K), meth)] = (ps.astype(np.float32), pw.astype(np.float32))
                pd.DataFrame(rows).to_csv(fn, index=False)
                print(tgt, K, seed, ' '.join(f"{k}:{v[0]['succ_auc']:.3f}/{v[0]['wd_auc']:.3f}" for k, v in res.items()), flush=True)
    np.savez_compressed(f'results/main_preds_{MSET}.npz', **{'|'.join(k) + '|s': v[0] for k, v in preds.items()}, **{'|'.join(k) + '|w': v[1] for k, v in preds.items()})

elif EXP == 'ablation':  # k=100: MTL vs STL, sequence on/off, domain-norm channel on/off
    X0, S, ys, yw, m, F = data(70); rows = []
    variants = {'MTL (full)': ([1, 1], 'all'), 'STL-success': ([1, 0], 'all'), 'STL-withdrawal': ([0, 1], 'all'),
                'MTL w/o sequence': ([1, 1], 'noseq'), 'MTL w/o domain-norm channel': ([1, 1], 'nonorm'),
                'MTL w/o assessment channels': ([1, 1], 'noassess')}
    for tgt in TARGETS:
        src, pool, te = split_coldstart(m, tgt)
        for name, (tw, mode) in variants.items():
            X = X0.copy()
            if mode == 'noseq': X[:] = 0
            if mode == 'nonorm': X[:, :, -1] = 0
            if mode == 'noassess': X[:, :, -4:-1] = 0
            L.TASK_W[:] = tw
            pre, norm = pretrain(X, S, ys, yw, src, f'{tgt}_c70' if (mode == 'all' and tw == [1, 1]) else f'{tgt}_c70_{mode}_{tw[0]}{tw[1]}')
            for seed in range(SEEDS):
                rng = np.random.default_rng(1000 + seed); tr = rng.choice(pool, 100, replace=False); K = 100
                res = run_methods(X, S, ys, yw, F, src, tr, te, pre, norm, seed, ['Hybrid-FT-L2SP'])
                r = res['Hybrid-FT-L2SP'][0]; rows.append(dict(target=tgt, variant=name, seed=seed, **r))
            L.TASK_W[:] = [1, 1]
            print(tgt, name, np.mean([x['succ_auc'] for x in rows[-SEEDS:]]).round(3), np.mean([x['wd_auc'] for x in rows[-SEEDS:]]).round(3), flush=True)
            pd.DataFrame(rows).to_csv('results/ablation.csv', index=False)

elif EXP == 'stream':  # target labels arrive in batches of 100 learners, ordered by registration time
    X, S, ys, yw, m, F = data(70); rows = []
    for tgt in TARGETS:
        src, pool, te = split_coldstart(m, tgt)
        order = pool[np.lexsort((m.date_registration.fillna(0).values[pool], m.code_presentation.values[pool]))]
        pre, norm = pretrain(X, S, ys, yw, src, f'{tgt}_c70'); Xn, Sn = norm(X, S)
        rng = np.random.default_rng(0); sv = rng.choice(src, 2000, replace=False)
        cont = copy.deepcopy(pre); ref = {k: v.clone() for k, v in pre.state_dict().items()}
        nb = min(10, len(order) // 100)
        for b in range(1, nb + 1):
            new = order[(b - 1) * 100: b * 100]; seen = order[: b * 100]
            train_dl(cont, Xn[new], Sn[new], ys[new], yw[new], steps=100, lr=3e-4, l2sp=1e-2, ref=ref, seed=b)
            ref = {k: v.clone() for k, v in cont.state_dict().items()}  # anchor moves with the stream (online L2-SP)
            re = copy.deepcopy(pre); train_dl(re, Xn[seen], Sn[seen], ys[seen], yw[seen], steps=300, lr=3e-4, l2sp=1e-2, ref=pre.state_dict(), seed=b)
            for name, mdl in [('Continual L2-SP (online)', cont), ('Re-fine-tune on all seen', re), ('Source-only', pre)]:
                ps, pw = predict_dl(mdl, Xn[te], Sn[te]); r = evaluate(ys[te], yw[te], ps, pw)
                ps2, pw2 = predict_dl(mdl, Xn[sv], Sn[sv]); r2 = evaluate(ys[sv], yw[sv], ps2, pw2)
                rows.append(dict(target=tgt, batch=b, n_seen=len(seen), method=name, **r, src_succ_auc=r2['succ_auc'], src_wd_auc=r2['wd_auc']))
            ps, pw = fit_predict_classical('LightGBM', F[seen], ys[seen], yw[seen], F[te]); r = evaluate(ys[te], yw[te], ps, pw)
            rows.append(dict(target=tgt, batch=b, n_seen=len(seen), method='LightGBM-T retrain', **r))
            print(tgt, b, ' '.join(f"{x['method'][:10]}:{x['succ_auc']:.3f}/{x['wd_auc']:.3f}" for x in rows[-4:]), flush=True)
        pd.DataFrame(rows).to_csv('results/stream.csv', index=False)

elif EXP == 'anytime':
    rows = []
    for C in [14, 28, 56, 84, 112]:  # day 70 comes from the main experiment
        X, S, ys, yw, m, F = data(C)
        for tgt in TARGETS:
            src, pool, te = split_coldstart(m, tgt)
            pre, norm = pretrain(X, S, ys, yw, src, f'{tgt}_c{C}', steps=2000)
            for K in [100, 'full']:
                for seed in range(2 if K == 100 else 1):
                    rng = np.random.default_rng(1000 + seed)
                    tr = pool if K == 'full' else rng.choice(pool, K, replace=False)
                    res = run_methods(X, S, ys, yw, F, src, tr, te, pre, norm, seed, os.environ.get('AMETH', 'LightGBM-T,LightGBM-P,Hybrid-FT-L2SP').split(','))
                    for meth, (r, _, _) in res.items(): rows.append(dict(cutoff=C, target=tgt, k=K, seed=seed, method=meth, n_test=len(te), **r))
            print(C, tgt, flush=True); pd.DataFrame(rows).to_csv(os.environ.get('AOUT', 'results/anytime.csv'), index=False)
