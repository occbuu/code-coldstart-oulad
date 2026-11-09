"""Measure source-target separability at day 70."""
import os

import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from p1lib import load, snapshot, tabularise, split_coldstart
seq, st, meta, *_ = load(os.environ.get('DATA', 'code'))
X, S, ys, yw, m = snapshot(seq, st, meta, 70); F = np.concatenate([tabularise(X), S], 1)
rows = []
for t in ['AAA','BBB','CCC','DDD','EEE','FFF','GGG']:
    src, pool, te = split_coldstart(m, t); tg = np.r_[pool, te]
    rng = np.random.default_rng(0); s = rng.choice(src, min(len(src), 4000), replace=False); g = rng.choice(tg, min(len(tg), 4000), replace=False)
    Z = F[np.r_[s, g]]; d = np.r_[np.zeros(len(s)), np.ones(len(g))]
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, n_jobs=2), Z, d, cv=5, method='predict_proba')[:, 1]
    err = ((p > .5) != d).mean()
    rows.append(dict(target=t, pad=2 * (1 - 2 * err), n_src=len(src), n_pool=len(pool), n_test=len(te),
                     test_succ_rate=np.nanmean(ys[te]), test_wd_rate=yw[te].mean(),
                     src_succ_rate=np.nanmean(ys[src]), src_wd_rate=yw[src].mean()))
pd.DataFrame(rows).to_csv('results/domain_shift.csv', index=False); print(pd.DataFrame(rows).round(3))
