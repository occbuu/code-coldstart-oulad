"""XAI (SHAP, Integrated Gradients, attention), fairness slicing and causal DML for Paper 1."""
import numpy as np, pandas as pd, copy, os, json, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb, shap
from sklearn.metrics import roc_auc_score
from p1lib import *
torch.set_num_threads(2)
seq, st, meta, SN, STN = load(os.environ.get('DATA', '.'))
X, S, ys, yw, m = snapshot(seq, st, meta, 70)
CH = list(SN) + ['click_pct_in_pres']
TABN = [f'sum_{c}' for c in CH] + [f'last2_{c}' for c in CH] + [f'trend_{c}' for c in CH] + ['n_active_weeks', 'last_active_week', 'click_slope'] + list(STN)
F = np.concatenate([tabularise(X), S], 1)
out = {}
# ---------------- 1. SHAP on pooled LightGBM (target-specific, use all 7 LOMO folds, aggregate) ----------------
rows = []
for tgt in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']:
    src, pool, te = split_coldstart(m, tgt); ii = np.r_[src, pool]
    for y, task in [(ys, 'success'), (yw, 'withdrawal')]:
        k = ~np.isnan(y[ii])
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2)
        mdl.fit(F[ii][k], y[ii][k])
        sv = shap.TreeExplainer(mdl).shap_values(F[te])
        sv = sv[1] if isinstance(sv, list) else sv
        if tgt == 'FFF': np.savez_compressed(f'results/shap_FFF_{task}.npz', sv=sv, X=F[te], names=np.array(TABN))
        rows += [dict(target=tgt, task=task, feature=n, mean_abs_shap=v) for n, v in zip(TABN, np.abs(sv).mean(0))]
pd.DataFrame(rows).to_csv('results/shap_global.csv', index=False); print('shap done', flush=True)

# ---------------- 2. Integrated gradients + attention on transferred hybrid model ----------------
def ig(model, x, s, task, steps=32):
    xb = torch.zeros_like(x); sb = torch.zeros_like(s); tot_x = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (xb + a * (x - xb)).requires_grad_(True); si = (sb + a * (s - sb)).requires_grad_(True)
        o = model(xi, si)[task].sum(); gx, = torch.autograd.grad(o, xi); tot_x += gx
    return ((x - xb) * tot_x / steps).detach().cpu().numpy()
igrows = []; att_all = []
for tgt in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']:
    src, pool, te = split_coldstart(m, tgt)
    pre = HybridMTL(X.shape[2], S.shape[1]).to(DEV); pre.load_state_dict(torch.load(f'models/pre_{tgt}_c70_s0.pt', map_location=DEV))
    rng = np.random.default_rng(0); idx = rng.permutation(src); nv = len(idx) // 10; norm = Norm(X[idx[nv:]], S[idx[nv:]])
    Xn, Sn = norm(X, S)
    mdl = copy.deepcopy(pre); train_dl(mdl, Xn[pool], Sn[pool], ys[pool], yw[pool], steps=1500, lr=3e-4, l2sp=1e-2, ref=pre.state_dict(), seed=0)
    mdl.eval()
    xt, stt = [torch.as_tensor(v, device=DEV) for v in (Xn[te], Sn[te])]
    with torch.no_grad(): (_, a) = mdl(xt, stt, return_att=True)
    att_all.append(pd.DataFrame(dict(target=tgt, week_idx=np.arange(a.shape[1]), attention=a.mean(0).cpu().numpy())))
    for task, nm in [(0, 'success'), (1, 'withdrawal')]:
        A = np.abs(ig(mdl, xt, stt, task)).mean(0)  # [T, C]
        for w in range(A.shape[0]):
            for c in range(A.shape[1]): igrows.append(dict(target=tgt, task=nm, week_idx=w, channel=CH[c], ig=A[w, c]))
    # fairness slicing on this transferred model
    ps, pw = predict_dl(mdl, Xn[te], Sn[te])
    info = pd.read_csv(os.environ.get('RAW', 'data_oulad') + '/studentInfo.csv', na_values=['?'])
    mt = m.iloc[te].merge(info, on=['code_module', 'code_presentation', 'id_student'], how='left')
    for attr in ['gender', 'age_band', 'disability', 'imd_band', 'highest_education']:
        for g, gi in mt.groupby(attr).indices.items():
            if len(gi) < 50: continue
            k = ~np.isnan(ys[te][gi])
            r = dict(target=tgt, attribute=attr, group=g, n=len(gi))
            try: r['succ_auc'] = roc_auc_score(ys[te][gi][k], ps[gi][k])
            except ValueError: r['succ_auc'] = np.nan
            try: r['wd_auc'] = roc_auc_score(yw[te][gi], pw[gi])
            except ValueError: r['wd_auc'] = np.nan
            out.setdefault('fair', []).append(r)
    print('ig', tgt, flush=True)
pd.DataFrame(igrows).to_csv('results/ig.csv', index=False); pd.concat(att_all).to_csv('results/attention.csv', index=False)
pd.DataFrame(out['fair']).to_csv('results/fairness.csv', index=False)

# ---------------- 3. Causal: DoubleML PLR, effect of early forum participation on success ----------------
import doubleml as dml
Xc, Sc, ysc, ywc, mc = snapshot(seq, st, meta, 28)
fi = CH.index('forumng')
D = (np.expm1(Xc[:, 1:, fi]).sum(1) > 0).astype(float)            # any forum activity in course weeks 0-3
Y = (mc.final_result.isin(['Pass', 'Distinction'])).astype(float).values  # pass (fail or withdrawal = 0)
others = [i for i in range(len(CH) - 1) if i != fi]
W = np.concatenate([Xc[:, :, others].sum(1), Xc[:, 0, :], Sc, pd.get_dummies(mc.code_module).values.astype(float),
                    Xc[:, :, -1].mean(1, keepdims=True)], 1)
crows = []
def fit(idx, label):
    d = dml.DoubleMLData.from_arrays(W[idx], Y[idx], D[idx])
    ml = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=2)
    mc_ = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=2)
    irm = dml.DoubleMLIRM(d, ml_g=mc_(), ml_m=mc_(), n_folds=5, trimming_threshold=0.02)
    irm.fit()
    ci = irm.confint().values[0]
    r = dict(scope=label, n=len(idx), treated_share=D[idx].mean(), ate=irm.coef[0], se=irm.se[0], ci_low=ci[0], ci_high=ci[1], p=irm.pval[0],
             naive_diff=Y[idx][D[idx] == 1].mean() - Y[idx][D[idx] == 0].mean())
    try:
        irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
        sp = irm.sensitivity_params; r['rv'] = float(np.ravel(sp['rv'])[0]); r['ci_low_sens'] = float(np.ravel(sp['ci']['lower'])[0])
    except Exception as e: print('sens', e)
    crows.append(r); print(r, flush=True)
fit(np.arange(len(Y)), 'All modules')
for mod, idx in mc.groupby('code_module').indices.items(): fit(idx, mod)
pd.DataFrame(crows).to_csv('results/causal_dml.csv', index=False)
