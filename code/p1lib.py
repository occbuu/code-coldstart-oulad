"""Core library for Paper 1: cold-start transfer learning for multi-task learner-outcome prediction on OULAD."""
import numpy as np, pandas as pd, time, warnings
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')
try:
    import torch, torch.nn as nn
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    torch = None

TEST_PRES = '2014J'

# ------------------------------------------------------------------ data
def load(path='.'):
    z = np.load(f'{path}/oulad_tensors.npz', allow_pickle=True)
    meta = pd.read_pickle(f'{path}/oulad_meta.pkl')
    return z['seq'], z['static'], meta, list(z['seq_names']), list(z['static_names'])

def snapshot(seq, static, meta, cutoff=70):
    """Return eligible enrollments at cutoff: observed weeks, labels, presentation-normalised engagement channel."""
    elig = (meta.date_unregistration.isna() | (meta.date_unregistration > cutoff)).values
    wc = cutoff // 7 + 1
    X = seq[elig, :wc, :].copy()
    m = meta[elig].reset_index(drop=True)
    # domain-normalisation channel: weekly total clicks as within-presentation percentile
    A = X.shape[2] - 3
    tot = np.expm1(X[:, :, :A]).sum(2)
    pct = np.zeros_like(tot)
    for _, idx in m.groupby(['code_module', 'code_presentation']).indices.items():
        pct[idx] = pd.DataFrame(tot[idx]).rank(pct=True).values
    X = np.concatenate([X, pct[:, :, None]], 2).astype(np.float32)
    S = static[elig].astype(np.float32)
    ys = m.final_result.map({'Pass': 1., 'Distinction': 1., 'Fail': 0.}).values.astype(float)  # Withdrawn -> NaN
    yw = (m.final_result == 'Withdrawn').values.astype(float)
    return X, S, ys, yw, m

def tabularise(X):
    """Hand-crafted aggregate features for classical baselines (no sequence model)."""
    T = X.shape[1]
    tsum = X.sum(1); last2 = X[:, -2:, :].sum(1); first = X[:, :max(2, T // 2), :].sum(1)
    act = (X[:, :, :-4].sum(2) > 0)
    nweeks = act.sum(1, keepdims=True)
    lastact = np.where(act.any(1), T - 1 - np.argmax(act[:, ::-1], 1), -1)[:, None]
    w = np.arange(T) - (T - 1) / 2
    slope = (X.sum(2) * w).sum(1, keepdims=True) / (w ** 2).sum()
    return np.concatenate([tsum, last2, last2 - first / max(1, T // 2) * 2, nweeks, lastact, slope], 1)

def split_coldstart(m, target, exclude_overlap=True):
    """Source = other modules, presentations before 2014J; pool = target module before 2014J; test = target 2014J.
    Learners appearing in the target test are removed from source and pool (strictly unseen learners)."""
    test = (m.code_module == target) & (m.code_presentation == TEST_PRES)
    pool = (m.code_module == target) & (m.code_presentation != TEST_PRES)
    src = (m.code_module != target) & (m.code_presentation != TEST_PRES)
    if exclude_overlap:
        ts = set(m.id_student[test])
        ov = m.id_student.isin(ts)
        pool &= ~ov; src &= ~ov
    return np.where(src)[0], np.where(pool)[0], np.where(test)[0]

# ------------------------------------------------------------------ metrics
def ece(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        k = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if k.any(): e += k.mean() * abs(y[k].mean() - p[k].mean())
    return e

def evaluate(ys, yw, ps, pw):
    k = ~np.isnan(ys)
    r = {}
    def safe(f, y, p):
        try: return f(y, p)
        except ValueError: return np.nan
    r['succ_auc'] = safe(roc_auc_score, ys[k], ps[k]); r['succ_ap'] = safe(average_precision_score, ys[k], ps[k])
    r['succ_brier'] = brier_score_loss(ys[k], ps[k]); r['succ_ece'] = ece(ys[k], ps[k])
    r['wd_auc'] = safe(roc_auc_score, yw, pw); r['wd_ap'] = safe(average_precision_score, yw, pw)
    r['wd_brier'] = brier_score_loss(yw, pw); r['wd_ece'] = ece(yw, pw); r['wd_prev'] = yw.mean()
    return r

# ------------------------------------------------------------------ classical baselines
def fit_predict_classical(name, Ftr, ys, yw, Fte, w=None, seed=0):
    import lightgbm as lgb, xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    out = []
    for y, task in [(ys, 's'), (yw, 'w')]:
        k = ~np.isnan(y); yy = y[k]; ww = None if w is None else w[k]
        if len(np.unique(yy)) < 2:
            out.append(np.full(len(Fte), yy.mean() if len(yy) else 0.5)); continue
        n = k.sum()
        if name == 'LR':
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            mdl.fit(Ftr[k], yy, logisticregression__sample_weight=ww)
        elif name == 'LightGBM':
            mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=max(5, min(20, n // 20)),
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=2)
            mdl.fit(Ftr[k], yy, sample_weight=ww)
        elif name == 'XGBoost':
            mdl = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                    min_child_weight=max(1, min(5, n // 50)), tree_method='hist', random_state=seed, n_jobs=2, verbosity=0)
            mdl.fit(Ftr[k], yy, sample_weight=ww)
        out.append(mdl.predict_proba(Fte)[:, 1])
    return out[0], out[1]

# ------------------------------------------------------------------ hybrid deep MTL model
if torch is not None:
    class HybridMTL(nn.Module):
        """GRU + temporal attention over weekly behaviour sequence, MLP over static profile; shared trunk; two task heads."""
        def __init__(self, c_in, s_in, h=64, d=0.2):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(c_in, 32), nn.ReLU())
            self.gru = nn.GRU(32, h, batch_first=True)
            self.att = nn.Linear(h, 1)
            self.stat = nn.Sequential(nn.Linear(s_in, 64), nn.ReLU(), nn.Dropout(d), nn.Linear(64, 32), nn.ReLU())
            self.trunk = nn.Sequential(nn.Linear(2 * h + 32, 128), nn.ReLU(), nn.Dropout(d), nn.Linear(128, 64), nn.ReLU())
            self.head_s = nn.Linear(64, 1); self.head_w = nn.Linear(64, 1)
        def forward(self, x, s, return_att=False):
            hseq, hl = self.gru(self.proj(x))
            a = torch.softmax(self.att(hseq).squeeze(-1), 1)
            ctx = (a.unsqueeze(-1) * hseq).sum(1)
            z = self.trunk(torch.cat([ctx, hl[-1], self.stat(s)], 1))
            o = (self.head_s(z).squeeze(-1), self.head_w(z).squeeze(-1))
            return (o, a) if return_att else o

    TASK_W = [1.0, 1.0]  # (success, withdrawal); set [1,0] or [0,1] for single-task ablations
    def masked_loss(o, ys, yw):
        bce = nn.functional.binary_cross_entropy_with_logits
        k = ~torch.isnan(ys)
        ls = bce(o[0][k], ys[k]) if k.any() else o[0].sum() * 0
        return TASK_W[0] * ls + TASK_W[1] * bce(o[1], yw)

    class Norm:
        def __init__(self, X, S):
            xs = X.reshape(-1, X.shape[2]); self.xm = xs.mean(0); self.xs = xs.std(0)
            self.sm = S.mean(0); self.ss = S.std(0)
            self.xs[self.xs < 1e-3] = 1.0; self.ss[self.ss < 1e-3] = 1.0   # constant columns in tiny samples
        def __call__(self, X, S):
            c = lambda a: np.clip(a, -10, 10).astype(np.float32)
            return c((X - self.xm) / self.xs), c((S - self.sm) / self.ss)

    def _t(*a): return [torch.as_tensor(v, dtype=torch.float32, device=DEV) for v in a]

    def train_dl(model, X, S, ys, yw, steps=300, lr=1e-3, bs=256, wd=1e-4, l2sp=0.0, ref=None, trainable=None, seed=0,
                 val=None, patience=None, eval_every=50, rand_trunc=False):
        """Generic trainer. l2sp>0 adds L2-SP penalty toward ref weights (Li et al., 2018). trainable: name prefixes to update."""
        g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        params = [p for n, p in model.named_parameters() if trainable is None or any(n.startswith(t) for t in trainable)]
        for n, p in model.named_parameters():
            p.requires_grad_(trainable is None or any(n.startswith(t) for t in trainable))
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd if l2sp == 0 else 0.0)
        Xt, St, yst, ywt = _t(X, S, ys, yw); n = len(Xt); bs = min(bs, n)
        refp = {k: v.detach().clone() for k, v in ref.items()} if ref is not None else None
        best, bstate, bad = -np.inf, None, 0
        model.train()
        for step in range(steps):
            idx = torch.randint(0, n, (bs,), generator=g).to(DEV)
            xb = Xt[idx]
            if rand_trunc:  # anytime training: random observation horizon, zero-pad the unseen future
                T = xb.shape[1]; cut = int(torch.randint(2, T + 1, (1,), generator=g))
                xb = xb.clone(); xb[:, cut:, :] = 0
            loss = masked_loss(model(xb, St[idx]), yst[idx], ywt[idx])
            if l2sp > 0:
                loss = loss + l2sp * sum(((p - refp[k]) ** 2).sum() for k, p in model.named_parameters() if p.requires_grad and not k.startswith('head'))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            if val is not None and (step + 1) % eval_every == 0:
                ps, pw = predict_dl(model, val[0], val[1]); r = evaluate(val[2], val[3], ps, pw)
                sc = np.nansum([r['succ_auc'], r['wd_auc']])
                if sc > best + 1e-4: best, bad, bstate = sc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if patience and bad >= patience: break
                model.train()
        if bstate is not None: model.load_state_dict(bstate)
        for p in model.parameters(): p.requires_grad_(True)
        return model

    @torch.no_grad()
    def predict_dl(model, X, S, bs=4096):
        model.eval(); P = [[], []]
        for i in range(0, len(X), bs):
            xb, sb = _t(X[i:i + bs], S[i:i + bs]); o = model(xb, sb)
            P[0].append(torch.sigmoid(o[0]).cpu().numpy()); P[1].append(torch.sigmoid(o[1]).cpu().numpy())
        return np.concatenate(P[0]), np.concatenate(P[1])
