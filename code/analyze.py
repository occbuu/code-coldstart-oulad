"""Build all tables (LaTeX) and figures (PDF) for Paper 1 from results/*.csv."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, os, itertools
from scipy import stats
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.edgecolor': '#888', 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
FIG = 'paper/figures'; TAB = 'paper/tables'; os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
COL = {'LightGBM-T': PAL[0], 'LightGBM-P': PAL[1], 'Hybrid-T': PAL[2], 'Hybrid-Head': PAL[3], 'Hybrid-FT': PAL[4], 'Hybrid-FT-L2SP': PAL[6],
       'LR-T': '#777777', 'XGBoost-T': '#444444', 'Hybrid-LPFT': PAL[7], 'Ens-Head+LGBM-P': PAL[5]}
MK = {'LightGBM-T': 'o', 'LightGBM-P': 's', 'Hybrid-T': '^', 'Hybrid-Head': 'v', 'Hybrid-FT': 'D', 'Hybrid-FT-L2SP': 'P', 'Hybrid-LPFT': 'X', 'Ens-Head+LGBM-P': '*'}
ORDER = ['LR-T', 'XGBoost-T', 'LightGBM-T', 'LightGBM-S', 'LightGBM-P', 'Hybrid-S', 'Hybrid-T', 'Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP', 'Hybrid-LPFT', 'Ens-Head+LGBM-P']
KLAB = ['0', '25', '50', '100', '200', 'full']
def kk(x): return str(x)

res = pd.concat([pd.read_csv(f) for f in ['results/main_classical.csv', 'results/main_hybrid.csv', 'results/main_extra.csv'] if os.path.exists(f)])
res['k'] = res.k.astype(str)
res = res[res.method != 'Ens-Head+LGBM-P']
# post-hoc ensemble (seed 0): average of head-only transfer and pooled LightGBM probabilities
import sys; sys.path.insert(0, 'code'); sys.path.insert(0, '.')
from p1lib import load, snapshot, split_coldstart, evaluate
_seq, _st, _meta, _, _ = load(os.environ.get('DATA', '.')); _X, _S, YS, YW, MM = snapshot(_seq, _st, _meta, 70)
PC = np.load('results/main_preds_classical.npz'); PH = np.load('results/main_preds_hybrid.npz'); erows = []
for tgt in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']:
    _, _, te = split_coldstart(MM, tgt)
    for K in ['25', '50', '100', '200', 'full']:
        ps = (PH[f'{tgt}|{K}|Hybrid-Head|s'] + PC[f'{tgt}|{K}|LightGBM-P|s']) / 2; pw = (PH[f'{tgt}|{K}|Hybrid-Head|w'] + PC[f'{tgt}|{K}|LightGBM-P|w']) / 2
        erows.append(dict(target=tgt, k=K, seed=0, method='Ens-Head+LGBM-P', n_train=np.nan, n_test=len(te), **evaluate(YS[te], YW[te], ps, pw)))
res = pd.concat([res, pd.DataFrame(erows)], ignore_index=True)
res.to_csv('results/main_all.csv', index=False)

# ---------------- Table: data per module ----------------
ds = pd.read_csv('results/domain_shift.csv')
t1 = ds.copy()
with open(f'{TAB}/tab_data.tex', 'w') as f:
    for _, r in t1.iterrows():
        f.write(f"{r.target} & {r.n_src:,} & {int(r.n_pool):,} & {int(r.n_test):,} & {r.test_succ_rate:.3f} & {r.test_wd_rate:.3f} & {r.pad:.2f}\\\\\n")

# ---------------- Table: main results (mean ± sd over targets x seeds) ----------------
def agg(df, col): g = df.groupby('method')[col]; return g.mean(), g.std()
lines = []
for K in ['0', '25', '50', '100', '200', 'full']:
    d = res[res.k == K]
    if d.empty: continue
    ms, ss = agg(d, 'succ_auc'); mw, sw = agg(d, 'wd_auc'); ma, sa = agg(d, 'wd_ap'); mb, sb = agg(d, 'succ_brier')
    best = {c: v.max() for c, v in [('s', ms), ('w', mw), ('a', ma)]}; bestb = mb.min()
    lines.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{$k={K if K != 'full' else '\\text{all}'}$}}}}\\\\")
    for meth in ORDER:
        if meth not in ms.index: continue
        def fmt(m, s, b, hi=True):
            t = f"{m:.3f}" + (f"\\,$\\pm$\\,{s:.3f}" if not np.isnan(s) else '')
            return f"\\textbf{{{t}}}" if abs(m - b) < 5e-4 else t
        lines.append(f"{meth} & {fmt(ms[meth], ss[meth], best['s'])} & {fmt(mw[meth], sw[meth], best['w'])} & {fmt(ma[meth], sa[meth], best['a'])} & {fmt(mb[meth], sb[meth], bestb)} & {d[d.method == meth].shape[0]}\\\\")
open(f'{TAB}/tab_main.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------- Wilcoxon (paired over target x seed) best transfer vs baselines ----------------
rows = []
for K in ['25', '50', '100', '200', 'full']:
    d = res[res.k == K]
    for metric in ['succ_auc', 'wd_auc', 'wd_ap']:
        p = d.pivot_table(index=['target', 'seed'], columns='method', values=metric)
        for a, b in [('Hybrid-Head', 'LightGBM-T'), ('Hybrid-Head', 'LightGBM-P'), ('Hybrid-Head', 'Hybrid-FT-L2SP'), ('Hybrid-FT-L2SP', 'Hybrid-FT'), ('Hybrid-FT-L2SP', 'Hybrid-T'), ('LightGBM-P', 'LightGBM-T'), ('Ens-Head+LGBM-P', 'LightGBM-P'), ('Hybrid-LPFT', 'Hybrid-FT-L2SP')]:
            if a not in p or b not in p: continue
            x = (p[a] - p[b]).dropna()
            if len(x) < 5: continue
            w = stats.wilcoxon(x) if (x != 0).any() else None
            rows.append(dict(k=K, metric=metric, a=a, b=b, n=len(x), mean_diff=x.mean(), win=(x > 0).mean(), p=w.pvalue if w else 1.0))
wt = pd.DataFrame(rows)
if len(wt):
    for (K, metric), g in wt.groupby(['k', 'metric']):  # Holm within each k x metric family
        o = np.argsort(g.p.values); m = len(o); adj = np.empty(m); run = 0
        for i, j in enumerate(o): run = max(run, min(1, (m - i) * g.p.values[j])); adj[j] = run
        wt.loc[g.index, 'p_holm'] = adj
    wt.to_csv('results/wilcoxon.csv', index=False)
    sel = wt[wt.k.isin(['25', '100', 'full']) & wt.metric.isin(['succ_auc', 'wd_auc'])]
    with open(f'{TAB}/tab_wilcoxon.tex', 'w') as f:
        last = None
        for _, r in sel.sort_values(['k', 'metric'], key=lambda c: c.map({'25': 0, '100': 1, 'full': 2}) if c.name == 'k' else c).iterrows():
            if last is not None and (r.k, r.metric) != last: f.write('\\midrule\n')
            last = (r.k, r.metric)
            star = '***' if r.p_holm < .001 else '**' if r.p_holm < .01 else '*' if r.p_holm < .05 else ''
            kl = 'all' if r.k == 'full' else r.k
            f.write(f"{kl} & {dict(succ_auc='Success AUC', wd_auc='Withdrawal AUC', wd_ap='Withdrawal AP')[r.metric]} & {r.a} vs {r.b} & {r.n} & {r.mean_diff:+.3f} & {r.win:.2f} & {r.p_holm:.2g}{star}\\\\\n")

# ---------------- adaptation vs zero-shot ----------------
zs = res[(res.method == 'Hybrid-S') & (res.k == '0')].set_index('target')
zrows = []
for K in ['25', '50', '100', '200', 'full']:
    for meth in ['Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP', 'Hybrid-LPFT']:
        d = res[(res.method == meth) & (res.k == K)].groupby('target')[['succ_auc', 'wd_auc']].mean()
        if d.empty: continue
        for mt in ['succ_auc', 'wd_auc']:
            dd = d[mt] - zs.loc[d.index, mt]
            zrows.append(dict(k=K, method=meth, metric=mt, mean_gain=dd.mean(), n_improved=int((dd > 0).sum()), n=len(dd)))
zr = pd.DataFrame(zrows); zr.to_csv('results/adapt_vs_zeroshot.csv', index=False)
with open(f'{TAB}/tab_zeroshot.tex', 'w') as f:
    ms = [x for x in ['Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP', 'Hybrid-LPFT'] if x in zr.method.unique()]
    f.write('\\begin{tabular}{ll' + 'c' * len(ms) + '}\n\\toprule\n$k$ & Metric & ' + ' & '.join(m.replace('Hybrid-', '') for m in ms) + '\\\\\n\\midrule\n')
    for K in ['25', '50', '100', '200', 'full']:
        for mt in ['succ_auc', 'wd_auc']:
            cells = []
            for m in ms:
                r = zr[(zr.k == K) & (zr.method == m) & (zr.metric == mt)]
                cells.append('--' if r.empty else f"{r.mean_gain.iloc[0]:+.3f} ({r.n_improved.iloc[0]}/{r.n.iloc[0]})")
            f.write(f"{(('all' if K == 'full' else K) if mt == 'succ_auc' else '')} & {'Succ.' if mt == 'succ_auc' else 'Wd.'} & " + ' & '.join(cells) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

# ---------------- Figure: learning curves ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
xs = np.arange(1, 6); xl = ['25', '50', '100', '200', 'all']
for ax, (metric, title) in zip(axes, [('succ_auc', 'Academic success — ROC-AUC'), ('wd_auc', 'Withdrawal — ROC-AUC'), ('wd_ap', 'Withdrawal — average precision')]):
    for meth in ['LightGBM-T', 'LightGBM-P', 'Hybrid-T', 'Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP', 'Hybrid-LPFT', 'Ens-Head+LGBM-P']:
        d = res[res.method == meth]
        if d.empty: continue
        g = d.groupby('k')[metric].agg(['mean', 'sem']).reindex(['25', '50', '100', '200', 'full'])
        ax.plot(xs, g['mean'], color=COL[meth], marker=MK[meth], ms=4, label=meth)
        ax.fill_between(xs, g['mean'] - 1.96 * g['sem'], g['mean'] + 1.96 * g['sem'], color=COL[meth], alpha=0.10, lw=0)
    for meth, ls in [('LightGBM-S', '--'), ('Hybrid-S', ':')]:
        d = res[(res.method == meth) & (res.k == '0')]
        if len(d): ax.axhline(d[metric].mean(), color='#666', ls=ls, lw=1.1, label=f'{meth} (k=0)')
    ax.set_xticks(xs, xl); ax.set_xlabel('labelled target learners $k$'); ax.set_title(title, fontsize=8.5, loc='left')
axes[0].set_ylabel('mean over 7 targets (95% CI)')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17), fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_learning_curves.pdf'); plt.close(fig)

# ---------------- Figure: per-target gain heatmap ----------------
HM = os.environ.get('HM', 'Hybrid-Head')
best_tl = res[res.method.isin([HM])].groupby(['target', 'k', 'method'])[['succ_auc', 'wd_auc']].mean()
best_bl = res[res.method.isin(['LR-T', 'XGBoost-T', 'LightGBM-T', 'LightGBM-P'])].groupby(['target', 'k', 'method'])[['succ_auc', 'wd_auc']].mean()
if len(best_tl):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, metric, nm in zip(axes, ['succ_auc', 'wd_auc'], ['Success AUC', 'Withdrawal AUC']):
        a = best_tl.xs(HM, level='method')[metric].unstack('k')[['25', '50', '100', '200', 'full']]
        b = best_bl[metric].groupby(['target', 'k']).max().unstack('k')[['25', '50', '100', '200', 'full']]
        D = (a - b)
        v = np.nanmax(np.abs(D.values))
        im = ax.imshow(D.values, cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
        ax.set_xticks(range(5), xl); ax.set_yticks(range(len(D)), D.index); ax.grid(False)
        for i, j in itertools.product(range(D.shape[0]), range(D.shape[1])):
            ax.text(j, i, f"{D.values[i, j]:+.3f}", ha='center', va='center', fontsize=6.5, color='white' if abs(D.values[i, j]) > 0.6 * v else '#222')
        ax.set_title(f'{nm}: {HM} − best target/pooled baseline', fontsize=8.5, loc='left'); ax.set_xlabel('$k$')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig4_gain_heatmap.pdf'); plt.close(fig)

# ---------------- Figure: critical difference (Nemenyi) at k=50 and k=all ----------------
def cd_plot(ax, K, metric):
    d = res[(res.k == K) & (~res.method.isin(['LightGBM-S', 'Hybrid-S']))]
    p = d.groupby(['target', 'method'])[metric].mean().unstack().dropna(axis=1)
    ranks = p.rank(axis=1, ascending=False).mean().sort_values()
    N, kM = p.shape; fr = stats.friedmanchisquare(*[p[c] for c in p.columns])
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}[kM]
    cd = q * np.sqrt(kM * (kM + 1) / (6 * N))
    ax.set_xlim(kM + 0.3, 0.7); ax.set_ylim(-0.05 - 0.11 * ((kM + 1) // 2), 0.32); ax.grid(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_visible(False)
    ax.set_yticks([]); ax.xaxis.set_ticks_position('top'); ax.spines['top'].set_visible(True); ax.set_xticks(range(1, kM + 1))
    ax.tick_params(axis='x', labelsize=7, pad=1)
    items = list(ranks.items()); half = (len(items) + 1) // 2
    for i, (mname, r) in enumerate(items):
        right = i < half; j = i if right else len(items) - 1 - i
        y = -0.08 - 0.11 * j; xt = 0.75 if right else kM + 0.25
        ax.plot([r, r, xt], [0.3, y, y], color=COL.get(mname, '#444'), lw=0.9)
        ax.text(xt + (-0.05 if right else 0.05), y, f"{mname} ({r:.2f})", ha='left' if right else 'right', va='center', fontsize=6.6)
    ax.plot([1, 1 + cd], [0.25, 0.25], color='k', lw=1.6); ax.text(1 + cd / 2, 0.27, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_title(f"$k$={'all' if K == 'full' else K}, {dict(succ_auc='success AUC', wd_auc='withdrawal AUC')[metric]} (Friedman p={fr.pvalue:.2g})", fontsize=8, loc='left', pad=14)
    return dict(k=K, metric=metric, friedman_chi2=fr.statistic, friedman_p=fr.pvalue, cd=cd, **{f'rank_{m}': r for m, r in ranks.items()})
if (res.method == 'Hybrid-FT-L2SP').any():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6)); frows = []
    for ax, (K, metric) in zip(axes.ravel(), [('50', 'succ_auc'), ('50', 'wd_auc'), ('full', 'succ_auc'), ('full', 'wd_auc')]):
        frows.append(cd_plot(ax, K, metric))
    fig.tight_layout(); fig.savefig(f'{FIG}/fig5_cd.pdf'); plt.close(fig); pd.DataFrame(frows).to_csv('results/friedman.csv', index=False)
print('main done')

# ---------------- Ablation ----------------
if os.path.exists('results/ablation.csv'):
    ab = pd.read_csv('results/ablation.csv')
    g = ab.groupby('variant')[['succ_auc', 'wd_auc', 'wd_ap']].agg(['mean', 'std'])
    ref = ab[ab.variant == 'MTL (full)'].set_index(['target', 'seed'])
    with open(f'{TAB}/tab_ablation.tex', 'w') as f:
        for v in ['MTL (full)', 'STL-success', 'STL-withdrawal', 'MTL w/o sequence', 'MTL w/o assessment channels', 'MTL w/o domain-norm channel']:
            if v not in g.index: continue
            cur = ab[ab.variant == v].set_index(['target', 'seed'])
            cells = []
            for mtr in ['succ_auc', 'wd_auc', 'wd_ap']:
                if (v == 'STL-success' and mtr != 'succ_auc') or (v == 'STL-withdrawal' and mtr == 'succ_auc'): cells.append('--'); continue
                dlt = (cur[mtr] - ref[mtr]).dropna(); pv = stats.wilcoxon(dlt).pvalue if v != 'MTL (full)' and (dlt != 0).any() else np.nan
                s = f"{g.loc[v, (mtr, 'mean')]:.3f}\\,$\\pm$\\,{g.loc[v, (mtr, 'std')]:.3f}"
                if v != 'MTL (full)': s += f" ({dlt.mean():+.3f}{'$^{*}$' if pv < .05 else ''})"
                cells.append(s)
            f.write(f"{v} & " + ' & '.join(cells) + '\\\\\n')

# ---------------- Streaming ----------------
if os.path.exists('results/stream.csv'):
    sr = pd.read_csv('results/stream.csv')
    SC = {'Continual L2-SP (online)': PAL[6], 'Re-fine-tune on all seen': PAL[4], 'LightGBM-T retrain': PAL[0], 'Source-only': '#777'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, metric, ttl in zip(axes, ['succ_auc', 'wd_auc', 'src_succ_auc'], ['Target success AUC', 'Target withdrawal AUC', 'Source success AUC (forgetting)']):
        for meth, c in SC.items():
            d = sr[sr.method == meth]
            if metric not in d or d[metric].isna().all(): continue
            g = d.groupby('n_seen')[metric].agg(['mean', 'sem'])
            ax.plot(g.index, g['mean'], color=c, label=meth, ls=':' if meth == 'Source-only' else '-')
            ax.fill_between(g.index, g['mean'] - 1.96 * g['sem'], g['mean'] + 1.96 * g['sem'], color=c, alpha=.1, lw=0)
        ax.set_title(ttl, fontsize=8.5, loc='left'); ax.set_xlabel('labelled target learners seen')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(f'{FIG}/fig6_stream.pdf'); plt.close(fig)

# ---------------- Anytime ----------------
if os.path.exists('results/anytime.csv'):
    an = pd.concat([pd.read_csv(f) for f in ['results/anytime.csv', 'results/anytime_head.csv'] if os.path.exists(f)]); an['k'] = an.k.astype(str)
    m70 = res[(res.method.isin(['LightGBM-T', 'LightGBM-P', 'Hybrid-FT-L2SP', 'Hybrid-Head'])) & (((res.k == '100') & (res.seed < 2)) | ((res.k == 'full') & (res.seed == 0)))].assign(cutoff=70)
    an = pd.concat([an, m70[an.columns.intersection(m70.columns)]]).sort_values('cutoff'); an.to_csv('results/anytime_with70.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, metric, ttl in zip(axes, ['succ_auc', 'wd_auc'], ['Academic success — ROC-AUC', 'Withdrawal — ROC-AUC']):
        for meth in ['LightGBM-T', 'LightGBM-P', 'Hybrid-FT-L2SP', 'Hybrid-Head']:
            for K, ls in [('100', '--'), ('full', '-')]:
                d = an[(an.method == meth) & (an.k == K)].groupby('cutoff')[metric].mean()
                ax.plot(d.index / 7, d.values, color=COL[meth], ls=ls, marker=MK[meth], ms=3, label=f"{meth}, k={'all' if K=='full' else K}")
        ax.set_xlabel('observation cutoff (course week)'); ax.set_title(ttl, fontsize=8.5, loc='left')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); fig.savefig(f'{FIG}/fig7_anytime.pdf'); plt.close(fig)
print('all done')

# ---------------- XAI figures ----------------
PRETTY = lambda s: (s.replace('sum_', 'total ').replace('last2_', 'last-2-wk ').replace('trend_', 'trend ').replace('_', ' ')
                    .replace('click pct in pres', 'click percentile (in-course)').replace('mean score', 'mean assessment score')
                    .replace('n submit', 'assessments submitted').replace('n late', 'late submissions'))
if os.path.exists('results/shap_global.csv'):
    sh = pd.read_csv('results/shap_global.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, task, c in zip(axes, ['success', 'withdrawal'], [PAL[0], PAL[1]]):
        g = sh[sh.task == task].groupby('feature').mean_abs_shap.agg(['mean', 'std']).sort_values('mean').tail(12)
        ax.barh([PRETTY(i) for i in g.index], g['mean'], xerr=g['std'], color=c, height=0.6, error_kw=dict(lw=0.7, ecolor='#555'))
        ax.set_title(f'{task.capitalize()}: mean |SHAP| (7 folds)', fontsize=8.5, loc='left'); ax.tick_params(axis='y', labelsize=6.8); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig8_shap.pdf'); plt.close(fig)
    sh.groupby(['task', 'feature']).mean_abs_shap.mean().groupby('task', group_keys=False).nlargest(10).to_csv('results/shap_top10.csv')
if os.path.exists('results/ig.csv'):
    ig = pd.read_csv('results/ig.csv'); at = pd.read_csv('results/attention.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, task in zip(axes[:2], ['success', 'withdrawal']):
        M = ig[ig.task == task].groupby(['channel', 'week_idx']).ig.mean().unstack()
        M = M.loc[M.sum(1).sort_values(ascending=False).index[:12]]
        M = M / M.values.max()
        ax.imshow(M.values, cmap='Blues', aspect='auto', vmin=0, vmax=1); ax.grid(False)
        ax.set_yticks(range(len(M)), [PRETTY(i) for i in M.index], fontsize=6.5)
        ax.set_xticks(range(M.shape[1]), ['pre'] + [str(i) for i in range(M.shape[1] - 1)], fontsize=6.5)
        ax.set_xlabel('course week'); ax.set_title(f'{task.capitalize()}: |IG| (normalised)', fontsize=8.5, loc='left')
    g = at.groupby('week_idx').attention.agg(['mean', 'std'])
    axes[2].bar(range(len(g)), g['mean'], yerr=g['std'], color=PAL[6], width=0.7, error_kw=dict(lw=0.7)); axes[2].grid(axis='x', visible=False)
    axes[2].axhline(1 / len(g), color='#777', ls=':', lw=0.9)
    axes[2].set_xticks(range(len(g)), ['pre'] + [str(i) for i in range(len(g) - 1)], fontsize=6.5)
    axes[2].set_title('Mean attention weight', fontsize=8.5, loc='left'); axes[2].set_xlabel('course week')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig9_ig_attention.pdf'); plt.close(fig)
if os.path.exists('results/fairness.csv'):
    fa = pd.read_csv('results/fairness.csv')
    s = fa.groupby(['attribute', 'group']).agg(n=('n', 'sum'), succ_auc=('succ_auc', 'mean'), wd_auc=('wd_auc', 'mean')).reset_index()
    s.to_csv('results/fairness_summary.csv', index=False)
    gap = s.groupby('attribute').agg(succ_gap=('succ_auc', lambda x: x.max() - x.min()), wd_gap=('wd_auc', lambda x: x.max() - x.min()))
    gap.to_csv('results/fairness_gap.csv')
    with open(f'{TAB}/tab_fairness.tex', 'w') as f:
        for a, g in s.groupby('attribute'):
            for i, (_, r) in enumerate(g.iterrows()):
                f.write(f"{a.replace('_', ' ') if i == 0 else ''} & {str(r.group).replace('%', '\\%').replace('<=', '$\\le$')} & {int(r.n):,} & {r.succ_auc:.3f} & {r.wd_auc:.3f}\\\\\n")
            f.write(f"\\cmidrule(lr){{2-5}} & \\textit{{max gap}} & & {gap.loc[a, 'succ_gap']:.3f} & {gap.loc[a, 'wd_gap']:.3f}\\\\\n\\midrule\n")
if os.path.exists('results/causal_dml.csv'):
    cz = pd.read_csv('results/causal_dml.csv')
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    y = np.arange(len(cz))[::-1]
    ax.errorbar(cz.ate, y, xerr=[cz.ate - cz.ci_low, cz.ci_high - cz.ate], fmt='o', color=PAL[0], ms=4, lw=1.2, label='DML ATE (95% CI)')
    ax.plot(cz.naive_diff, y, 'x', color=PAL[1], ms=5, label='naïve difference')
    ax.axvline(0, color='#777', lw=0.8); ax.set_yticks(y, [f"{s} (n={n:,})" for s, n in zip(cz.scope, cz.n)], fontsize=7)
    ax.set_xlabel('effect of early forum participation on P(pass)'); ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig10_causal.pdf'); plt.close(fig)
    with open(f'{TAB}/tab_causal.tex', 'w') as f:
        for _, r in cz.iterrows():
            f.write(f"{r.scope} & {int(r.n):,} & {r.treated_share:.2f} & {r.naive_diff:+.3f} & {r.ate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {r.p:.2g} & {r.get('rv', np.nan):.3f}\\\\\n")
print('xai figs done')

# ---------------- Bayesian signed-rank (Benavoli et al., 2017), ROPE = 0.01 AUC ----------------
def bayes_signed_rank(x, rope=0.01, n=20000, seed=0):
    rng = np.random.default_rng(seed); x = np.r_[0.0, np.asarray(x)]
    W = rng.dirichlet(np.ones(len(x)), n); s = x[:, None] + x[None, :]
    L = (s < -2 * rope).astype(float); R = (s > 2 * rope).astype(float); E = 1 - L - R
    pl = np.einsum('ni,ij,nj->n', W, L, W); pr = np.einsum('ni,ij,nj->n', W, R, W); pe = np.einsum('ni,ij,nj->n', W, E, W)
    win = np.argmax(np.c_[pl, pe, pr], 1); return (win == 0).mean(), (win == 1).mean(), (win == 2).mean()
brows = []
for K in ['25', '100', 'full']:
    p = res[res.k == K].groupby(['target', 'method'])[['succ_auc', 'wd_auc']].mean()
    for a, b in [('Hybrid-Head', 'LightGBM-P'), ('Hybrid-Head', 'LightGBM-T'), ('Hybrid-Head', 'Hybrid-FT-L2SP')]:
        for mt in ['succ_auc', 'wd_auc']:
            try: d = (p.xs(a, level='method')[mt] - p.xs(b, level='method')[mt]).dropna().values
            except KeyError: continue
            pl, pe, pr = bayes_signed_rank(d)
            brows.append(dict(k=K, a=a, b=b, metric=mt, mean_diff=d.mean(), p_b_better=pl, p_rope=pe, p_a_better=pr))
bs = pd.DataFrame(brows); bs.to_csv('results/bayes_signed_rank.csv', index=False)
with open(f'{TAB}/tab_bayes.tex', 'w') as f:
    for _, r in bs.iterrows():
        f.write(f"{'all' if r.k == 'full' else r.k} & {'Success' if r.metric == 'succ_auc' else 'Withdrawal'} & {r.a} vs {r.b} & {r.mean_diff:+.3f} & {r.p_a_better:.2f} & {r.p_rope:.2f} & {r.p_b_better:.2f}\\\\\n")

# ---------------- per-target table at k=100 ----------------
pt = res[res.k == '100'].groupby(['target', 'method'])[['succ_auc', 'wd_auc']].mean().unstack('method')
ms = [m for m in ['LR-T', 'XGBoost-T', 'LightGBM-T', 'LightGBM-P', 'Hybrid-T', 'Hybrid-Head', 'Hybrid-FT', 'Hybrid-FT-L2SP', 'Hybrid-LPFT', 'Ens-Head+LGBM-P'] if m in pt['succ_auc']]
zs0 = res[(res.k == '0')].groupby(['target', 'method'])[['succ_auc', 'wd_auc']].mean().unstack('method')
with open(f'{TAB}/tab_pertarget.tex', 'w') as f:
    f.write('\\begin{tabular}{l' + 'c' * (len(ms) + 2) + '}\n\\toprule\nTarget & ' + ' & '.join(['LGBM-S', 'Hyb-S'] + [m.replace('LightGBM', 'LGBM').replace('Hybrid', 'Hyb').replace('XGBoost', 'XGB') for m in ms]) + '\\\\\n\\midrule\n')
    for mt, nm in [('succ_auc', 'Success ROC-AUC'), ('wd_auc', 'Withdrawal ROC-AUC')]:
        f.write(f'\\multicolumn{{{len(ms) + 3}}}{{l}}{{\\textit{{{nm}}}}}\\\\\n')
        for t in pt.index:
            vals = [zs0[mt].loc[t, 'LightGBM-S'], zs0[mt].loc[t, 'Hybrid-S']] + [pt[mt].loc[t, m] for m in ms]
            best = np.nanmax(vals)
            f.write(t + ' & ' + ' & '.join((f'\\textbf{{{v:.3f}}}' if abs(v - best) < 5e-4 else f'{v:.3f}') for v in vals) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n')

# ---------------- learner-level bootstrap: Head vs LightGBM-P (seed 0) ----------------
try:
    import sys; sys.path.insert(0, 'code'); sys.path.insert(0, '.')
    from p1lib import load, snapshot, split_coldstart
    from sklearn.metrics import roc_auc_score
    ys, yw, m = YS, YW, MM
    pc = np.load('results/main_preds_classical.npz'); ph = np.load('results/main_preds_hybrid.npz')
    rng = np.random.default_rng(42); rows = []
    for tgt in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']:
        _, _, te = split_coldstart(m, tgt)
        for K in ['25', '100', 'full']:
            for task, y in [('s', ys[te]), ('w', yw[te])]:
                a = ph[f'{tgt}|{K}|Hybrid-Head|{task}']; b = pc[f'{tgt}|{K}|LightGBM-P|{task}']
                k = ~np.isnan(y); yy, aa, bb = y[k], a[k], b[k]; d = []
                for _ in range(1000):
                    i = rng.integers(0, len(yy), len(yy))
                    if len(np.unique(yy[i])) == 2: d.append(roc_auc_score(yy[i], aa[i]) - roc_auc_score(yy[i], bb[i]))
                rows.append(dict(target=tgt, k=K, task='success' if task == 's' else 'withdrawal', delta=roc_auc_score(yy, aa) - roc_auc_score(yy, bb), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    bt = pd.DataFrame(rows); bt.to_csv('results/bootstrap_head_vs_lgbmP.csv', index=False)
    with open(f'{TAB}/tab_bootstrap.tex', 'w') as f:
        for t, g in bt.groupby('target'):
            cells = []
            for K in ['25', '100', 'full']:
                for task in ['success', 'withdrawal']:
                    r = g[(g.k == K) & (g.task == task)].iloc[0]; sig = (r.lo > 0) or (r.hi < 0)
                    c = f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"; cells.append(f'\\textbf{{{c}}}' if sig else c)
            f.write(t + ' & ' + ' & '.join(cells) + '\\\\\n')
except Exception as e: print('bootstrap skipped:', e)
print('stats done')
