"""Build sequence + static tensors from raw OULAD (Kuzilek et al., 2017).
Output: oulad_tensors.npz + oulad_meta.pkl. Weekly grid: idx0 = pre-start (date<0), idx w = course week w-1."""
import pandas as pd, numpy as np, sys, os, time
RAW = sys.argv[1] if len(sys.argv) > 1 else 'oulad'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
W = 27  # pre-start + weeks 0..25 (day 181)
t0 = time.time()
info = pd.read_csv(f'{RAW}/studentInfo.csv', na_values=['?'])
reg = pd.read_csv(f'{RAW}/studentRegistration.csv', na_values=['?'])
crs = pd.read_csv(f'{RAW}/courses.csv', na_values=['?'])
ass = pd.read_csv(f'{RAW}/assessments.csv', na_values=['?'])
sa = pd.read_csv(f'{RAW}/studentAssessment.csv', na_values=['?'])
vle = pd.read_csv(f'{RAW}/vle.csv', na_values=['?'])
key = ['code_module', 'code_presentation', 'id_student']
meta = info.merge(reg, on=key, how='left').merge(crs, on=['code_module', 'code_presentation'], how='left')
meta = meta.reset_index(drop=True); meta['eid'] = np.arange(len(meta))
eid = meta.set_index(key)['eid']
acts = sorted(vle.activity_type.unique()); A = len(acts); amap = {a: i for i, a in enumerate(acts)}
# --- VLE clicks (chunked) ---
seq = np.zeros((len(meta), W, A + 3), dtype=np.float32)
vle_map = vle.set_index('id_site')['activity_type'].map(amap)
for ch in pd.read_csv(f'{RAW}/studentVle.csv', chunksize=2_000_000,
                      dtype={'code_module': 'category', 'code_presentation': 'category', 'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int32'}):
    ch = ch[ch.date < (W - 1) * 7]
    ch['a'] = ch.id_site.map(vle_map).values
    ch['w'] = np.where(ch.date < 0, 0, ch.date // 7 + 1)
    g = ch.groupby(['code_module', 'code_presentation', 'id_student', 'w', 'a'], observed=True).sum_click.sum().reset_index()
    e = eid.reindex(pd.MultiIndex.from_frame(g[key].astype({'code_module': str, 'code_presentation': str}))).values
    ok = ~np.isnan(e)
    np.add.at(seq, (e[ok].astype(int), g.w.values[ok].astype(int), g.a.values[ok].astype(int)), g.sum_click.values[ok])
    print('chunk', round(time.time() - t0), flush=True)
seq[:, :, :A] = np.log1p(seq[:, :, :A])
# --- assessment channels: n_submitted, mean score/100, n_late (non-banked, by date_submitted) ---
s = sa.merge(ass, on='id_assessment')
s = s[(s.is_banked == 0) & (s.assessment_type != 'Exam')]
s['late'] = (s.date_submitted > s.date).astype(float)
s['w'] = np.where(s.date_submitted < 0, 0, s.date_submitted // 7 + 1)
s = s[s.w < W]
e = eid.reindex(pd.MultiIndex.from_frame(s[key])).values; ok = ~np.isnan(e)
ei, wi = e[ok].astype(int), s.w.values[ok].astype(int)
np.add.at(seq, (ei, wi, A), 1.0)
np.add.at(seq, (ei, wi, A + 1), (s.score.fillna(0).values[ok] / 100.0))
np.add.at(seq, (ei, wi, A + 2), s.late.values[ok])
n = np.maximum(seq[:, :, A], 1); seq[:, :, A + 1] /= n
# --- static features (domain-agnostic: no module identity) ---
cat = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
num = ['num_of_prev_attempts', 'studied_credits', 'date_registration', 'module_presentation_length']
st = pd.get_dummies(meta[cat].fillna('NA'), dtype=np.float32)
nm = meta[num].astype(float)
nm['reg_missing'] = nm.date_registration.isna().astype(float)
nm = nm.fillna(nm.median())
nm['studied_credits'] = np.log1p(nm.studied_credits)
static = pd.concat([nm, st], axis=1)
np.savez_compressed(f'{OUT}/oulad_tensors.npz', seq=seq, static=static.values.astype(np.float32),
                    seq_names=np.array(acts + ['n_submit', 'mean_score', 'n_late']), static_names=np.array(static.columns))
meta[key + ['eid', 'final_result', 'date_unregistration', 'date_registration', 'module_presentation_length']].to_pickle(f'{OUT}/oulad_meta.pkl')
print('done', seq.shape, static.shape, round(time.time() - t0), 's')
