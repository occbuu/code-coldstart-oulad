"""Generate the exploratory behaviour profiles in Online Resource 1."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from p1lib import load, snapshot

seq, static, meta, *_ = load(os.environ.get('DATA', 'code'))
X, _, _, _, m = snapshot(seq, static, meta, 70)
Path('paper/figures').mkdir(parents=True, exist_ok=True)
tot = np.expm1(X[:, :, :20]).sum(2)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for lab, c in [('Distinction', '#2a78d6'), ('Pass', '#1baf7a'), ('Fail', '#eda100'), ('Withdrawn', '#e34948')]:
    k = (m.final_result == lab).values
    axes[0].plot(np.median(tot[k], 0), label=lab, color=c); axes[1].plot((tot[k] > 0).mean(0), label=lab, color=c)
    axes[2].plot(np.nanmean(np.where(X[k, :, 20] > 0, X[k, :, 21], np.nan), 0), label=lab, color=c)
for a, t in zip(axes, ['median weekly clicks', 'share active in week', 'mean assessment score (submitters)']): a.set_title(t, loc='left'); a.set_xlabel('week bin (0 = pre-start)')
axes[0].legend(); plt.tight_layout(); plt.savefig('paper/figures/figS1_eda.pdf'); plt.close(fig)