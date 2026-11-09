# Cold-start transfer learning on OULAD

Reproduction code for *Benchmarking transfer and adaptation for cold-start multi-task learner-outcome prediction on OULAD* by Hieu Le Ngoc, Nguyen Huynh Quang Vuong, Nguyen Quoc Viet, and Bui Dinh Nguyen.

The benchmark predicts academic success and withdrawal at day 70. It trains on earlier presentations of other modules, adapts with a limited number of labelled enrolments from the target module, and evaluates on its 2014J presentation. Learners in the target test cohort are excluded from that fold's training data. The experiments cover source-only, target-only and pooled baselines, multi-task transfer, online adaptation, observation cutoffs, attribution, subgroup evaluation and an exploratory forum-participation analysis.

## Repository layout

| Path | Purpose |
| --- | --- |
| run_pipeline.py | Runs the reproduction workflow from an OULAD directory or ZIP. |
| code/build_features.py | Builds weekly sequences and static profiles from the seven source CSVs. |
| code/p1lib.py | Implements cohort construction, model training and evaluation. |
| code/run_exp.py | Runs the benchmark, ablations, streaming and observation-cutoff experiments. |
| code/shift.py and code/eda.py | Produce the domain-shift results and exploratory behaviour figure. |
| code/xai.py | Runs TreeSHAP, integrated gradients, subgroup analysis and double machine learning. |
| code/analyze.py | Builds the ensemble, statistical summaries, PDF plots and LaTeX table fragments. |
| reference-results/ | Aggregate CSV outputs archived with the paper. |
| diagrams/ | TikZ sources for the workflow and model architecture figures. |
| P1_ColdStart_Transfer_OULAD.ipynb | Optional notebook for inspecting archived results or invoking the same pipeline. |

## Setup

Use Python 3.13, the Python series recorded in the paper notebook. Run the commands below from the repository root.

Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch uses CUDA when available and otherwise uses CPU. For notebook use, install JupyterLab with `python -m pip install jupyterlab` and open the notebook from the repository root. The archived run recorded Python 3.13.18, PyTorch 2.15.0+cu132 (CPU execution), LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1 and SHAP 0.54.1. The requirements file lists installation dependencies. Exact scores can vary with library versions, hardware and fresh checkpoint initialization.

## Data

Download the [Open University Learning Analytics Dataset from UCI](https://doi.org/10.24432/C5KK69). Use the archive directly or extract assessments.csv, courses.csv, studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv and vle.csv into data_oulad/. The raw data and generated tensors are excluded from the repository.

## Reproduce the results

Run the complete workflow using the archive or extracted directory.

```bash
python run_pipeline.py --raw "path/to/oulad.zip"
# Or use an extracted directory
python run_pipeline.py --raw data_oulad
```

To run in stages, use the following commands in order.

```bash
python run_pipeline.py --stage prepare --raw "path/to/oulad.zip"
python run_pipeline.py --stage benchmark
python run_pipeline.py --stage explain --raw data_oulad
python run_pipeline.py --stage report
```

If you prepared the dataset from an extracted directory outside the repository, pass that same directory to the explain stage. The prepare stage writes the tensors, domain-shift table data and exploratory behaviour plot. The benchmark stage runs the seven-fold experiments and writes checkpoints to models/. The explain stage uses the day-70 source checkpoints from that benchmark. The report stage uses the generated CSVs and prediction archives in results/ and writes PDF plots to paper/figures/ and LaTeX fragments to paper/tables/.

Existing checkpoints are reused. For a fresh reproduction, start from a clean checkout with an empty models/ directory. Keep outputs from different runs separate. This workflow retrains multiple models across modules, label budgets and observation cutoffs, so allow time for CPU execution.

To build the two TikZ diagrams, run `pdflatex` on diagrams/fig1_workflow.tex and diagrams/fig2_architecture.tex with a TeX installation that includes standalone and TikZ.

## Read the outputs

- main_all.csv combines the model runs and the post-hoc ensemble. The ensemble uses seed 0.
- adapt_vs_zeroshot.csv compares each adaptation strategy against the source model at matched modules.
- wilcoxon.csv and bootstrap_head_vs_lgbmP.csv report paired-run comparisons and per-module learner bootstrap intervals. Repeated seeds share a test cohort, so seed-level tests are exploratory.
- anytime_with70.csv combines the cutoff runs with the day-70 benchmark results.
- fairness_summary.csv summarises the reported subgroup slices. Small per-module groups and missing categories are excluded.
- causal_dml.csv reports the day-28 forum-participation analysis. Its adjusted contrast is exploratory because some controls are measured during the treatment window.

## Citation

If you use this code, cite the companion manuscript and the OULAD data descriptor. The manuscript citation below is provisional.

Ngoc, H. L., Vuong, N. H. Q., Viet, N. Q., & Nguyen, B. D. *Benchmarking transfer and adaptation for cold-start multi-task learner-outcome prediction on OULAD* [Manuscript].

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, Article 170171. [https://doi.org/10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171)
