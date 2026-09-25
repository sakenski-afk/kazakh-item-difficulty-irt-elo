# Item Difficulty and Learner Performance on a Kazakh–Russian Gamified Learning Platform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22961905.svg)](https://doi.org/10.5281/zenodo.22961905)

Code accompanying the paper

> **Text-Based Item Difficulty Estimation, IRT Calibration and Learner Performance Prediction on a Kazakh–Russian Gamified Learning Platform.**
> A. Kulatay, D. Zhaisanova, B. Kulambayev, and S. Mambetov. *International Journal of Advanced Computer Science and Applications (IJACSA)*, 2026 (submitted).

Data source: **Aqyl Battle**, a Kazakh–Russian mobile quiz-based learning game. The data were provided by Thousand IT Company LLP. One of the authors (A. Kulatay) is the CEO of Aqyl Battle (see the article's Conflict of Interest statement).

The framework has three parts, each built on a separate, internally consistent subset of the game's production database:

| Part | Question | Data | Models |
|---|---|---|---|
| 1 | How difficult is a question, judging from its text? | 30,444 items, 177.5 M attempts (Kazakh / Russian / English) | TF-IDF + Ridge, LightGBM, sentence embeddings (LaBSE, mE5-large, BGE-M3) + Ridge, fine-tuned mBERT, XLM-R base/large, KazRoBERTa |
| 2 | What is the psychometric difficulty of tournament items, and do text predictions transfer to it? | 67,990 responses, 680 learners, 416 items | 1PL / 2PL / 3PL IRT (marginal MAP, Gauss–Hermite quadrature); cold-start validation on 327 new items (347 with text, 20 duplicates of item-bank content excluded); calibration experiment |
| 3 | How proficient are learners, overall and by subject? | 100,000 duels, 322,046 per-question responses | Elo, Glicko-2, TrueSkill, logistic regression, LightGBM; learner-by-category Elo |

Headline results (test sets; see `results/reference/tables/summary.json`):

- Part 1: selected model XLM-R large, category + text input, seed ensemble: r = 0.527, against r = 0.461 for TF-IDF + metadata (Δr = +0.066, 95% CI 0.032–0.099, bootstrap over item groups).
- Part 2: 3PL gives the best held-out AUC (0.834; the ordering also holds for held-out learners); AIC prefers 2PL and BIC prefers 1PL, so the choice is tentative. On 327 tournament items without duplicates in the item bank, text predictions correlate moderately with 2PL difficulty (r = 0.24–0.33 for the main models); KazRoBERTa has the highest observed value (r = 0.331), but differences between models are not significant. In a calibration simulation, a text-based prior helps before any responses are observed, but not beyond a population prior once five or more responses are available.
- Part 3: duel outcome AUC is 0.828 with Elo and 0.830 with LightGBM (history + Elo; difference not significant). Response correctness AUC is 0.697 with learner × category Elo, against 0.684 without the category term (ΔAUC 95% CI 0.011–0.017).

## Repository layout

```
run_all.py                     pipeline driver (stages in dependency order)
src/
  audit_export.py              catalogue of the export (rows, columns, PII exclusions, time windows)
  prepare_items.py             Part 1 dataset: items with >= 30 attempts, logit difficulty, leakage-safe group split
  part2_irt.py                 Part 2: 1PL/2PL/3PL IRT, held-out prediction, AIC/BIC, item and person parameters
  prepare_tournament_items.py  cold-start item set (tournament items with text + IRT parameters)
  part1_baselines.py           Part 1: mean, category mean, TF-IDF + Ridge, LightGBM, TF-IDF + metadata
  part1_options_ablation.py    Part 1: answer-option ablation, five inputs (group 5-fold CV)
  part1_embeddings.py          Part 1: frozen sentence embeddings + Ridge
  part1_finetune.py            Part 1: fine-tune one encoder (one model / input / seed)
  run_finetune_grid.py         Part 1: full grid (4 encoders x 2 inputs x 3 seeds), divergence handling
  part3_elo.py                 Part 3: Elo / Glicko-2 / TrueSkill / LR / LightGBM; learner-vs-category Elo
  analysis.py                  aggregation: model tables, bootstrap CIs, language breakdown,
                               cross-lingual difficulty comparison, cold-start validation, Elo-category linkage
  fig_paper.py, fig_framework.py   figures of the paper
  compare_to_reference.py      checks a re-run against the published numbers
  export_open_data.py          builds the open data package (item statistics without texts or learner data)
  uncertainty_analysis.py      duplicate check of tournament items, group / learner-clustered bootstrap,
                               model comparisons, Wald-test adjustments, platform language labels
  irt_robustness.py            learner hold-out, ML refits, discrimination SEs, 3PL prior sensitivity,
                               item fit and Q3, calibration experiment with text-based priors
  language_check.py            accuracy of the language heuristic against a manual annotation
  figures_extended.py          Fig. 3 (items without duplicates) and Fig. 4 (calibration experiment)
  metrics.py                   shared regression metrics
data/README.md                 required input files (data are not distributed)
docs/data_dictionary.md        tables and columns used
results/reference/             tables and figures reported in the paper (extended/: robustness, uncertainty, calibration)
```

## Installation

Tested with Python 3.14.2 on Windows 10 with an NVIDIA RTX 5070 Ti Laptop GPU (12 GB), CUDA 12.8.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
```

> On Windows, create the virtual environment on a short path (e.g. `C:\venvs\ijacsa`). Deep paths can exceed the Windows limit and break loading of scikit-learn's DLLs (`WinError 206`).

## Data

The data are an anonymised export of the Aqyl Battle database, owned by the platform operator. Question texts and learner-level records are not redistributed; item-level statistics without texts are available as open data. See [`data/README.md`](data/README.md) for the expected files and how to request access. All scripts read from `data/raw/` and write to `data/` and `results/`.

## Open data

Item-level statistics without texts or learner data are published separately on Zenodo: [doi:10.5281/zenodo.22960055](https://doi.org/10.5281/zenodo.22960055). They are built with `python src/export_open_data.py --out <dir> --private_dir <dir>`. Platform ids are replaced by random ids, and the id mapping is written to `--private_dir` and must not be published. The package contains `verify_open_data.py`, which recomputes the key results of the paper from the open files alone.

## Reproducing the results

```bash
python run_all.py                    # full pipeline
python run_all.py --skip-finetune    # without the GPU fine-tuning grid
python run_all.py --from part3       # resume from a stage
```

Stage order: `audit → prepare → part2 → tournament → baselines → ablation → embeddings → finetune → part3 → analysis → uncertainty → irt_robustness → language → figures → figures_extended → framework → compare`. The language check needs a manual annotation that is not distributed and is skipped without it. Part 2 runs before the Part 1 baselines because the cold-start item set needs the IRT parameters.

Approximate run times on the reference machine:

| Stage | Time |
|---|---|
| All CPU stages (audit, IRT, baselines, option ablation, rating models, analysis) | ~3 min |
| Embeddings | ~6 min (GPU) |
| Fine-tuning grid | ~3 h (GPU); XLM-R large needs gradient checkpointing on 12 GB |

**Checking a re-run.** `python src/compare_to_reference.py` (also the last stage of `run_all.py`) compares key numbers with the published ones in `results/reference/tables/`. On the reference machine all checks match exactly.

**Model selection.** The configuration reported as the main Part 1 model is selected by mean validation RMSE, never by test metrics.

**Replaced fine-tuning run.** Fine-tuning of large encoders is occasionally unstable. In the reported experiments one of six XLM-R large runs (text input, seed 3) collapsed to near-constant predictions (validation RMSE 0.924, close to the 0.926 of a constant predictor; validation r = 0.03). No divergence criterion had been specified in advance: the collapse was noticed after training and the run was replaced by seed 4 as a post hoc decision. The article reports results with and without the replacement. `run_finetune_grid.py` now implements an explicit rule (best validation Pearson r below 0.2), formalised after the experiments; diverged runs are moved to `results/finetune_diverged/`.

**Determinism.** IRT, the baselines and the rating models are deterministic. Fine-tuning uses fixed seeds, but GPU kernels are not bit-exact, so re-runs can differ in the third decimal.

## Citation

If you use this code, please cite the paper and the software (see [`CITATION.cff`](CITATION.cff)). Software (release v1.0.0): [doi:10.5281/zenodo.22961905](https://doi.org/10.5281/zenodo.22961905). Open data: [doi:10.5281/zenodo.22960055](https://doi.org/10.5281/zenodo.22960055).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Reported results and figures in `results/reference/`: CC BY 4.0. Pretrained models are subject to their own licences on Hugging Face.
