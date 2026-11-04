# Hybrid Windowing for ECG Class Balancing

Reference implementation for the article *Adaptive stride and energy pooling hybrid windowing for non-synthetic class balancing of electrocardiogram signals: intra-patient versus patient-independent evaluation* (under review at *Array*).

Hybrid windowing changes ECG class volumes without a generative model. It reduces the majority class through adaptive and energy pooling, and it expands the minority classes through overlapping windows whose stride scales with class size. This repository contains the method, the experiments, the statistical analysis, and the recorded results.

## Main finding

The procedure reduced the largest class ratio in the 44 analysed MIT-BIH records from 118.8:1 to 9.4:1. This did not translate into better classification of unseen patients.

| Evaluation | Baseline | Class weights | Hybrid | Hybrid + class weights |
|---|---|---|---|---|
| Intra-patient split (same patients) | n/a | 0.730 | n/a | 0.966 |
| DS1/DS2 split | 0.433 | 0.400 | 0.388 | 0.357 |
| DS1/DS2 split, R-peak-centred windows | 0.425 | 0.423 | 0.426 | 0.453 |
| Patient-level five-fold cross-validation | 0.398 | 0.358 | 0.333 | 0.335 |

Values are mean macro-F1 from the second computing environment (three seeds for the intra-patient split, five seeds for DS1/DS2, and five folds for cross-validation). Intra-patient scores are optimistic because correlated windows from the same recordings appear in both the training and test subsets. No macro-F1 difference between configurations on unseen patients remained significant after Holm correction in this environment. The article reports the full analysis, including the results of the first environment. All results apply only to the 44 records studied.

## Method

Each annotated beat defines a variable-length segment that extends from the preceding annotation to the following annotation and inherits the AAMI class of the beat.

- **Static baseline:** one window per segment, taken from the first 360 samples. Shorter segments yield no window.
- **Majority class:** each segment yields one window chosen by similarity to a pooled reference (adaptive pooling), and each group of windows then yields the window closest to the mean group energy (energy pooling).
- **Minority classes:** forward and half-stride backward passes, with a stride proportional to the number of segments in the class (adaptive stride).

### Data handling

- The source is the MLII channel of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). Paced records 102, 104, 107, and 217 are excluded.
- Signals are used in millivolts as provided by PhysioNet. No bandpass filter or amplitude normalization is applied.
- Windows are slices of the original recordings, with one exception. A segment shorter than the window, or a backward window that passes the end of a segment, is completed by repeating the nearest sample. The `no_padding` variant removes these windows.
- A window inherits the label of its segment, so it may omit the labelled beat or contain beats from another class. `scripts/audit_labels.py` quantifies both effects.

## Repository structure

```text
src/hybrid_windowing/
    config.py        parameters, record partitions, and expected window counts
    dataset.py       segment construction from the MIT-BIH files
    windowing.py     static and hybrid windowing, including ablation variants
    rpeak.py         R-peak-centred windows
    model.py         one-dimensional CNN classifier
    training.py      deterministic training
    metrics.py       classification metrics
    experiments.py   experiment groups
    statistics.py    statistical helpers
scripts/             data preparation, experiments, analysis, and label audit
tests/               unit tests and checks against the reported window counts
results/             recorded per-run metrics and statistical tables
```

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/BayuSetiaa/hybrid-windowing-ecg.git
cd hybrid-windowing-ecg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the tests with `pytest`. The checks against the reported window counts are skipped until the dataset has been prepared.

## Usage

```bash
python scripts/prepare_data.py
```

This command downloads the 44 records from PhysioNet into `data/mitdb` and writes `data/mitbih_segments.npz`.

```python
from hybrid_windowing.config import TRAIN_RECORDS
from hybrid_windowing.dataset import load_dataset
from hybrid_windowing.windowing import build_hybrid

dataset = load_dataset("data/mitbih_segments.npz")
windows, labels = build_hybrid(dataset.subset(TRAIN_RECORDS))
print(windows.shape)  # (24548, 360, 1)
```

## Reproducing the experiments

```bash
python scripts/run_experiments.py --output outputs
python scripts/analyze_reference_tables.py --results-dir outputs --output outputs/reference_statistics
python scripts/analyze_results.py --results-dir outputs --output outputs/statistics \
    --reference-table results/first_environment/statistics/table_for_manuscript.csv
python scripts/audit_labels.py --output outputs/label_audit.csv
```

Add `--smoke` to `run_experiments.py` for a short check of the pipeline; its scores are not meaningful. Use `--groups` to run part of the experiments, and note that completed runs are skipped when a command is repeated. The 115 runs of the second environment took about 4 hours of training time on an NVIDIA A100.

| Article | Experiment group | Output |
|---|---|---|
| Tables 1 and 2 | `prepare_data.py` | Console output, `results/first_environment/segments_per_record.csv` |
| Table 3 | `src/hybrid_windowing/model.py` | Model definition |
| Tables 4 and 4b | `intra`, `grid` | `table4_corrected.csv`, `results_all_runs.csv` |
| Table 5 | `intra` | `window_counts.csv` |
| Table 6 | `reference`, `intra` | `table_for_manuscript.csv`, `paired_tests.csv`, `mcnemar_per_seed.csv` |
| Table 7 | `audit_labels.py` | `label_audit.csv` |
| Table 8 | `reference`, `rpeak`, `intra` | `summary_mean_sd.csv`, `A_ablation_paired.csv`, `B_rpeak_paired.csv`, `D_intra_paired.csv` |
| Table 9 | `components` | `summary_mean_sd.csv`, `C_components_paired.csv` |
| Tables 10 and 10b | `cross_validation` | `E_cv5_paired.csv`, `E_cv5_pooled.csv`, `E_cv5_macro_f1_per_fold.csv` |

## Computing environments

The reported scores come from two environments. Scores changed between them, including for identical configurations and seeds, although a same-seed repeat within the second environment reproduced all 47,806 test predictions.

| | First environment | Second environment |
|---|---|---|
| Article tables | 4 to 6 | 8 to 10b |
| Platform | JupyterHub | Google Colab |
| GPU | NVIDIA A100-SXM4, MIG 3g.20gb | NVIDIA A100-SXM4, 40 GB |
| Driver | 450.80.02 | 580.82.07 |
| Software | CUDA 11.0 | Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.4.6 |
| XLA compilation | Default | Disabled |

This repository uses the settings of the second environment, so re-running the groups behind Tables 4 to 6 will not reproduce their exact values. The values recorded for both environments are in `results/`.

This release restructures the scripts used for the reported runs. It rebuilds the same segments and windows as those scripts and gave identical predictions in 25 short test runs (one epoch on CPU, TensorFlow 2.18). The full set of runs has not been repeated with this release, so the values in `results/` come from the original scripts.

## Limitations

- The evaluation is limited to 44 records from one database, and no external validation was performed.
- Hybrid windowing is not presented as a way to improve classification. These data provide no evidence of improvement on unseen patients.
- Class F is concentrated in few records, which limits the diversity available for learning this class.
- R-peak-centred windows rely on reference annotations, so their scores do not represent a deployable system.

## Citation

The article is under review. Until it is published, please cite this software using `CITATION.cff`.

The MIT-BIH Arrhythmia Database is not redistributed here. Please cite its sources when using it:

- Moody, G. B., and Mark, R. G. The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50, 2001. https://doi.org/10.1109/51.932724
- Goldberger, A. L., et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220, 2000. https://doi.org/10.1161/01.CIR.101.23.e215

The DS1/DS2 partition follows de Chazal, P., O'Dwyer, M., and Reilly, R. B., *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206, 2004. https://doi.org/10.1109/TBME.2004.827359

## Acknowledgement of AI assistance

The code was written with the assistance of Claude (Anthropic) and checked by the authors, as declared in the article.

## License

The code is released under the [MIT License](LICENSE).

## Contact

Corresponding author: Umar Zaky, Universitas Teknologi Yogyakarta, umarzaky@uty.ac.id
