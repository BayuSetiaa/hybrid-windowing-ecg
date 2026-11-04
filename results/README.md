# Recorded results

These files record the runs behind the article. Per-run prediction files are not stored here because of their size; they are written to `runs/` when the experiments are repeated.

| Path | Contents |
|---|---|
| `label_audit.csv` | Label validity and edge padding by configuration and class (Table 7) |
| `first_environment/statistics/` | Statistics for Tables 4 and 6, from `scripts/analyze_reference_tables.py` |
| `first_environment/window_counts.csv` | Window counts by protocol and configuration |
| `first_environment/data_partition_records.csv` | Records and segments in each DS1/DS2 partition |
| `first_environment/segments_per_record.csv` | Segments per record and class, including paced records and class Q |
| `second_environment/statistics/` | Statistics for Tables 8 to 10b, from `scripts/analyze_results.py` |
| `second_environment/window_counts.csv` | Window counts for the additional experiments |
| `second_environment/env.json` | Software and hardware versions |
| `second_environment/determinism_check.json` | Comparison of two runs with the same seed |

## Experiment groups in the second environment

The `group` column of the recorded files uses the letters below. Repeating the experiments writes the group names instead.

| Letter | Group | Files |
|---|---|---|
| A | `reference` | `A_ablation_paired.csv`, `A_rerun_vs_first_environment.csv` |
| B | `rpeak` | `B_rpeak_paired.csv`, `B_rpeak_common_beats_*.csv` |
| C | `components` | `C_components_paired.csv`, `C_no_padding_with_cw_paired.csv` |
| D | `intra` | `D_intra_paired.csv` |
| E | `cross_validation` | `E_cv5_*.csv` |

## Experiment names

| Name | Meaning |
|---|---|
| `baseline` | Static windows, no class weights |
| `cw_only` | Static windows, class weights |
| `hw_only` | Hybrid windowing, no class weights |
| `hw_cw` | Hybrid windowing, class weights |
| `rp_*` | The same configuration with R-peak-centred windows |
| `no_adaptive` | A single stride for all minority classes |
| `no_phase` | No backward pass |
| `no_padding` | Padded windows removed |
| `no_energy` | Random selection within each group instead of energy pooling |
| `uw_only` | Majority reduction only |
| `ow_only` | Minority expansion only |
| `matched_random` | Static windows resampled at random to the hybrid class volumes |
| `grid_<pool>_g<size>` | Hybrid windowing with another pooling method and group size |
