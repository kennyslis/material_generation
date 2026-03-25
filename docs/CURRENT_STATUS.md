# Current Data Status

## Current local state

As of this workspace snapshot:

- `dataset/raw/2dmatpedia.json` is present.
- `dataset/processed/real2d_train_materials.json`, `real2d_val_materials.json`, and `real2d_test_materials.json` have been generated.
- Therefore, the current figures and metrics under `results/` now correspond to the real public 2DMatPedia-backed run.

## Confirmed indicators of real-data mode

- processed cache prefix: `real2d_`
- metadata source: `2dmatpedia`
- raw public dataset file detected in `dataset/raw/`

## Current real-data metrics

- Baseline avg `|ΔG_H|`: `0.5176 eV`
- Ours avg `|ΔG_H|`: `0.0689 eV`
- Baseline stability: `0.8024`
- Ours stability: `0.8425`
- Baseline synthesis success rate: `0.20`
- Ours synthesis success rate: `1.00`

## Submission guidance

This workspace is now in a valid state for a “real public dataset connected” submission.

Before pushing GitHub, make sure the following updated artifacts are included:

- `results/loss_curve.png`
- `results/her_performance.png`
- `results/stability_curve.png`
- `results/generated_structures.png`
- `results/comparison_metrics.json`
- `results/top_candidates.csv`
- `results/top_candidates.md`
- `README.md`
