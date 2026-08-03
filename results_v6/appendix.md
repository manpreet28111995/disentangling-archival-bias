# Experiment Runtime Appendix

Generated (UTC): 2026-08-03T12:11:32.532555+00:00
Git commit: `e150f4024786b753317e1acfb275a9c8d79d09b0`
Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
Python: `3.14.6`
PyTorch: `2.13.0`
Runtime device: `mps`

## Exact command

```bash
python3 run_experiment.py --out-dir results_v6 --harvest-only
```

## Metropolitan Museum API query

- Base: `https://collectionapi.metmuseum.org/public/collection/v1`
- Departments: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25`
- Candidate: `/objects?departmentIds=1|6|8|9|11|12|15|19|21`
- Fallback: `/search?departmentId=<id>&q=<term>&hasImages=true`
- Terms: `painting, drawing, print, photograph, sculpture, woman, female, portrait, art, artist`
- Object endpoint: `/objects/<objectID>`; image fields: `primaryImage`, `primaryImageSmall`.
- Retention rule: non-empty real `primaryImage` required.

## Dataset accounting

- API metadata rows saved: **445**
- Locked real-image cohort: **445**
- No minimum image threshold; exact real-image count is reported below.
- `objectID` is primary reproduction key.

## Models and scoring

| Label | Model | Score |
|---|---|---|
| OpenAI_CLIP | `openai/clip-vit-base-patch32` | softmax probability difference |
| OpenCLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | softmax probability difference |
| OpenCLIP_L14 | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | softmax probability difference |
| SigLIP | `google/siglip-base-patch16-224` | logit difference |

- Masterpiece: `an important masterpiece of fine art` vs `a minor, forgettable work of art`
- Quality: `a museum-quality masterwork` vs `an amateur painting`
- Influence: `a groundbreaking and influential artwork` vs `a decorative craft object`

## Reproduction artifacts

- Raw metadata: `/Users/manpreet/Documents/GitHub/disentangling-archival-bias/met_metadata_merged.csv`
- Locked cohort: `results_v6/met_metadata_evaluation_cohort.csv`
- Model artifacts: `results_v6/models/`
- Object IDs: `object_selection_manifest.csv` and model-specific `*_image_manifest.csv`.
- Novelty outputs: `models/novelty/object_risk_ranking.csv`, `department_consensus.csv`, and `consensus_summary.csv`.
- Reports: model-specific `*_publication_report.txt`.

## Interpretation boundary

Visual-model results use all real-image records; gender summaries use named-attribution subset. Results do not establish causal effects or population-wide fairness.
