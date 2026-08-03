# Archival Selection and Prompt-Relative Associations in Vision-Language Models for Museum Collections

[![Paper Target](https://img.shields.io/badge/Journal-AI%20%26%20Society%20(Springer)-blue)](https://www.springer.com/journal/146)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Repository for the museum-archive audit accompanying the manuscript **"Archival Selection and Prompt-Relative Associations in Vision-Language Models for Museum Collections."** The manuscript targets *AI & Society: Journal of Knowledge, Culture and Communication*.

## Overview

This project studies prompt-relative image-text associations in two vision-language models:

- OpenAI CLIP, ViT-B/32
- OpenCLIP, ViT-B/32, LAION-2B

The audit uses artwork metadata and image assets from the Metropolitan Museum of Art Open Access API. It compares three prompt contrasts involving masterpiece, quality, and influence language. These prompts are treated as separate operational probes, not as validated measures of artistic quality or museum value.

The analysis reports Mann-Whitney tests, rank-biserial effects, bootstrap confidence intervals, TOST mean-equivalence checks, and covariate-adjusted OLS associations with HC3 standard errors. Covariates include physical medium, creation century, and image aspect ratio.

## Dataset accounting

```text
Raw API harvest:                         N = 1,500
Unknown or unattributed creators:        n = 618  (41.20%)
Named-attribution records:               n = 882  (58.80%)
Named records excluded for image issues: n = 139
Final image-complete cohort:             N = 743
  Male-inferred records:                 n = 534
  Female-inferred records:               n = 209
```

"Male-inferred" and "female-inferred" describe a name-based analytic category. They do not measure personal gender identity. Unknown, unidentified, unattributed, and anonymous are retained as distinct metadata concepts; aggregate reporting uses "unknown or unattributed" where appropriate.

## Pipeline

```text
1. fetch_met_data.py          -> Harvest Met Open Access metadata
2. analyze_representation.py  -> Classify creator category, century, and medium
3. clip_audit.py              -> Run OpenAI CLIP image-text scoring and publication diagnostics
4. openclip_audit.py          -> Run OpenCLIP image-text scoring and publication diagnostics
5. additional_model_audit.py  -> Run scale/objective sensitivity models
```

## Installation

```bash
git clone https://github.com/manpreet28111995/disentangling-archival-bias.git
cd disentangling-archival-bias

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To continue an interrupted run, add `--resume`; completed stages with intact output files are skipped.

## Execution

### Complete four-model run from scratch

```bash
python run_experiment.py --out-dir results_full_run --harvest-only
```

On Apple Silicon, code automatically uses `mps`; otherwise it falls back to
CUDA or CPU. Unsupported MPS operations use CPU fallback.

This single command fetches real Metropolitan Museum API records, enriches
metadata, runs OpenAI CLIP, OpenCLIP ViT-B/32, OpenCLIP ViT-L/14, and SigLIP,
then writes scores, object IDs, image manifests, statistical tables, sensitivity
analyses, and publication reports under `results_full_run/`. OpenAI CLIP image
validation defines one locked objectID cohort reused by all four models.

Run the complete experiment from an existing real merged dataset:

```bash
python3 run_experiment.py \
  --metadata met_metadata_merged.csv \
  --out-dir results_merged_full
```
Harvest uses Met departments 1–25, department-wide `/objects` listing plus keyword fallbacks. Run
reports exact real primary-image count; no synthetic records used.
Each completed run writes `appendix.md` with runtime query details, software and
device versions, model IDs, prompts, counts, object-ID manifests, and output paths.

### 1. Harvest metadata

```bash
python fetch_met_data.py --departments 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 --out met_metadata.csv
```

### 2. Enrich metadata

```bash
python analyze_representation.py --in met_metadata.csv --out-dir results
```

### 3. Run OpenAI CLIP audit

```bash
python clip_audit.py --in results/met_metadata_enriched.csv --out-dir results
```

### 4. Run OpenCLIP audit

```bash
python openclip_audit.py --in results/met_metadata_enriched.csv --out-dir results

# Additional model: scale sensitivity
python additional_model_audit.py --in results/met_metadata_enriched.csv --out-dir results \
  --model-name laion/CLIP-ViT-L-14-laion2B-s32B-b82K --label OpenCLIP_L14

# Additional model: different contrastive objective
python additional_model_audit.py --in results/met_metadata_enriched.csv --out-dir results \
  --model-name google/siglip-base-patch16-224 --label SigLIP
```

Both audits use the full real-image cohort by default. Optional exploratory
subsampling is available only for OpenAI CLIP with `--n-female` and `--n-male`.
Each audit also writes prompt-level comparisons with Bonferroni-adjusted p-values,
TOST results, evaluation counts, and regression sensitivity tables.
Additional models use the exact same real object cohort and prompts. SigLIP scores
use logit differences rather than CLIP softmax probabilities because SigLIP uses
independent sigmoid-style image-text scores; model results must therefore be
interpreted as model-specific prompt contrasts.
All records come from the Metropolitan Museum Open Access API. The pipeline does
not generate synthetic objects. Object IDs and real image URLs are preserved in
`object_selection_manifest.csv` and model-specific image manifests for reproduction.

## Repository structure

```text
.
├── fetch_met_data.py
├── analyze_representation.py
├── clip_audit.py
├── openclip_audit.py
├── met_metadata.csv
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

`Paper/` contains local manuscript and generated LaTeX artifacts. It is ignored by `.gitignore` and is not part of the tracked code repository.

## Results

Main result files are stored in `results/`:

- `clip_academic_stats_report.txt`
- `openclip_academic_stats_report.txt`
- `clip_scores.csv`
- `openclip_scores.csv`
- `openai_clip_prompt_comparisons.csv`
- `openclip_prompt_comparisons.csv`
- `openai_clip_regression_sensitivity.csv`
- `openclip_regression_sensitivity.csv`
- `openai_clip_publication_report.txt`
- `openclip_publication_report.txt`
- `gender_representation.csv`
- `gender_by_century.csv`
- `met_metadata_enriched.csv`
- `object_selection_manifest.csv`
- `openai_clip_image_manifest.csv`
- `openclip_image_manifest.csv`
- `openclip_l14_image_manifest.csv`
- `siglip_image_manifest.csv`

The reported associations apply only to tested score instruments and this museum collection. Gender analyses use named-attribution records; visual audits include unknown/unattributed works. They do not establish causal model effects or general fairness.

## Reproducibility

Code, analysis outputs, and reproducibility materials are available at the anonymous repository:

[https://anonymous.4open.science/r/disentangling-archival-bias-138F/](https://anonymous.4open.science/r/disentangling-archival-bias-138F/)

## Citation

If you use this code or results, cite:

```bibtex
@article{archival_selection_prompt_associations_2026,
  title     = {Archival Selection and Prompt-Relative Associations in Vision-Language Models for Museum Collections},
  journal   = {AI \\& Society: Journal of Knowledge, Culture and Communication},
  publisher = {Springer Nature},
  year      = {2026}
}
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
