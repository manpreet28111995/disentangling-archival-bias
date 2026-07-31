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
3. clip_audit.py              -> Run OpenAI CLIP image-text scoring
4. openclip_audit.py          -> Run OpenCLIP image-text scoring
```

## Installation

```bash
git clone https://github.com/manpreet28111995/disentangling-archival-bias.git
cd disentangling-archival-bias

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

### 1. Harvest metadata

```bash
python fetch_met_data.py --departments 1 6 8 9 11 12 15 19 21 --max-objects 1500 --out met_metadata.csv
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
```

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
- `gender_representation.csv`
- `gender_by_century.csv`
- `met_metadata_enriched.csv`

The reported associations apply only to this score instrument, museum collection, and selected named-attribution cohort. They do not establish causal model effects or general fairness.

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
