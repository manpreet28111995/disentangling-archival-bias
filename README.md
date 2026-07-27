# Disentangling Algorithmic Bias from Archival Artifacts

[![Paper Target](https://img.shields.io/badge/Journal-AI%20%26%20Society%20(Springer)-blue)](https://www.springer.com/journal/146)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official repository and audit pipeline for the paper: **"Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Metropolitan Museum Archives"** (targeted for *AI & Society: Journal of Knowledge, Culture and Communication*, Springer).

---

## Overview

This repository provides an end-to-end quantitative audit framework that evaluates representational and aesthetic valuation bias in Vision-Language Models (VLMs)—specifically **OpenAI CLIP** (ViT-B/32) and **OpenCLIP** (LAION-2B)—using digitized artwork metadata and real high-resolution artwork images harvested from the Metropolitan Museum of Art Open Access RESTful API.

The pipeline isolates demographic main effects (artist gender) from structural collection confounders (physical artwork medium, creation era, and photographic framing aspect ratio) via non-parametric hypothesis testing (Mann-Whitney $U$), rank-biserial effect sizes ($r$), Two One-Sided Tests for equivalence (TOST), bootstrapped 95% confidence intervals, and multivariate Ordinary Least Squares (OLS) regression with HC3 robust standard errors.

---

## Dataset Accounting & Real Image Pipeline

The audit pipeline evaluates primary visual assets downloaded directly from museum API endpoints:

```
Harvested RESTful Records (N = 1,500)
  ├── Unattributed / Anonymous Holdings (n = 618, 41.2%)
  └── Named Attributed Harvest (n = 882, 58.8%)
        ├── Excluded (Missing / Broken Image URLs): n = 139 (15.8%)
        └── Final Audited Real-Image Cohort: N = 743 (534 Male, 209 Female)
```

---

## Pipeline Architecture

```
1. fetch_met_data.py          -> Harvest RESTful API metadata across 9 curatorial departments (.met_cache)
2. analyze_representation.py  -> Enrich metadata with gender resolution, century bucketing, and medium classification
3. clip_audit.py              -> OpenAI CLIP (ViT-B/32) zero-shot audit over real downloaded image assets
4. openclip_audit.py          -> OpenCLIP (LAION-2B) cross-model audit over real downloaded image assets
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manpreet28111995/disentangling-archival-bias.git
cd disentangling-archival-bias

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution Guide

### Step 1: Harvest Metadata from Museum API
Harvest public domain artwork metadata across 9 curatorial departments (Departments 1, 6, 8, 9, 11, 12, 15, 19, 21):

```bash
python fetch_met_data.py --departments 1 6 8 9 11 12 15 19 21 --max-objects 1500 --out met_metadata.csv
```

### Step 2: Perform Archival Representation Audit
Enrich metadata with artist gender resolution (`gender-guesser` dictionary lookup), temporal century bucketing, and physical medium categorization:

```bash
python analyze_representation.py --in met_metadata.csv --out-dir results
```

### Step 3: Run OpenAI CLIP Real-Image Valuation Audit
Dynamically resolve primary high-resolution image URLs (`primaryImageSmall`), download RGB image tensors, compute image aspect ratios, and evaluate zero-shot probability differential scores across value prompt pairs (*Masterpiece*, *Quality*, *Influence*):

```bash
python clip_audit.py --in results/met_metadata_enriched.csv --out-dir results
```

### Step 4: Run OpenCLIP (LAION-2B) Cross-Model Audit
Perform cross-model validation over real downloaded artwork images using open-weights LAION-2B pretraining backbones:

```bash
python openclip_audit.py --in results/met_metadata_enriched.csv --out-dir results
```

---

## Repository Structure

```
.
├── Paper/                         # LaTeX manuscript source files (Paper.tex, Letter.tex, Bibliography.bib)
├── fetch_met_data.py              # Met Open Access API harvester with caching
├── analyze_representation.py      # Demographic resolution & archival audit script
├── clip_audit.py                  # OpenAI CLIP (ViT-B/32) real-image audit engine
├── openclip_audit.py              # OpenCLIP (LAION-2B) real-image cross-model audit engine
├── met_metadata.csv               # Harvested Met Open Access metadata corpus (N=1,500)
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
└── README.md                      # Repository documentation
```

---

## Citation

If you use this codebase, real-image pipeline, or audit framework in your research, please cite our paper:

### APA 7th Edition
> Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Metropolitan Museum Archives. (2026). *AI & Society: Journal of Knowledge, Culture and Communication*. Springer Nature.

### BibTeX
```bibtex
@article{disentangling_vlm_archival_bias_2026,
  title     = {Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Metropolitan Museum Archives},
  journal   = {AI \& Society: Journal of Knowledge, Culture and Communication},
  publisher = {Springer Nature},
  year      = {2026}
}
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
