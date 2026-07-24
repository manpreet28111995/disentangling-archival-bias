# Disentangling Algorithmic Bias from Archival Artifacts

[![Paper Target](https://img.shields.io/badge/Journal-AI%20%26%20Society%20(Springer)-blue)](https://www.springer.com/journal/146)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official repository and audit pipeline for the paper: **"Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Museum Collections"** (targeted for *AI & Society: Journal of Knowledge, Culture and Communication*, Springer).

---

## Overview

This repository provides an end-to-end quantitative audit framework that evaluates representational and aesthetic valuation bias in Vision-Language Models (VLMs)—specifically **OpenAI CLIP** and **OpenCLIP**—using digitized artwork metadata harvested from the Metropolitan Museum of Art Open Access API.

The pipeline isolates demographic main effects (artist gender) from structural collection confounders (physical artwork medium and photographic aspect ratio) via non-parametric hypothesis testing (Mann-Whitney $U$), rank-biserial effect sizes ($r$), bootstrapped 95% confidence intervals, and multivariate Ordinary Least Squares (OLS) regression.

---

## Pipeline Architecture

```
1. fetch_met_data.py          -> Harvesting RESTful API metadata & images (.met_cache)
2. analyze_representation.py  -> Archival representation analysis & metadata enrichment
3. clip_audit.py              -> OpenAI CLIP (ViT-B/32) zero-shot valuation audit
4. openclip_audit.py          -> OpenCLIP (LAION-2B) cross-model validation audit
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/disentangling-archival-bias.git
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
Harvest public domain artwork metadata across European Paintings (Department 11) and Modern/Contemporary Art (Department 21):

```bash
python fetch_met_data.py --departments 11 21 --max-objects 1500 --out met_metadata.csv
```

### Step 2: Perform Archival Representation Audit
Enrich metadata with artist gender resolution (`gender-guesser`), temporal century bucketing, and physical medium categorization:

```bash
python analyze_representation.py --in met_metadata.csv --out-dir results
```

### Step 3: Run OpenAI CLIP Zero-Shot Valuation Audit
Audit zero-shot probability differential scores across three distinct value prompt formulations (*Masterpiece*, *Quality*, *Influence*):

```bash
python clip_audit.py --in results/met_metadata_enriched.csv --out-dir results --n-female 150 --n-male 200
```

### Step 4: Run OpenCLIP (LAION-2B) Cross-Model Validation Audit
Perform cross-model audit using uncurated open-weights LAION-2B pretraining models:

```bash
python openclip_audit.py --in results/met_metadata_enriched.csv --out-dir results
```

---

## Repository Structure

```
.
├── Paper/
│   ├── Paper.tex                  # Primary LaTeX manuscript (Springer sn-jnl template)
│   ├── Bibliography.bib           # Complete BibTeX citations
│   └── Paper.pdf                  # Compiled manuscript PDF
├── fetch_met_data.py              # Met Open Access API harvester with caching
├── analyze_representation.py      # Demographic resolution & archival audit script
├── clip_audit.py                  # OpenAI CLIP (ViT-B/32) audit engine
├── openclip_audit.py              # OpenCLIP (LAION-2B) cross-model audit engine
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git exclusion rules
├── LICENSE                        # MIT License
└── README.md                      # Repository documentation
```

---

## Citation

If you use this codebase, dataset pipeline, or audit framework in your research, please cite our paper:

### APA 7th Edition
> Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Museum Collections. (2026). *AI & Society: Journal of Knowledge, Culture and Communication*. Springer Nature.

### BibTeX
```bibtex
@article{disentangling_vlm_archival_bias_2026,
  title     = {Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Museum Collections},
  journal   = {AI \& Society: Journal of Knowledge, Culture and Communication},
  publisher = {Springer Nature},
  year      = {2026}
}
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
