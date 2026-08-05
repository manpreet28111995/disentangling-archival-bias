# Whose Art Counts? Model- and Prompt-Dependent Associations in Vision-Language Judgments of Museum Art

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Repository for the museum-archive audit accompanying the manuscript **"Whose Art Counts? Model- and Prompt-Dependent Associations in Vision-Language Judgments of Museum Art."**

**Authors:** Manpreet Singh (Boston University), Nandakishor Reddy Pulagam (Independent Researcher), Rhythm Bhatia (University of Eastern Finland), and Rahul Joshi (Symbiosis Institute of Technology).

## Overview

Vision-language models (VLMs) are increasingly used to search, rank, and describe cultural collections. This study introduces an **archive-conditioned audit design** to evaluate prompt-relative image-text associations across four pretrained vision-language models:

- **OpenAI CLIP** (ViT-B/32)
- **OpenCLIP** (ViT-B/32, LAION-2B)
- **OpenCLIP** (ViT-L/14, LAION-2B)
- **Google SigLIP** (base patch16/224)

The audit uses artwork metadata and public image assets from the Metropolitan Museum of Art Open Access collection. It evaluates three value-oriented prompt pairs (Masterpiece, Quality, Influence) alongside a generic framing control. Prompts are treated as operational probes of image-to-language mapping, not as validated measures of intrinsic artistic quality.

## Dataset Accounting

```text
Image-complete evaluation cohort:          N = 445 (100.0%)
  Records with creator display string:     n = 174  (39.1%)
  Records with explicit female field:      n = 8    (1.8%)
  Resolved binary comparison cohort:       n = 61   (13.7%)
    - Male-inferred creator category:      n = 39   (8.8%)
    - Female-inferred creator category:    n = 22   (4.9%)
  Unknown or unattributed category:        n = 384  (86.3%)
```

*"Male-inferred"* and *"female-inferred"* describe catalog-derived analytic labels from explicit museum fields and first-name dictionary lookup. They do not represent self-identified gender identity or verified historical truth. Unknown and unattributed works are preserved in the visual audit to prevent silently reducing the dataset to named attributions only.

## Prompt Contrasts

| Prompt Set | Higher-Value Phrase | Lower-Value Phrase |
| :--- | :--- | :--- |
| **Masterpiece** | important masterpiece of fine art | minor, forgettable work of art |
| **Quality** | museum-quality masterwork | amateur painting |
| **Influence** | groundbreaking and influential artwork | decorative craft object |
| **Neutral Control** | museum-quality masterwork | artwork |

For CLIP models, scores represent softmax probability differences; for SigLIP, scores represent logit differences. Cross-model diagnostics are computed only after within-model standardization.

## Statistical Protocol & Audit Design

1. **Archive Freeze:** Lock object IDs, image validation rules, and metadata fields before inference.
2. **Paired Inference:** Run every model on the exact same locked set of 445 image-complete objects.
3. **Within-Model Estimation:** Compute mean differences ($\Delta = \text{Male} - \text{Female}$), 95% bootstrap confidence intervals, Mann-Whitney $U$ rank tests, rank-biserial effect sizes, and within-model Bonferroni correction over value contrasts.
4. **Sensitivity & Diagnostics:** Perform 10,000-label permutation tests, leave-one-department-out aggregate sensitivity analysis, and standardized cross-model disagreement diagnostics.

## Pipeline & Execution

### Installation

```bash
git clone https://github.com/manpreet28111995/disentangling-archival-bias.git
cd disentangling-archival-bias

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Complete Four-Model Run

Run the complete experiment pipeline from scratch:

```bash
python src/run_experiment.py --out-dir results_full_run --harvest-only
```

Or run from an existing merged metadata dataset:

```bash
python3 src/run_experiment.py \
  --metadata met_metadata_merged.csv \
  --out-dir results_v6
```

On Apple Silicon, code automatically uses `mps`; otherwise it falls back to CUDA or CPU. Unsupported MPS operations use CPU fallback.

### Individual Stage Execution

```bash
# 1. Harvest Met Open Access metadata
python src/fetch_met_data.py --departments 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 --out met_metadata.csv

# 2. Enrich metadata & derive creator categories
python src/analyze_representation.py --in met_metadata.csv --out-dir results

# 3. Run OpenAI CLIP audit
python src/clip_audit.py --in results/met_metadata_enriched.csv --out-dir results

# 4. Run OpenCLIP audit (B/32)
python src/openclip_audit.py --in results/met_metadata_enriched.csv --out-dir results

# 5. Run OpenCLIP L/14 scale sensitivity audit
python src/additional_model_audit.py --in results/met_metadata_enriched.csv --out-dir results \
  --model-name laion/CLIP-ViT-L-14-laion2B-s32B-b82K --label OpenCLIP_L14

# 6. Run SigLIP sigmoid loss objective audit
python src/additional_model_audit.py --in results/met_metadata_enriched.csv --out-dir results \
  --model-name google/siglip-base-patch16-224 --label SigLIP
```

Each completed run generates score tables, object selection manifests, prompt comparison summaries, sensitivity reports, and `appendix.md` documenting environment and execution details.

## Repository Structure

```text
.
├── src/
│   ├── fetch_met_data.py
│   ├── analyze_representation.py
│   ├── model_audit.py
│   ├── clip_audit.py
│   ├── openclip_audit.py
│   ├── additional_model_audit.py
│   └── run_experiment.py
├── Paper/
│   ├── main_v6.tex
│   ├── cover_letter.tex
│   └── manuscript_figures/
├── met_metadata.csv
├── met_metadata_merged.csv
├── results_v6/
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this code or findings, please cite:

```bibtex
@article{singh2026whoseartcounts,
  title     = {Whose Art Counts? Model- and Prompt-Dependent Associations in Vision-Language Judgments of Museum Art},
  author    = {Singh, Manpreet and Pulagam, Nandakishor Reddy and Bhatia, Rhythm and Joshi, Rahul},
  year      = {2026}
}
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

