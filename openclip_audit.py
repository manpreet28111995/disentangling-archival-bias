"""
openclip_audit.py

Evaluates OpenCLIP (laion/CLIP-ViT-B-32-laion2B-s34B-b79K) on the harvested artwork dataset
for cross-model validation.

Includes:
- Multi-prompt set evaluation across LAION-2B weights
- Mann-Whitney U test & Rank-biserial effect sizes
- Bootstrapped 95% confidence intervals
- Multivariate OLS regression (confound control)
"""

import argparse
import os
import io
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from transformers import CLIPProcessor, CLIPModel

PROMPT_SETS = {
    "set_1_masterpiece": {
        "high": "an important masterpiece of fine art",
        "low": "a minor, forgettable work of art"
    },
    "set_2_quality": {
        "high": "a museum-quality masterwork",
        "low": "an amateur painting"
    },
    "set_3_influence": {
        "high": "a groundbreaking and influential artwork",
        "low": "a decorative craft object"
    }
}

NEUTRAL_PROMPTS = ["a painting", "a photograph of an artwork"]
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def load_model():
    print(f"Loading OpenCLIP model weights: {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def download_image(url, timeout=15):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None


def score_image(model, processor, image, prompts):
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).tolist()
    return probs


def bootstrap_ci(group1, group2, n_boot=1000, ci=95, seed=42):
    np.random.seed(seed)
    diffs = []
    g1 = np.array(group1)
    g2 = np.array(group2)
    for _ in range(n_boot):
        sample1 = np.random.choice(g1, size=len(g1), replace=True)
        sample2 = np.random.choice(g2, size=len(g2), replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return np.mean(diffs), lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=str, default="results/met_metadata_enriched.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.infile)
    print(f"Loaded {len(df)} artwork metadata records for OpenCLIP audit.")

    model, processor = load_model()

    all_prompts = []
    for pset in PROMPT_SETS.values():
        all_prompts.extend([pset["high"], pset["low"]])
    all_prompts.extend(NEUTRAL_PROMPTS)
    all_prompts = list(dict.fromkeys(all_prompts))

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring images with OpenCLIP"):
        img_url = row.get("primaryImageSmall") or row.get("primaryImage")
        if not img_url or not isinstance(img_url, str):
            continue
        img = download_image(img_url)
        if img is None:
            continue

        probs = score_image(model, processor, img, all_prompts)
        prob_dict = dict(zip(all_prompts, probs))

        rec = {
            "objectID": row["objectID"],
            "title": row.get("title"),
            "artistDisplayName": row.get("artistDisplayName"),
            "inferred_gender": row.get("inferred_gender"),
            "century": row.get("century", "unknown"),
            "medium_category": row.get("medium_category", "other"),
            "img_width": img.width,
            "img_height": img.height,
            "aspect_ratio": round(img.width / img.height, 2),
        }

        set_scores = []
        for set_name, pset in PROMPT_SETS.items():
            diff = prob_dict[pset["high"]] - prob_dict[pset["low"]]
            rec[f"score_{set_name}"] = diff
            set_scores.append(diff)

        rec["mean_value_score"] = np.mean(set_scores)
        records.append(rec)

    results = pd.DataFrame(records)
    results.to_csv(os.path.join(args.out_dir, "openclip_scores.csv"), index=False)
    print(f"Saved {len(results)} scored images to openclip_scores.csv")

    # --- Statistical Analysis ---
    male_res = results[results["inferred_gender"] == "male"]
    female_res = results[results["inferred_gender"] == "female"]

    report_lines = ["=" * 60, "CROSS-MODEL AUDIT REPORT: OpenCLIP (LAION-2B) ARTISTIC VALUE SCORES", "=" * 60, ""]

    if len(male_res) > 0 and len(female_res) > 0:
        m_scores = male_res["mean_value_score"]
        f_scores = female_res["mean_value_score"]

        # 1. Mann-Whitney U test + Effect Size
        stat, p_val = mannwhitneyu(m_scores, f_scores, alternative="two-sided")
        n1, n2 = len(m_scores), len(f_scores)
        rank_biserial = 1 - (2 * stat) / (n1 * n2)

        report_lines.append("1. NON-PARAMETRIC TEST (MALE vs FEMALE)")
        report_lines.append(f"   Male (N={n1}): mean={m_scores.mean():.4f}, std={m_scores.std():.4f}")
        report_lines.append(f"   Female (N={n2}): mean={f_scores.mean():.4f}, std={f_scores.std():.4f}")
        report_lines.append(f"   Mann-Whitney U = {stat:.2f}, p-value = {p_val:.4e}")
        report_lines.append(f"   Rank-Biserial Correlation r = {rank_biserial:.4f}")
        report_lines.append("")

        # 2. Bootstrap Confidence Intervals
        mean_diff, ci_low, ci_high = bootstrap_ci(m_scores, f_scores)
        report_lines.append("2. BOOTSTRAP 95% CONFIDENCE INTERVAL (MALE MEAN - FEMALE MEAN)")
        report_lines.append(f"   Mean Difference = {mean_diff:.4f}")
        report_lines.append(f"   95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
        report_lines.append("")

        # 3. Robustness per prompt set
        report_lines.append("3. PROMPT SET ROBUSTNESS CHECK")
        for set_name in PROMPT_SETS.keys():
            col = f"score_{set_name}"
            m_s = male_res[col]
            f_s = female_res[col]
            u_s, p_s = mannwhitneyu(m_s, f_s, alternative="two-sided")
            r_s = 1 - (2 * u_s) / (n1 * n2)
            report_lines.append(f"   [{set_name}] Male mean={m_s.mean():.4f} | Female mean={f_s.mean():.4f} | p={p_s:.4f} | r={r_s:.4f}")
        report_lines.append("")

        # 4. OLS Confound Control Regression
        report_lines.append("4. MULTIVARIATE OLS REGRESSION (CONFOUND CONTROL)")
        try:
            model_ols = smf.ols(
                "mean_value_score ~ C(inferred_gender) + C(medium_category) + aspect_ratio",
                data=results
            ).fit()
            report_lines.append(str(model_ols.summary()))
        except Exception as e:
            report_lines.append(f"   OLS regression error: {e}")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.out_dir, "openclip_academic_stats_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print("\n" + report_text)

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(6, 4.5))
    results.boxplot(column="mean_value_score", by="inferred_gender", ax=ax, grid=False)
    ax.set_title("OpenCLIP (LAION-2B) Value Score by Artist Gender")
    ax.set_ylabel("Mean Relative Value Prompt Score")
    ax.set_xlabel("Inferred Artist Gender")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "openclip_bias_boxplot.png"), dpi=150)
    plt.close(fig)

    print(f"Saved OpenCLIP audit report and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
