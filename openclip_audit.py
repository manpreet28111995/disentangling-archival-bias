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
from audit_statistics import run_publishable_analysis
from device_utils import get_device, move_inputs

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
    },
    "set_4_neutral_control": {
        "high": "a museum-quality masterwork",
        "low": "an artwork"
    }
}

NEUTRAL_PROMPTS = ["a painting", "a photograph of art", "a museum object"]
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def load_model(device):
    print(f"Loading OpenCLIP model weights: {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    return model, processor


def resolve_met_image_url(object_id, cache_dir=".met_cache", timeout=10):
    """Query the Met Open Access API for the real image URL of an object."""
    import json
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{object_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                return data.get("primaryImageSmall") or data.get("primaryImage")
        except Exception:
            pass
    try:
        resp = requests.get(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data.get("primaryImageSmall") or data.get("primaryImage")
    except Exception as e:
        print(f"  [WARN] Met API lookup failed for {object_id}: {e}")
    return None

def image_url(row):
    return row.get("primaryImageSmall") or row.get("primaryImage") or resolve_met_image_url(row.get("objectID"))


def download_image(url, timeout=10, object_id=None):
    """Download image from URL via HTTP. Returns PIL Image or None on failure."""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"  [WARN] Failed to download image for object {object_id}: {e}")
        return None


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


def batch_score_images(model, processor, images, prompts, device, batch_size=32):
    all_probs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = move_inputs(processor(text=prompts, images=batch, return_tensors="pt", padding=True), device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
    return all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=str, default="results/met_metadata_enriched.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    df = pd.read_csv(args.infile)
    print(f"Loaded {len(df)} artwork metadata records for OpenCLIP audit.")

    model, processor = load_model(device)

    all_prompts = []
    for pset in PROMPT_SETS.values():
        all_prompts.extend([pset["high"], pset["low"]])
    all_prompts.extend(NEUTRAL_PROMPTS)
    all_prompts = list(dict.fromkeys(all_prompts))

    images = []
    valid_rows = []
    image_manifest = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images for OpenCLIP"):
        oid = row.get("objectID")
        real_url = image_url(row)
        img = download_image(real_url, object_id=oid)
        image_manifest.append({
            "objectID": oid,
            "source_image_url": real_url,
            "image_status": "included" if img is not None else "excluded_image_unavailable",
            "model": "OpenCLIP",
        })
        if img is not None:
            images.append(img)
            valid_rows.append(row)
    print(f"Successfully loaded {len(images)} / {len(df)} images")

    print(f"Executing batch OpenCLIP inference for {len(images)} images...")
    probs_list = batch_score_images(model, processor, images, all_prompts, device, batch_size=32)

    records = []
    for row, img, probs in zip(valid_rows, images, probs_list):
        prob_dict = dict(zip(all_prompts, probs))

        rec = {
            "objectID": row["objectID"],
            "department": row.get("department"),
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
            diff = float(prob_dict[pset["high"]] - prob_dict[pset["low"]])
            rec[f"score_{set_name}"] = diff
            set_scores.append(diff)

        rec["mean_value_score"] = float(np.mean(set_scores))
        records.append(rec)

    results = pd.DataFrame(records)
    pd.DataFrame(image_manifest).to_csv(
        os.path.join(args.out_dir, "openclip_image_manifest.csv"), index=False
    )
    results.to_csv(os.path.join(args.out_dir, "openclip_scores.csv"), index=False)
    print(f"Saved {len(results)} scored images to openclip_scores.csv")

    male_res = results[results["inferred_gender"] == "male"]
    female_res = results[results["inferred_gender"] == "female"]

    report_lines = ["=" * 60, "CROSS-MODEL AUDIT REPORT: OpenCLIP (LAION-2B) ARTISTIC VALUE SCORES", "=" * 60, ""]

    if len(male_res) > 0 and len(female_res) > 0:
        m_scores = male_res["mean_value_score"]
        f_scores = female_res["mean_value_score"]

        stat, p_val = mannwhitneyu(m_scores, f_scores, alternative="two-sided")
        n1, n2 = len(m_scores), len(f_scores)
        rank_biserial = 1 - (2 * stat) / (n1 * n2)

        report_lines.append("1. NON-PARAMETRIC TEST (MALE vs FEMALE)")
        report_lines.append(f"   Male (N={n1}): mean={m_scores.mean():.4f}, std={m_scores.std():.4f}")
        report_lines.append(f"   Female (N={n2}): mean={f_scores.mean():.4f}, std={f_scores.std():.4f}")
        report_lines.append(f"   Mann-Whitney U = {stat:.2f}, p-value = {p_val:.4e}")
        report_lines.append(f"   Rank-Biserial Correlation r = {rank_biserial:.4f}")
        report_lines.append("")

        mean_diff, ci_low, ci_high = bootstrap_ci(m_scores, f_scores)
        report_lines.append("2. BOOTSTRAP 95% CONFIDENCE INTERVAL (MALE MEAN - FEMALE MEAN)")
        report_lines.append(f"   Mean Difference = {mean_diff:.4f}")
        report_lines.append(f"   95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
        report_lines.append("")

        report_lines.append("3. PROMPT SET ROBUSTNESS CHECK")
        for set_name in PROMPT_SETS.keys():
            col = f"score_{set_name}"
            m_s = male_res[col]
            f_s = female_res[col]
            u_s, p_s = mannwhitneyu(m_s, f_s, alternative="two-sided")
            r_s = 1 - (2 * u_s) / (n1 * n2)
            report_lines.append(f"   [{set_name}] Male mean={m_s.mean():.4f} | Female mean={f_s.mean():.4f} | p={p_s:.4f} | r={r_s:.4f}")
        report_lines.append("")

        report_lines.append("4. MULTIVARIATE OLS REGRESSION (CONFOUND CONTROL)")
        try:
            model_ols = smf.ols(
                "mean_value_score ~ C(inferred_gender) + C(medium_category) + C(century) + aspect_ratio",
                data=results
            ).fit(cov_type="HC3")
            report_lines.append(str(model_ols.summary()))
            
            model_rlm = smf.rlm(
                "mean_value_score ~ C(inferred_gender) + C(medium_category) + C(century) + aspect_ratio",
                data=results
            ).fit()
            report_lines.append("\nHUBER ROBUST LINEAR MODEL (RLM) CROSS-VALIDATION:")
            report_lines.append(str(model_rlm.summary()))
        except Exception as e:
            report_lines.append(f"   OLS regression error: {e}")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(args.out_dir, "openclip_academic_stats_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print("\n" + report_text)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    results.boxplot(column="mean_value_score", by="inferred_gender", ax=ax, grid=False)
    ax.set_title("OpenCLIP (LAION-2B) Value Score by Artist Gender")
    ax.set_ylabel("Mean Relative Value Prompt Score")
    ax.set_xlabel("Inferred Artist Gender")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "openclip_bias_boxplot.png"), dpi=150)
    plt.close(fig)

    run_publishable_analysis(results, "OpenCLIP", args.out_dir)

    print(f"Saved OpenCLIP audit report and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
