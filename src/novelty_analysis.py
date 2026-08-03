"""Cross-model consensus and object-level instability audit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROMPTS = ["score_set_1_masterpiece", "score_set_2_quality", "score_set_3_influence"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="Model score CSV directory")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tables = []
    for path in sorted(Path(args.models).glob("*_scores.csv")):
        d = pd.read_csv(path)
        cols = [c for c in PROMPTS + ["mean_value_score"] if c in d]
        d = d[["objectID"] + cols].copy().set_index("objectID")
        d.columns = [f"{path.stem}_{c}" for c in d.columns]
        tables.append(d)
    if not tables:
        raise RuntimeError("No model score CSVs found")
    scores = pd.concat(tables, axis=1, join="inner")
    model_means = [c for c in scores if c.endswith("mean_value_score")]
    # Standardize each model before consensus; raw CLIP probabilities and SigLIP logits differ in scale.
    standardized = scores[model_means].apply(lambda c: (c - c.mean()) / c.std(ddof=0), axis=0)
    standardized.columns = [f"z_{c}" for c in model_means]
    scores = pd.concat([scores, standardized], axis=1)
    z_means = list(standardized.columns)
    prompt_cols = [c for c in scores if any(c.endswith(p) for p in PROMPTS)]
    scores["cross_model_mean_raw"] = scores[model_means].mean(axis=1)
    scores["cross_model_mean"] = scores[z_means].mean(axis=1)
    scores["cross_model_sd"] = scores[z_means].std(axis=1).fillna(0)
    scores["prompt_sd"] = scores[prompt_cols].std(axis=1).fillna(0)
    scores["model_rank_disagreement"] = scores[model_means].rank(axis=0, pct=True).std(axis=1).fillna(0)
    scores["risk_score"] = scores[["cross_model_sd", "prompt_sd", "model_rank_disagreement"]].sum(axis=1)
    meta = pd.read_csv(args.metadata).set_index("objectID")
    keep = [c for c in ["title", "department", "classification", "culture", "medium", "artistDisplayName", "inferred_gender", "century", "medium_category"] if c in meta]
    scores.join(meta[keep], how="left").sort_values("risk_score", ascending=False).to_csv(out / "object_risk_ranking.csv")
    joined = scores.join(meta[keep], how="left")
    joined.groupby("department", dropna=False).agg(
        n=("cross_model_mean", "size"), mean_score=("cross_model_mean", "mean"),
        model_sd=("cross_model_sd", "mean"), prompt_sd=("prompt_sd", "mean"),
        risk=("risk_score", "mean")
    ).sort_values("risk", ascending=False).to_csv(out / "department_consensus.csv")
    pd.DataFrame({
        "metric": ["objects", "models", "mean_cross_model_sd", "mean_prompt_sd", "mean_risk_score"],
        "value": [len(scores), len(model_means), scores.cross_model_sd.mean(), scores.prompt_sd.mean(), scores.risk_score.mean()]
    }).to_csv(out / "consensus_summary.csv", index=False)
    (out / "novelty_report.md").write_text(
        "# Cross-model novelty audit\n\n"
        "`cross_model_sd` measures model disagreement; `prompt_sd` measures prompt instability; "
        "`risk_score` ranks objects requiring manual review. These are diagnostic scores, not ground truth.\n"
    )
    print(f"Saved novelty audit for {len(scores)} objects and {len(model_means)} models")

if __name__ == "__main__":
    main()
