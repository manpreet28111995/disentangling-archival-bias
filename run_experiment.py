"""Run complete real-data museum VLM experiment from scratch.

Pipeline: Met API harvest -> metadata enrichment -> four-model audit -> reports.
No synthetic data path.
"""

import argparse
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_appendix(out, args, raw, cohort, results):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unavailable"
    raw_df, cohort_df = pd.read_csv(raw), pd.read_csv(cohort)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    lines = [
        "# Experiment Runtime Appendix", "",
        f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"Git commit: `{commit}`", f"Platform: `{platform.platform()}`",
        f"Python: `{platform.python_version()}`", f"PyTorch: `{torch.__version__}`",
        f"Runtime device: `{device}`", "", "## Exact command", "",
        "```bash",
        f"python3 run_experiment.py --out-dir {args.out_dir} --harvest-only",
        "```", "", "## Metropolitan Museum API query", "",
        "- Base: `https://collectionapi.metmuseum.org/public/collection/v1`",
        f"- Departments: `{', '.join(map(str, args.departments))}`",
        "- Candidate: `/objects?departmentIds=1|6|8|9|11|12|15|19|21`",
        "- Fallback: `/search?departmentId=<id>&q=<term>&hasImages=true`",
        "- Terms: `painting, drawing, print, photograph, sculpture, woman, female, portrait, art, artist`",
        "- Object endpoint: `/objects/<objectID>`; image fields: `primaryImage`, `primaryImageSmall`.",
        "- Retention rule: non-empty real `primaryImage` required.", "", "## Dataset accounting", "",
        f"- API metadata rows saved: **{len(raw_df)}**",
        f"- Locked real-image cohort: **{len(cohort_df)}**",
        "- No minimum image threshold; exact real-image count is reported below.",
        "- `objectID` is primary reproduction key.", "", "## Models and scoring", "",
        "| Label | Model | Score |", "|---|---|---|",
        "| OpenAI_CLIP | `openai/clip-vit-base-patch32` | softmax probability difference |",
        "| OpenCLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | softmax probability difference |",
        "| OpenCLIP_L14 | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | softmax probability difference |",
        "| SigLIP | `google/siglip-base-patch16-224` | logit difference |", "",
        "- Masterpiece: `an important masterpiece of fine art` vs `a minor, forgettable work of art`",
        "- Quality: `a museum-quality masterwork` vs `an amateur painting`",
        "- Influence: `a groundbreaking and influential artwork` vs `a decorative craft object`", "",
        "## Reproduction artifacts", "",
        f"- Raw metadata: `{raw}`", f"- Locked cohort: `{cohort}`", f"- Model artifacts: `{results}/`",
        "- Object IDs: `object_selection_manifest.csv` and model-specific `*_image_manifest.csv`.",
        "- Novelty outputs: `models/novelty/object_risk_ranking.csv`, `department_consensus.csv`, and `consensus_summary.csv`.",
        "- Reports: model-specific `*_publication_report.txt`.", "", "## Interpretation boundary", "",
        "Visual-model results use all real-image records; gender summaries use named-attribution subset. Results do not establish causal effects or population-wide fairness.", "",
    ]
    (out / "appendix.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results_full_run")
    parser.add_argument("--metadata", type=Path,
                        help="Existing real Met metadata CSV; skip API harvest")
    parser.add_argument("--harvest-only", action="store_true",
                        help="Fetch/enrich real records, then stop before model inference")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pipeline stages whose output files already exist")
    parser.add_argument(
        "--departments", nargs="+", type=int,
        default=list(range(1, 26)),
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = args.metadata.resolve() if args.metadata else out / "met_metadata.csv"
    enriched_dir = out / "enrichment"
    results = out / "models"
    enriched_dir.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    if args.metadata:
        print(f"Using existing real metadata: {raw}")
    elif args.resume and raw.exists():
        print(f"Reusing existing metadata: {raw}")
    else:
        run([
            sys.executable, "fetch_met_data.py",
            "--departments", *map(str, args.departments),
            "--out", str(raw),
        ])
    enriched = enriched_dir / "met_metadata_enriched.csv"
    if not (args.resume and enriched.exists()):
        run([
            sys.executable, "analyze_representation.py",
            "--in", str(raw),
            "--out-dir", str(enriched_dir),
        ])
    else:
        print(f"Reusing existing enrichment: {enriched}")

    if args.harvest_only:
        print(f"\nHarvest complete. Real image records: {len(pd.read_csv(raw))}")
        return

    clip_scores = results / "clip_scores.csv"
    clip_manifest = results / "openai_clip_image_manifest.csv"
    if not (args.resume and clip_scores.exists() and clip_manifest.exists()):
        run([sys.executable, "clip_audit.py", "--in", str(enriched), "--out-dir", str(results)])
    else:
        print("Reusing OpenAI CLIP results")
    # Lock one real-image objectID cohort for every model.
    manifest = pd.read_csv(results / "openai_clip_image_manifest.csv")
    included = manifest.loc[manifest.image_status == "included", "objectID"]
    cohort_path = out / "met_metadata_evaluation_cohort.csv"
    if args.resume and cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        print(f"Reusing evaluation cohort: {cohort_path}")
    else:
        cohort = pd.read_csv(enriched)
        cohort = cohort[cohort.objectID.isin(included)]
        cohort.to_csv(cohort_path, index=False)
    print(f"Locked common evaluation cohort: {len(cohort)} objectIDs")

    stages = [
        ("OpenCLIP", results / "openclip_scores.csv", results / "openclip_image_manifest.csv", ["openclip_audit.py"]),
        ("OpenCLIP_L14", results / "openclip_l14_scores.csv", results / "openclip_l14_image_manifest.csv", ["additional_model_audit.py", "--model-name", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "--label", "OpenCLIP_L14"]),
        ("SigLIP", results / "siglip_scores.csv", results / "siglip_image_manifest.csv", ["additional_model_audit.py", "--model-name", "google/siglip-base-patch16-224", "--label", "SigLIP"]),
    ]
    for label, score_file, manifest_file, command in stages:
        if args.resume and score_file.exists() and manifest_file.exists():
            print(f"Reusing {label} results")
            continue
        run([sys.executable, *command, "--in", str(cohort_path), "--out-dir", str(results)])
    novelty_dir = results / "novelty"
    if not (args.resume and (novelty_dir / "consensus_summary.csv").exists()):
        run([sys.executable, "novelty_analysis.py", "--models", str(results),
             "--metadata", str(cohort_path), "--out-dir", str(novelty_dir)])
    else:
        print("Reusing novelty analysis")
    write_appendix(out, args, raw, cohort_path, results)
    print(f"\nComplete. Real-data artifacts: {out}")


if __name__ == "__main__":
    main()
