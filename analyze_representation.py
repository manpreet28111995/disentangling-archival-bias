"""
analyze_representation.py

Audits representation gaps in the fetched Met Museum metadata:
- Gender of attributed artists (using Met's own field where present,
  falling back to first-name gender inference for the rest)
- Nationality / culture spread
- How these gaps shift over historical period

Usage:
    python analyze_representation.py --in met_metadata.csv --out-dir results
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import gender_guesser.detector as gender

detector = gender.Detector(case_sensitive=False)

GENDER_MAP = {
    "male": "male", "mostly_male": "male",
    "female": "female", "mostly_female": "female",
    "andy": "unknown", "unknown": "unknown",
}


def extract_first_name(display_name):
    """Best-effort extraction of a first name from Met's 'artistDisplayName' field."""
    if not isinstance(display_name, str) or not display_name.strip():
        return None
    # Met sometimes lists multiple artists separated by '|'; take the first
    name = display_name.split("|")[0].strip()
    # Strip leading titles
    name = re.sub(r"^(Sir|Dame|Madame|Mademoiselle|Monsieur)\s+", "", name)
    parts = name.split()
    return parts[0] if parts else None


def infer_gender(display_name):
    first = extract_first_name(display_name)
    if not first:
        return "unknown"
    guess = detector.get_gender(first)
    return GENDER_MAP.get(guess, "unknown")


def resolve_gender(row):
    """Prefer Met's own artistGender field if populated; fall back to name-based inference.
    Note: Met Open Access dataset populates artistGender primarily for female artists ('Female')
    and leaves blank for male/unknown artists. Blanks are treated as unpopulated fall-through.
    """
    met_gender = str(row.get("artistGender") or "").strip().lower()
    if "female" in met_gender:
        return "female"
    if "male" in met_gender:
        return "male"
    return infer_gender(row.get("artistDisplayName"))


def century_bucket(begin_date):
    try:
        year = int(begin_date)
    except (TypeError, ValueError):
        return "unknown"
    if year == 0:
        return "unknown"
    century = ((abs(year) - 1) // 100 + 1)
    era = "BCE" if year < 0 else "CE"
    return f"{century}th c. {era}"


def categorize_medium(medium):
    if not isinstance(medium, str) or not medium.strip():
        return "other"
    m = medium.lower()
    if any(k in m for k in ["oil", "canvas", "tempera", "panel", "acrylic"]):
        return "painting"
    if any(k in m for k in ["drawing", "ink", "pencil", "graphite", "chalk", "charcoal", "watercolor"]):
        return "drawing_paper"
    if any(k in m for k in ["print", "etching", "engraving", "woodcut", "lithograph"]):
        return "print"
    if any(k in m for k in ["bronze", "marble", "sculpture", "statue", "terracotta"]):
        return "sculpture"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=str, default="met_metadata.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.infile)

    # Drop rows with no attributed artist at all (anonymous / workshop objects)
    named = df[df["artistDisplayName"].notna() & (df["artistDisplayName"].str.strip() != "")].copy()

    # Keep all real records for visual audit; named subset powers metadata summaries.
    df["inferred_gender"] = df.apply(resolve_gender, axis=1)
    df["century"] = df["objectBeginDate"].apply(century_bucket)
    df["medium_category"] = df["medium"].apply(categorize_medium)
    named = df.loc[named.index].copy()

    # Reproduction manifest: preserve every real Met object ID and selection state.
    manifest = df[[c for c in ["objectID", "title", "department", "primaryImage", "primaryImageSmall"] if c in df.columns]].copy()
    manifest["has_named_creator"] = manifest.objectID.isin(named.objectID)
    manifest["inferred_gender"] = manifest.objectID.map(named.set_index("objectID")["inferred_gender"])
    manifest["selection_stage"] = manifest.has_named_creator.map(
        {True: "named_attribution", False: "unknown_or_unattributed"}
    )
    manifest.to_csv(os.path.join(args.out_dir, "object_selection_manifest.csv"), index=False)

    # --- Gender representation summary ---
    gender_counts = named["inferred_gender"].value_counts(normalize=True) * 100
    gender_summary = gender_counts.rename("pct_of_named_works").reset_index()
    gender_summary.columns = ["gender", "pct_of_named_works"]
    gender_summary.to_csv(os.path.join(args.out_dir, "gender_representation.csv"), index=False)
    print("\nGender representation among named artists (%):")
    print(gender_summary.to_string(index=False))

    # --- Nationality representation summary (top 15) ---
    nat_counts = named["artistNationality"].fillna("unknown").value_counts(normalize=True).head(15) * 100
    nat_summary = nat_counts.rename("pct_of_named_works").reset_index()
    nat_summary.columns = ["nationality", "pct_of_named_works"]
    nat_summary.to_csv(os.path.join(args.out_dir, "nationality_representation.csv"), index=False)

    # --- Gender representation over time ---
    cross = pd.crosstab(named["century"], named["inferred_gender"], normalize="index") * 100
    cross.to_csv(os.path.join(args.out_dir, "gender_by_century.csv"))

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(6, 4))
    gender_summary.set_index("gender")["pct_of_named_works"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("% of named works")
    ax.set_title("Artist gender representation in sample")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "gender_representation.png"), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    nat_summary.set_index("nationality")["pct_of_named_works"].plot(kind="barh", ax=ax2, color="#55A868")
    ax2.set_xlabel("% of named works")
    ax2.set_title("Top 15 artist nationalities in sample")
    ax2.invert_yaxis()
    plt.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "nationality_representation.png"), dpi=150)

    # Save the enriched dataframe (with inferred gender + century) for the next stage
    df.to_csv(os.path.join(args.out_dir, "met_metadata_enriched.csv"), index=False)

    print(f"\nSaved summary tables, charts, object manifest, and enriched CSV to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
