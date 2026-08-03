"""Create manuscript figures from completed merged-run model outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v6" / "models"
OUT = RESULTS / "manuscript_figures"
PROMPTS = {
    "score_set_1_masterpiece": "Masterpiece",
    "score_set_2_quality": "Quality",
    "score_set_3_influence": "Influence",
    "score_set_4_neutral_control": "Neutral control",
}
MODELS = [
    ("OpenAI CLIP", "openai_clip_prompt_comparisons.csv"),
    ("OpenCLIP B/32", "openclip_prompt_comparisons.csv"),
    ("OpenCLIP L/14", "openclip_l14_prompt_comparisons.csv"),
    ("SigLIP", "siglip_prompt_comparisons.csv"),
]
COLORS = {"OpenAI CLIP": "#0072B2", "OpenCLIP B/32": "#D55E00",
          "OpenCLIP L/14": "#009E73", "SigLIP": "#CC79A7"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def forest_plot():
    rows = []
    for label, filename in MODELS:
        data = pd.read_csv(RESULTS / filename)
        data = data[data["score"].isin(PROMPTS)]
        for _, row in data.iterrows():
            rows.append({
                "model": label,
                "prompt": PROMPTS[row["score"]],
                "difference": row["mean_difference_male_minus_female"],
                "low": row["ci_low"],
                "high": row["ci_high"],
                "p": row["bonferroni_p"],
            })
    data = pd.DataFrame(rows)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    for i, row in data.iterrows():
        ax.errorbar(row.difference, y[i],
                    xerr=[[row.difference - row.low], [row.high - row.difference]],
                    fmt="o", color=COLORS[row.model], ecolor=COLORS[row.model],
                    capsize=3, markersize=5, linewidth=1.4)
        if row.p < 0.05:
            ax.text(row.high + 0.006, y[i], "*", va="center", fontsize=12)
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.model}: {r.prompt}" for _, r in data.iterrows()])
    ax.set_xlabel("Male − female mean score difference")
    ax.set_title("Prompt-relative score differences across models")
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", label=m)
               for m, c in COLORS.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.text(0.01, 0.01, "Points show mean differences; bars show 95% CIs. * Bonferroni-adjusted p < .05.",
             ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "figure_1_prompt_contrasts")


def distributions():
    frames = []
    for label, filename in [("OpenAI CLIP", "clip_scores.csv"),
                            ("OpenCLIP B/32", "openclip_scores.csv"),
                            ("OpenCLIP L/14", "openclip_l14_scores.csv"),
                            ("SigLIP", "siglip_scores.csv")]:
        data = pd.read_csv(RESULTS / filename)
        data = data[data.inferred_gender.isin(["male", "female"])].copy()
        data["model"] = label
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 8.2), sharey="row")
    for i, model in enumerate([m[0] for m in MODELS]):
        for j, (column, prompt) in enumerate(PROMPTS.items()):
            ax = axes[i, j]
            subset = data[data.model == model]
            values = [subset.loc[subset.inferred_gender == g, column].dropna().to_numpy()
                      for g in ["female", "male"]]
            ax.boxplot(values, positions=[0, 1], widths=0.5, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": "0.25"},
                       medianprops={"color": "0.15"}, whiskerprops={"color": "0.35"},
                       capprops={"color": "0.35"}, flierprops={"marker": "", "markersize": 0})
            rng = np.random.default_rng(42 + i * 3 + j)
            for x, vals, color in zip([0, 1], values, ["#CC79A7", "#0072B2"]):
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(vals)), vals,
                           s=10, alpha=0.55, color=color, edgecolors="none")
            ax.set_xticks([0, 1], ["Female", "Male"])
            ax.set_title(prompt)
            ax.grid(axis="y", color="0.9", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{model}\nRelative score")
    fig.suptitle("Score distributions by inferred creator category", y=1.01)
    fig.text(0.01, 0.01, "Boxes show interquartile ranges; points show individual records. Gender labels are name-inferred categories.",
             ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save(fig, "figure_S1_score_distributions")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    forest_plot()
    distributions()
    print(f"Wrote figures to {OUT}")
