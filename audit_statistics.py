"""Reproducible, publication-facing statistics for both VLM audits."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf


PROMPT_COLUMNS = ["score_set_1_masterpiece", "score_set_2_quality", "score_set_3_influence"]


def _bootstrap_mean_difference(male, female, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    male = np.asarray(male, dtype=float)
    female = np.asarray(female, dtype=float)
    draws = rng.choice(male, (n_boot, len(male))).mean(axis=1) - rng.choice(
        female, (n_boot, len(female))
    ).mean(axis=1)
    return float(draws.mean()), *np.percentile(draws, [2.5, 97.5])


def _tost_pvalue(male, female, standardized_bound=0.30):
    """TOST for mean difference using pooled SD and Welch SE."""
    from scipy.stats import t

    male, female = np.asarray(male, float), np.asarray(female, float)
    diff = male.mean() - female.mean()
    se = np.sqrt(male.var(ddof=1) / len(male) + female.var(ddof=1) / len(female))
    pooled_sd = np.sqrt(
        ((len(male) - 1) * male.var(ddof=1) + (len(female) - 1) * female.var(ddof=1))
        / (len(male) + len(female) - 2)
    )
    bound = standardized_bound * pooled_sd
    df = (male.var(ddof=1) / len(male) + female.var(ddof=1) / len(female)) ** 2 / (
        (male.var(ddof=1) / len(male)) ** 2 / (len(male) - 1)
        + (female.var(ddof=1) / len(female)) ** 2 / (len(female) - 1)
    )
    p_lower = 1 - t.cdf((diff + bound) / se, df)
    p_upper = 1 - t.cdf((bound - diff) / se, df)
    return float(max(p_lower, p_upper)), float(bound)


def _comparison(results, column):
    male = results.loc[results.inferred_gender == "male", column].dropna()
    female = results.loc[results.inferred_gender == "female", column].dropna()
    u, p = mannwhitneyu(male, female, alternative="two-sided")
    r = 1 - 2 * u / (len(male) * len(female))
    mean_diff, ci_low, ci_high = _bootstrap_mean_difference(male, female)
    tost_p, bound = _tost_pvalue(male, female)
    return {
        "score": column,
        "n_male": len(male),
        "n_female": len(female),
        "male_mean": male.mean(),
        "female_mean": female.mean(),
        "mean_difference_male_minus_female": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mann_whitney_u": u,
        "p_value": p,
        "rank_biserial_r": r,
        "tost_p_d03": tost_p,
        "tost_bound": bound,
    }


def _permutation_pvalue(results, column, n_perm=10000, seed=42):
    values = results[column].to_numpy(float)
    labels = results.inferred_gender.to_numpy()
    observed = values[labels == "male"].mean() - values[labels == "female"].mean()
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        diff = values[shuffled == "male"].mean() - values[shuffled == "female"].mean()
        extreme += abs(diff) >= abs(observed)
    return (extreme + 1) / (n_perm + 1)


def run_publishable_analysis(results, model_name, out_dir):
    """Write transparent tables, diagnostics, and sensitivity models."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = results.copy()
    results = results[results.inferred_gender.isin(["male", "female"])].copy()

    comparison = pd.DataFrame([_comparison(results, c) for c in PROMPT_COLUMNS + ["mean_value_score"]])
    comparison["bonferroni_p"] = np.minimum(comparison.p_value * len(comparison), 1.0)
    comparison["permutation_p_10000"] = [
        _permutation_pvalue(results, c) for c in PROMPT_COLUMNS + ["mean_value_score"]
    ]
    comparison.to_csv(out / f"{model_name.lower()}_prompt_comparisons.csv", index=False)

    leave_one_out = []
    departments = results["department"].dropna().unique() if "department" in results else []
    for department in sorted(departments):
        subset = results[results.department != department]
        if subset.inferred_gender.nunique() < 2:
            continue
        row = _comparison(subset, "mean_value_score")
        row["excluded_department"] = department
        row["permutation_p_10000"] = _permutation_pvalue(subset, "mean_value_score")
        leave_one_out.append(row)
    pd.DataFrame(leave_one_out).to_csv(out / f"{model_name.lower()}_leave_one_department_out.csv", index=False)

    flow = pd.DataFrame(
        [
            ["image_complete_named_cohort", len(results)],
            ["male_inferred", int((results.inferred_gender == "male").sum())],
            ["female_inferred", int((results.inferred_gender == "female").sum())],
        ],
        columns=["stage", "n"],
    )
    flow.to_csv(out / f"{model_name.lower()}_evaluation_counts.csv", index=False)

    formula = "mean_value_score ~ C(inferred_gender) + C(medium_category) + C(century) + aspect_ratio"
    model = smf.ols(formula, data=results).fit(cov_type="HC3")
    no_sculpture = results[results.medium_category != "sculpture"]
    pooled = results.copy()
    pooled.loc[pooled.medium_category == "sculpture", "medium_category"] = "other"
    sensitivity = {
        "specification": ["primary", "exclude_sculpture", "pool_sculpture_with_other"],
        "n": [len(results), len(no_sculpture), len(pooled)],
        "male_inferred_coefficient": [
            model.params.get("C(inferred_gender)[T.male]", np.nan),
            smf.ols(formula, data=no_sculpture).fit(cov_type="HC3").params.get("C(inferred_gender)[T.male]", np.nan),
            smf.ols(formula, data=pooled).fit(cov_type="HC3").params.get("C(inferred_gender)[T.male]", np.nan),
        ],
        "male_inferred_p_value": [
            model.pvalues.get("C(inferred_gender)[T.male]", np.nan),
            smf.ols(formula, data=no_sculpture).fit(cov_type="HC3").pvalues.get("C(inferred_gender)[T.male]", np.nan),
            smf.ols(formula, data=pooled).fit(cov_type="HC3").pvalues.get("C(inferred_gender)[T.male]", np.nan),
        ],
        "r_squared": [model.rsquared, smf.ols(formula, data=no_sculpture).fit().rsquared, smf.ols(formula, data=pooled).fit().rsquared],
    }
    pd.DataFrame(sensitivity).to_csv(out / f"{model_name.lower()}_regression_sensitivity.csv", index=False)

    report = [
        f"{model_name} publication analysis",
        "=" * 72,
        "Interpretation: name-inferred creator category; not gender identity.",
        "Scope: image-complete named-attribution cohort only; no population or causal claim.",
        "",
        "Prompt comparisons (male minus female; primary p-values and Bonferroni p-values):",
        comparison.to_string(index=False),
        "",
        "Primary HC3 OLS model:",
        model.summary().as_text(),
        "",
        "Sensitivity models saved separately: exclude sculpture; pool sculpture with other.",
        "Permutation p-values use 10,000 label shuffles; leave-one-department-out results saved separately.",
    ]
    (out / f"{model_name.lower()}_publication_report.txt").write_text("\n".join(report))
