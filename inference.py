from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


RESULT_COLUMNS = [
    "test",
    "metric",
    "grouping",
    "statistic",
    "p_value",
    "effect_size",
    "n",
    "interpretation",
]

TRAJECTORY_RESULT_COLUMNS = [
    "progress_bin",
    "metric",
    "summary",
    "grouping",
    "test",
    "statistic",
    "p_value",
    "p_value_corrected",
    "effect_size",
    "n",
    "interpretation",
]


def empty_result(message: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test": "Not run",
                "metric": "",
                "grouping": "",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
                "n": 0,
                "interpretation": message,
            }
        ],
        columns=RESULT_COLUMNS,
    )


def empty_trajectory_result(message: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "progress_bin": "",
                "metric": "",
                "summary": "",
                "grouping": "",
                "test": "Not run",
                "statistic": np.nan,
                "p_value": np.nan,
                "p_value_corrected": np.nan,
                "effect_size": np.nan,
                "n": 0,
                "interpretation": message,
            }
        ],
        columns=TRAJECTORY_RESULT_COLUMNS,
    )


def numeric_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pair = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return pair.astype({x_col: "float64", y_col: "float64"})


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p is unavailable"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def interpretation(test: str, p_value: float) -> str:
    if pd.isna(p_value):
        return f"{test} could not estimate a p-value for this filtered sample."
    direction = "suggests evidence of a difference/association" if p_value < 0.05 else "does not show strong evidence of a difference/association"
    return f"{test} {direction} at alpha = 0.05 ({format_p_value(p_value)})."


def trajectory_interpretation(test: str, p_value: float) -> str:
    if pd.isna(p_value):
        return f"{test} could not estimate a p-value for this progress bin."
    direction = "shows evidence of a group difference" if p_value < 0.05 else "does not show strong evidence of a group difference"
    return f"{test} {direction} in this progress bin ({format_p_value(p_value)})."


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    corrected = pd.Series(np.nan, index=p_values.index, dtype="float64")
    valid = p_values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return corrected

    ordered = valid.sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1)
    adjusted = (ordered.to_numpy(dtype=float) * m) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    corrected.loc[ordered.index] = adjusted
    return corrected


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a_vals = a.dropna().to_numpy(dtype=float)
    b_vals = b.dropna().to_numpy(dtype=float)
    if len(a_vals) == 0 or len(b_vals) == 0:
        return float("nan")
    greater = sum(float((value > b_vals).sum()) for value in a_vals)
    less = sum(float((value < b_vals).sum()) for value in a_vals)
    return float((greater - less) / (len(a_vals) * len(b_vals)))


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    if n <= 0:
        return float("nan")
    r, k = table.shape
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(chi2 / denom))


def safe_exp(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(np.exp(np.clip(value, -709, 709)))


def chi_square_win_by_enemy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["win"].isin([0.0, 1.0])].copy()
    if work["enemy"].nunique() < 2:
        return empty_result("Select at least two enemies with win/loss episodes."), pd.DataFrame()
    if work["win"].nunique() < 2:
        return empty_result("The filtered sample needs both wins and losses."), pd.DataFrame()

    table = pd.crosstab(work["enemy"], work["win"].map({0.0: "loss", 1.0: "win"}))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return empty_result("The contingency table needs at least two enemies and two outcomes."), table

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    result = pd.DataFrame(
        [
            {
                "test": "Chi-square",
                "metric": "win",
                "grouping": "enemy",
                "statistic": chi2,
                "p_value": p_value,
                "effect_size": cramers_v(table, chi2),
                "n": int(table.to_numpy().sum()),
                "interpretation": interpretation("Chi-square", p_value),
            }
        ],
        columns=RESULT_COLUMNS,
    )
    return result, table.reset_index()


def compare_metric_by_outcome(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df[df["win"].isin([0.0, 1.0])][["win", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    wins = work.loc[work["win"] == 1.0, metric].astype(float)
    losses = work.loc[work["win"] == 0.0, metric].astype(float)
    if len(wins) < 2 or len(losses) < 2:
        return empty_result("Need at least two wins and two losses with this metric.")

    rows: list[dict[str, Any]] = []
    t_stat, t_p = stats.ttest_ind(wins, losses, equal_var=False, nan_policy="omit")
    rows.append(
        {
            "test": "Welch t-test",
            "metric": metric,
            "grouping": "win_vs_loss",
            "statistic": float(t_stat),
            "p_value": float(t_p),
            "effect_size": cohen_d(wins, losses),
            "n": int(len(work)),
            "interpretation": interpretation("Welch t-test", float(t_p)),
        }
    )

    u_stat, u_p = stats.mannwhitneyu(wins, losses, alternative="two-sided")
    rows.append(
        {
            "test": "Mann-Whitney U",
            "metric": metric,
            "grouping": "win_vs_loss",
            "statistic": float(u_stat),
            "p_value": float(u_p),
            "effect_size": cliffs_delta(wins, losses),
            "n": int(len(work)),
            "interpretation": interpretation("Mann-Whitney U", float(u_p)),
        }
    )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def compare_metric_by_enemy(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = df[["enemy", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    eligible_enemies = work.groupby("enemy").size()
    eligible_enemies = eligible_enemies[eligible_enemies >= 2].index
    work = work[work["enemy"].isin(eligible_enemies)].copy()
    groups = [group[metric].astype(float) for _, group in work.groupby("enemy")]
    if len(groups) < 2:
        return empty_result("Need at least two enemies with two or more observations for this metric.")
    if work[metric].nunique() < 2:
        return empty_result("Need metric variation across eligible enemy groups.")

    rows: list[dict[str, Any]] = []
    h_stat, h_p = stats.kruskal(*groups)
    rows.append(
        {
            "test": "Kruskal-Wallis",
            "metric": metric,
            "grouping": "enemy",
            "statistic": float(h_stat),
            "p_value": float(h_p),
            "effect_size": float((h_stat - len(groups) + 1) / max(len(work) - len(groups), 1)),
            "n": int(len(work)),
            "interpretation": interpretation("Kruskal-Wallis", float(h_p)),
        }
    )

    f_stat, f_p = stats.f_oneway(*groups)
    grand_mean = work[metric].astype(float).mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_total = ((work[metric].astype(float) - grand_mean) ** 2).sum()
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    rows.append(
        {
            "test": "One-way ANOVA",
            "metric": metric,
            "grouping": "enemy",
            "statistic": float(f_stat),
            "p_value": float(f_p),
            "effect_size": eta_squared,
            "n": int(len(work)),
            "interpretation": interpretation("One-way ANOVA", float(f_p)),
        }
    )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def correlation_tests(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if x_col == y_col:
        return empty_result("Select two different metrics for a correlation test.")

    pair = numeric_pair(df, x_col, y_col)
    if len(pair) < 3:
        return empty_result("Need at least three complete observations.")
    if pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        return empty_result("Both selected metrics need at least two distinct values.")

    pearson = stats.pearsonr(pair[x_col], pair[y_col])
    spearman = stats.spearmanr(pair[x_col], pair[y_col])
    rows = [
        {
            "test": "Pearson correlation",
            "metric": f"{x_col} vs {y_col}",
            "grouping": "",
            "statistic": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
            "effect_size": float(pearson.statistic),
            "n": int(len(pair)),
            "interpretation": interpretation("Pearson correlation", float(pearson.pvalue)),
        },
        {
            "test": "Spearman correlation",
            "metric": f"{x_col} vs {y_col}",
            "grouping": "",
            "statistic": float(spearman.statistic),
            "p_value": float(spearman.pvalue),
            "effect_size": float(spearman.statistic),
            "n": int(len(pair)),
            "interpretation": interpretation("Spearman correlation", float(spearman.pvalue)),
        },
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def logistic_regression_win(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    work = df[df["win"].isin([0.0, 1.0])][["win", "enemy", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 10:
        return empty_result("Need at least ten win/loss episodes for logistic regression.")
    if work["win"].nunique() < 2:
        return empty_result("Need both wins and losses for logistic regression.")
    if work[metric].nunique() < 2:
        return empty_result("The selected metric needs at least two distinct values.")

    captured_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.logit(f"win ~ Q('{metric}') + C(enemy)", data=work).fit(disp=False)
            captured_warnings = [str(warning.message) for warning in caught]
    except Exception as e:
        return empty_result(f"Logistic regression could not be fit: {e}")

    conf = model.conf_int()
    converged = bool(model.mle_retvals.get("converged", False))
    warning_suffix = ""
    if not converged:
        warning_suffix = " Model did not converge; interpret cautiously."
    elif captured_warnings:
        warning_suffix = f" Warning: {captured_warnings[0]}"

    rows: list[dict[str, Any]] = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        p_value = float(model.pvalues[term])
        odds_ratio = safe_exp(float(model.params[term]))
        ci_low = safe_exp(float(conf.loc[term, 0]))
        ci_high = safe_exp(float(conf.loc[term, 1]))
        rows.append(
            {
                "test": "Logistic regression",
                "metric": term,
                "grouping": "win ~ metric + enemy",
                "statistic": float(model.params[term]),
                "p_value": p_value,
                "effect_size": odds_ratio,
                "n": int(model.nobs),
                "interpretation": (
                    f"Odds ratio = {odds_ratio:.3g}; "
                    f"95% CI [{ci_low:.3g}, {ci_high:.3g}]; "
                    f"{format_p_value(p_value)}.{warning_suffix}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize_bin(group: pd.DataFrame, metric: str, summary: str) -> float:
    values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if values.empty:
        return float("nan")
    if summary == "mean":
        return float(values.mean())
    if summary == "max":
        return float(values.max())
    if summary == "final":
        ordered = group.sort_values("t") if "t" in group.columns else group
        return float(ordered[metric].replace([np.inf, -np.inf], np.nan).dropna().astype(float).iloc[-1])
    if summary == "sum":
        return float(values.sum())
    if summary == "AUC":
        ordered = group.sort_values("progress")
        y = ordered[metric].replace([np.inf, -np.inf], np.nan).astype(float)
        x = ordered["progress"].astype(float)
        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < 2:
            return float(values.mean())
        return float(np.trapezoid(valid.iloc[:, 1].to_numpy(), valid.iloc[:, 0].to_numpy()))
    raise ValueError(f"Unknown trajectory summary: {summary}")


def build_trajectory_summary(
    time_df: pd.DataFrame,
    metric: str,
    bins: int,
    summary: str,
) -> pd.DataFrame:
    required = {"episode_id", "enemy", "win", "t", "progress", metric}
    missing = sorted(required - set(time_df.columns))
    if missing:
        raise ValueError(f"timeseries is missing required columns: {', '.join(missing)}")

    work = time_df[["episode_id", "enemy", "win", "t", "progress", metric]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["episode_id", "enemy", "win", "progress", metric])
    work["progress"] = work["progress"].clip(0.0, 1.0)
    work["progress_bin_index"] = pd.cut(
        work["progress"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        labels=False,
    )
    work = work.dropna(subset=["progress_bin_index"])
    work["progress_bin_index"] = work["progress_bin_index"].astype(int)
    work["progress_bin"] = work["progress_bin_index"].map(
        lambda idx: f"{idx / bins:.0%}-{(idx + 1) / bins:.0%}"
    )
    work["outcome"] = work["win"].map({0.0: "Loss", 0.5: "Draw", 1.0: "Win"}).fillna(work["win"].astype(str))

    group_cols = ["episode_id", "enemy", "win", "outcome", "progress_bin_index", "progress_bin"]
    grouped = work.groupby(group_cols, sort=False, observed=True)
    if summary in {"mean", "max", "sum"}:
        agg_func = {"mean": "mean", "max": "max", "sum": "sum"}[summary]
        result = grouped[metric].agg(agg_func).reset_index(name="value")
    elif summary == "final":
        result = (
            work.sort_values(["episode_id", "progress_bin_index", "t"])
            .drop_duplicates(group_cols, keep="last")[group_cols + [metric]]
            .rename(columns={metric: "value"})
        )
    elif summary == "AUC":
        ordered = work.sort_values(group_cols + ["progress"]).copy()
        ordered_grouped = ordered.groupby(group_cols, sort=False, observed=True)
        ordered["_prev_progress"] = ordered_grouped["progress"].shift()
        ordered["_prev_value"] = ordered_grouped[metric].shift()
        ordered["_auc_piece"] = (
            (ordered["progress"] - ordered["_prev_progress"])
            * (ordered[metric].astype(float) + ordered["_prev_value"].astype(float))
            / 2.0
        )
        result = ordered.groupby(group_cols, sort=False, observed=True)["_auc_piece"].sum(min_count=1).reset_index(name="value")
        single_point = result["value"].isna()
        if single_point.any():
            fallback = grouped[metric].mean().reset_index(name="_fallback")
            result = result.merge(fallback, on=group_cols, how="left")
            result.loc[single_point, "value"] = result.loc[single_point, "_fallback"]
            result = result.drop(columns=["_fallback"])
    else:
        raise ValueError(f"Unknown trajectory summary: {summary}")

    result["metric"] = metric
    result["summary"] = summary
    return result.dropna(subset=["value"])


def run_trajectory_inference(
    trajectory_df: pd.DataFrame,
    metric: str,
    summary: str,
    grouping: str,
    apply_fdr: bool = True,
) -> pd.DataFrame:
    if trajectory_df.empty:
        return empty_trajectory_result("No episode-bin summaries are available for this metric.")

    rows: list[dict[str, Any]] = []
    for bin_index, bin_df in trajectory_df.groupby("progress_bin_index", sort=True):
        bin_label = str(bin_df["progress_bin"].iloc[0])
        if grouping == "Outcome":
            work = bin_df[bin_df["win"].isin([0.0, 1.0])]
            wins = work.loc[work["win"] == 1.0, "value"].astype(float)
            losses = work.loc[work["win"] == 0.0, "value"].astype(float)
            if len(wins) < 2 or len(losses) < 2:
                rows.append(
                    {
                        "progress_bin": bin_label,
                        "metric": metric,
                        "summary": summary,
                        "grouping": "Outcome",
                        "test": "Not run",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "p_value_corrected": np.nan,
                        "effect_size": np.nan,
                        "n": int(len(work)),
                        "interpretation": "Need at least two wins and two losses in this progress bin.",
                    }
                )
                continue

            if pd.concat([wins, losses]).nunique() < 2:
                t_stat, t_p, d_value = np.nan, np.nan, np.nan
                t_text = "Welch t-test could not run because this progress bin has no metric variation."
            else:
                t_stat, t_p = stats.ttest_ind(wins, losses, equal_var=False, nan_policy="omit")
                d_value = cohen_d(wins, losses)
                t_text = trajectory_interpretation("Welch t-test", float(t_p))
            rows.append(
                {
                    "progress_bin": bin_label,
                    "metric": metric,
                    "summary": summary,
                    "grouping": "Outcome",
                    "test": "Welch t-test",
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "p_value_corrected": np.nan,
                    "effect_size": d_value,
                    "n": int(len(work)),
                    "interpretation": t_text,
                }
            )
            u_stat, u_p = stats.mannwhitneyu(wins, losses, alternative="two-sided")
            rows.append(
                {
                    "progress_bin": bin_label,
                    "metric": metric,
                    "summary": summary,
                    "grouping": "Outcome",
                    "test": "Mann-Whitney U",
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "p_value_corrected": np.nan,
                    "effect_size": cliffs_delta(wins, losses),
                    "n": int(len(work)),
                    "interpretation": trajectory_interpretation("Mann-Whitney U", float(u_p)),
                }
            )
        else:
            groups = [group["value"].astype(float) for _, group in bin_df.groupby("enemy") if len(group) >= 2]
            if len(groups) < 2:
                rows.append(
                    {
                        "progress_bin": bin_label,
                        "metric": metric,
                        "summary": summary,
                        "grouping": "Enemy",
                        "test": "Not run",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "p_value_corrected": np.nan,
                        "effect_size": np.nan,
                        "n": int(len(bin_df)),
                        "interpretation": "Need at least two enemies with two or more episodes in this progress bin.",
                    }
                )
                continue

            eligible_values = pd.concat(groups, ignore_index=True)
            eligible_n = int(len(eligible_values))
            if eligible_values.nunique() < 2:
                h_stat, h_p, h_effect = np.nan, np.nan, np.nan
                h_text = "Kruskal-Wallis could not run because this progress bin has no metric variation."
            else:
                h_stat, h_p = stats.kruskal(*groups)
                h_effect = float((h_stat - len(groups) + 1) / max(eligible_n - len(groups), 1))
                h_text = trajectory_interpretation("Kruskal-Wallis", float(h_p))
            rows.append(
                {
                    "progress_bin": bin_label,
                    "metric": metric,
                    "summary": summary,
                    "grouping": "Enemy",
                    "test": "Kruskal-Wallis",
                    "statistic": float(h_stat),
                    "p_value": float(h_p),
                    "p_value_corrected": np.nan,
                    "effect_size": h_effect,
                    "n": eligible_n,
                    "interpretation": h_text,
                }
            )

    result = pd.DataFrame(rows, columns=TRAJECTORY_RESULT_COLUMNS)
    if apply_fdr and not result.empty:
        for test_name, test_rows in result.groupby("test"):
            if test_name == "Not run":
                continue
            result.loc[test_rows.index, "p_value_corrected"] = benjamini_hochberg(test_rows["p_value"])
        result["interpretation"] = result.apply(
            lambda row: trajectory_interpretation(str(row["test"]), float(row["p_value_corrected"]))
            if row["test"] != "Not run" and not pd.isna(row["p_value_corrected"])
            else row["interpretation"],
            axis=1,
        )
    else:
        result["p_value_corrected"] = result["p_value"]
    return result
