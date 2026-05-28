from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inference import (
    build_trajectory_summary,
    chi_square_win_by_enemy,
    compare_metric_by_enemy,
    compare_metric_by_outcome,
    correlation_tests,
    logistic_regression_win,
    run_trajectory_inference,
)


DEFAULT_DATA_DIR = Path("stats_viz")

METRIC_LABELS = {
    "resources_gathered": "Resources gathered",
    "resources_returned": "Resources returned",
    "resources_spent": "Resources spent",
    "damage_dealt": "Damage dealt",
    "damage_taken": "Damage taken",
    "units_produced": "Units produced",
    "units_lost": "Units lost",
    "units_killed": "Units killed",
    "resource_bank": "Resource bank",
    "resources_carried": "Resources carried",
    "army_value": "Army value",
    "economy_value": "Economy value",
    "infrastructure_value": "Infrastructure value",
    "total_unit_value": "Total unit value",
    "unit_value_advantage": "Unit value advantage",
}

TRAJECTORY_SUMMARIES = ["mean", "max", "final", "sum", "AUC"]

EPISODE_SCATTER_COLUMNS = {
    "resources_gathered_total": "Resources gathered",
    "resources_returned_total": "Resources returned",
    "resource_return_success_rate": "Resource return success rate",
    "resources_spent_total": "Resources spent",
    "units_produced_total": "Units produced",
    "units_lost_total": "Units lost",
    "units_killed_total": "Units killed",
    "damage_dealt_total": "Damage dealt",
    "damage_taken_total": "Damage taken",
    "value_produced_total": "Value produced",
    "value_lost_total": "Value lost",
    "value_killed_total": "Value killed",
    "max_army_value": "Max army value",
    "max_total_unit_value": "Max total unit value",
    "final_total_unit_value": "Final total unit value",
    "mean_unit_value_advantage": "Mean unit value advantage",
    "final_unit_value_advantage": "Final unit value advantage",
    "duration": "Duration",
    "average_distance_to_resource": "Average distance to resource",
    "shortest_distance_to_resource": "Shortest distance to resource",
    "distance_to_enemy_base": "Distance to enemy base",
    "worker_produced_total": "Workers produced",
    "worker_lost_total": "Workers lost",
    "worker_killed_total": "Workers killed",
    "max_worker": "Max workers",
    "final_worker": "Final workers",
    "action_harvest_total": "Harvest actions",
    "action_return_total": "Return actions",
    "action_produce_total": "Produce actions",
    "action_attack_total": "Attack actions",
    "win": "Win value",
}

KEY_STATS_EXCLUDED_COLUMNS = {
    "win",
    "result",
    "map_width",
    "map_height",
}

KEY_STATS_EXCLUDED_PREFIXES = (
    "base_",
    "barracks_",
    "light_",
    "heavy_",
    "ranged_",
    "max_base",
    "max_barracks",
    "max_light",
    "max_heavy",
    "max_ranged",
    "final_base",
    "final_barracks",
    "final_light",
    "final_heavy",
    "final_ranged",
)

OUTCOME_ORDER = ["Win", "Draw", "Loss"]

STAT_METHOD_SECTIONS = [
    {
        "title": "Data Summaries",
        "methods": [
            {
                "name": "Counts, means, medians, standard deviations, and win rates",
                "used_for": "Overview cards, enemy summaries, supporting tables, distribution plots, and binned charts.",
                "explanation": (
                    "Counts show sample size. Means summarize the arithmetic average, medians summarize the middle observed value, "
                    "and standard deviations summarize spread around the mean. Win rate is calculated as the mean of the encoded "
                    "win value, where wins are 1.0, draws are 0.5, and losses are 0.0."
                ),
                "resources": [
                    ("pandas descriptive statistics", "https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics"),
                ],
            },
            {
                "name": "Standard error bands and approximate confidence bands",
                "used_for": "Time-series ribbons, correlation line plots, and trajectory charts.",
                "explanation": (
                    "The dashboard computes standard error as standard deviation divided by the square root of n. Time-series plots "
                    "show mean +/- one standard error, while binned correlation and trajectory plots use mean +/- 1.96 standard errors "
                    "as an approximate 95% confidence band."
                ),
                "resources": [
                    ("NIST standard error overview", "https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm"),
                ],
            },
            {
                "name": "Normalized progress bins",
                "used_for": "Normalized time-series plots and trajectory inference.",
                "explanation": (
                    "Episodes with different durations are mapped onto progress from 0 to 1, then divided into equal-width bins. "
                    "This lets the dashboard compare early, middle, and late game behavior without treating longer episodes as "
                    "having more independent observations."
                ),
                "resources": [
                    ("pandas cut", "https://pandas.pydata.org/docs/reference/api/pandas.cut.html"),
                ],
            },
            {
                "name": "Trajectory summaries: mean, max, final, sum, and AUC",
                "used_for": "The Trajectory mode in the Inference tab.",
                "explanation": (
                    "Each episode is summarized inside every normalized progress bin before tests are run. Mean, max, final, and sum "
                    "use the corresponding within-bin value. AUC uses trapezoidal integration over progress, which captures both the "
                    "level of a metric and how long it stays high or low."
                ),
                "resources": [
                    ("NumPy trapezoid integration", "https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html"),
                ],
            },
        ],
    },
    {
        "title": "Associations And Model Fits",
        "methods": [
            {
                "name": "Pearson correlation",
                "used_for": "Correlation plots, selected correlation tables, correlation matrices, and the Correlation test.",
                "explanation": (
                    "Pearson correlation measures the strength and direction of a linear association between two numeric variables. "
                    "The coefficient ranges from -1 to 1, and the p-value tests whether the population correlation is zero."
                ),
                "resources": [
                    ("SciPy pearsonr", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html"),
                ],
            },
            {
                "name": "Spearman rank correlation",
                "used_for": "The Correlation test in the Inference tab.",
                "explanation": (
                    "Spearman correlation converts values to ranks before estimating association. It is useful when a relationship is "
                    "monotonic but not necessarily linear, or when outliers make Pearson correlation too sensitive."
                ),
                "resources": [
                    ("SciPy spearmanr", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"),
                ],
            },
            {
                "name": "Least-squares regression lines",
                "used_for": "Dashed trend lines on scatter plots.",
                "explanation": (
                    "Regression lines are visual guides fit separately by enemy using a first-degree polynomial. They show the estimated "
                    "linear trend in the plotted points, but the dashboard does not use these lines as formal hypothesis tests."
                ),
                "resources": [
                    ("NumPy polyfit", "https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html"),
                ],
            },
            {
                "name": "Logistic regression",
                "used_for": "The Logistic regression test for binary win/loss outcomes.",
                "explanation": (
                    "Logistic regression models the log odds of winning from a selected metric while adjusting for enemy. The dashboard "
                    "reports coefficients as odds ratios with 95% confidence intervals. Draws are excluded because the model is binary."
                ),
                "resources": [
                    ("statsmodels Logit", "https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html"),
                    ("statsmodels formula API", "https://www.statsmodels.org/stable/example_formulas.html"),
                ],
            },
        ],
    },
    {
        "title": "Group Difference Tests",
        "methods": [
            {
                "name": "Chi-square test of independence",
                "used_for": "Win rate by enemy.",
                "explanation": (
                    "The chi-square test compares a contingency table of enemies by win/loss outcome. It asks whether outcome frequencies "
                    "differ more across enemies than expected if enemy and outcome were independent. Draws are excluded from this test."
                ),
                "resources": [
                    ("SciPy chi2_contingency", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html"),
                ],
            },
            {
                "name": "Welch t-test",
                "used_for": "Metric by outcome and trajectory outcome comparisons.",
                "explanation": (
                    "Welch's t-test compares the means of two independent groups without assuming equal variances. Here it compares wins "
                    "against losses, either at the episode level or within each trajectory progress bin."
                ),
                "resources": [
                    ("SciPy ttest_ind", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html"),
                ],
            },
            {
                "name": "Mann-Whitney U test",
                "used_for": "Metric by outcome and trajectory outcome comparisons.",
                "explanation": (
                    "The Mann-Whitney U test is a rank-based comparison of two independent groups. It is included alongside Welch's t-test "
                    "because it is less tied to normal-distribution assumptions."
                ),
                "resources": [
                    ("SciPy mannwhitneyu", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html"),
                ],
            },
            {
                "name": "Kruskal-Wallis test",
                "used_for": "Metric by enemy and trajectory enemy comparisons.",
                "explanation": (
                    "Kruskal-Wallis is a rank-based test for comparing more than two independent groups. The dashboard uses it to test "
                    "whether metric distributions differ across enemies."
                ),
                "resources": [
                    ("SciPy kruskal", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html"),
                ],
            },
            {
                "name": "One-way ANOVA",
                "used_for": "Metric by enemy.",
                "explanation": (
                    "One-way ANOVA compares group means across enemies by splitting variation into between-group and within-group parts. "
                    "It complements Kruskal-Wallis when mean differences are the main question."
                ),
                "resources": [
                    ("SciPy f_oneway", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html"),
                ],
            },
        ],
    },
    {
        "title": "Effect Sizes And Multiple Testing",
        "methods": [
            {
                "name": "Cramer's V",
                "used_for": "Effect size for the chi-square win-rate test.",
                "explanation": (
                    "Cramer's V rescales the chi-square statistic to a 0-to-1 association measure for contingency tables. Values closer "
                    "to 1 indicate a stronger relationship between enemy and win/loss outcome."
                ),
                "resources": [
                    ("Real Statistics Cramer's V overview", "https://real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/"),
                ],
            },
            {
                "name": "Cohen's d",
                "used_for": "Effect size for Welch t-tests.",
                "explanation": (
                    "Cohen's d expresses the win-versus-loss mean difference in pooled standard-deviation units. Positive values mean "
                    "the metric is higher in wins; negative values mean it is higher in losses."
                ),
                "resources": [
                    ("Effect size overview", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/"),
                ],
            },
            {
                "name": "Cliff's delta",
                "used_for": "Effect size for Mann-Whitney U tests.",
                "explanation": (
                    "Cliff's delta estimates how often values from one group exceed values from the other, minus the reverse. It ranges "
                    "from -1 to 1 and keeps the same win-minus-loss direction used elsewhere in the dashboard."
                ),
                "resources": [
                    ("Cliff's delta paper", "https://doi.org/10.3102/10769986021002101"),
                ],
            },
            {
                "name": "Eta squared and epsilon-squared-like effects",
                "used_for": "Effect sizes for one-way ANOVA and Kruskal-Wallis.",
                "explanation": (
                    "Eta squared reports the share of metric variation explained by enemy groups for ANOVA. The Kruskal-Wallis effect "
                    "uses an epsilon-squared-like calculation from the H statistic, giving a similar sense of explained group difference."
                ),
                "resources": [
                    ("Effect size overview", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444174/"),
                ],
            },
            {
                "name": "Benjamini-Hochberg false discovery rate correction",
                "used_for": "Corrected p-values in trajectory inference.",
                "explanation": (
                    "Trajectory mode can run the same test across many progress bins. Benjamini-Hochberg adjustment controls the expected "
                    "false discovery rate across those repeated tests, reducing the chance of over-reading isolated small p-values."
                ),
                "resources": [
                    ("NCES Benjamini-Hochberg procedure", "https://nces.ed.gov/nationsreportcard/tdw/analysis/2000_2001/infer_multiplecompare_fdr.aspx"),
                    ("Original paper DOI", "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"),
                ],
            },
            {
                "name": "P-values and alpha = 0.05",
                "used_for": "Interpretation text throughout the Inference tab.",
                "explanation": (
                    "The dashboard treats p < 0.05 as evidence of a difference or association for the filtered sample. This threshold "
                    "is a convention, not proof of practical importance, so the effect size and sample size should be read alongside it."
                ),
                "resources": [
                    ("American Statistical Association statement on p-values", "https://doi.org/10.1080/00031305.2016.1154108"),
                ],
            },
        ],
    },
]


st.set_page_config(
    page_title="OCGDT Stats Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clean_label(value: str) -> str:
    return value.replace("_", " ").title()


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def pearsonr_safe(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))


def spearmanr_safe(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def metric_label(column: str) -> str:
    return EPISODE_SCATTER_COLUMNS.get(column, METRIC_LABELS.get(column, clean_label(column)))


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        outcome=np.select(
            [df["win"] == 1.0, df["win"] == 0.5, df["win"] == 0.0],
            ["Win", "Draw", "Loss"],
            default=df["win"].astype(str),
        )
    )


def numeric_episode_columns(episode_df: pd.DataFrame) -> list[str]:
    numeric_cols = episode_df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        col
        for col in numeric_cols
        if col not in KEY_STATS_EXCLUDED_COLUMNS
        and not col.startswith(KEY_STATS_EXCLUDED_PREFIXES)
        and not col.startswith("time_to_")
        and episode_df[col].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1
    ]


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a_values = a.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    b_values = b.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if a_values.empty or b_values.empty:
        return float("nan")

    combined = pd.concat([a_values, b_values], ignore_index=True)
    ranks = combined.rank(method="average")
    rank_sum_a = ranks.iloc[: len(a_values)].sum()
    u_stat = rank_sum_a - (len(a_values) * (len(a_values) + 1) / 2)
    return float((2 * u_stat / (len(a_values) * len(b_values))) - 1)


def build_outcome_associations(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcome_df = add_outcome_label(episode_df)
    outcome_means = outcome_df.groupby("outcome", observed=True)

    for col in numeric_episode_columns(episode_df):
        pair = episode_df[[col, "win"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2 or pair["win"].nunique() < 2:
            continue
        pearson = pearsonr_safe(pair[col], pair["win"])
        spearman = spearmanr_safe(pair[col], pair["win"])
        rows.append(
            {
                "metric": col,
                "metric_label": metric_label(col),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "abs_pearson_r": abs(pearson) if not pd.isna(pearson) else np.nan,
                "win_mean": outcome_means[col].mean().get("Win", np.nan),
                "draw_mean": outcome_means[col].mean().get("Draw", np.nan),
                "loss_mean": outcome_means[col].mean().get("Loss", np.nan),
                "n": int(len(pair)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)


def build_win_loss_separations(episode_df: pd.DataFrame) -> pd.DataFrame:
    work = episode_df[episode_df["win"].isin([0.0, 1.0])].copy()
    if work["win"].nunique() < 2:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for col in numeric_episode_columns(work):
        wins = work.loc[work["win"] == 1.0, col]
        losses = work.loc[work["win"] == 0.0, col]
        if wins.dropna().empty or losses.dropna().empty:
            continue
        delta = cliffs_delta(wins, losses)
        rows.append(
            {
                "metric": col,
                "metric_label": metric_label(col),
                "cliffs_delta": delta,
                "abs_cliffs_delta": abs(delta) if not pd.isna(delta) else np.nan,
                "win_mean": wins.mean(),
                "loss_mean": losses.mean(),
                "mean_difference": wins.mean() - losses.mean(),
                "win_median": wins.median(),
                "loss_median": losses.median(),
                "n": int(wins.count() + losses.count()),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_cliffs_delta", ascending=False)


def build_metric_correlations(episode_df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_episode_columns(episode_df)
    if len(cols) < 2:
        return pd.DataFrame()

    corr = episode_df[cols].replace([np.inf, -np.inf], np.nan).corr(method="pearson")
    rows: list[dict[str, Any]] = []
    for i, left_col in enumerate(corr.columns):
        for right_col in corr.columns[i + 1 :]:
            pearson = corr.loc[left_col, right_col]
            if pd.isna(pearson):
                continue
            rows.append(
                {
                    "metric_a": left_col,
                    "metric_a_label": metric_label(left_col),
                    "metric_b": right_col,
                    "metric_b_label": metric_label(right_col),
                    "pearson_r": pearson,
                    "abs_pearson_r": abs(pearson),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)


def share_where(episode_df: pd.DataFrame, condition: pd.Series) -> float:
    if episode_df.empty:
        return float("nan")
    return float(condition.fillna(False).mean())


def add_regression_lines(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for enemy, group in df.groupby("enemy", sort=True):
        pair = group[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2 or pair[x_col].nunique() < 2:
            continue
        x_vals = pair[x_col].to_numpy(dtype=float)
        y_vals = pair[y_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=slope * xs + intercept,
                mode="lines",
                name=f"{enemy} regression",
                line={"dash": "dash", "width": 2},
                hovertemplate=f"{enemy}<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra>Regression</extra>",
            )
        )


def read_table(data_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {data_dir}")


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def add_derived_episode_metrics(episode_df: pd.DataFrame) -> pd.DataFrame:
    df = episode_df.copy()
    required = {"resources_returned_total", "resources_gathered_total"}
    if "resource_return_success_rate" not in df.columns and required.issubset(df.columns):
        gathered = df["resources_gathered_total"].replace(0, np.nan)
        df["resource_return_success_rate"] = df["resources_returned_total"] / gathered
    return df


@st.cache_data(show_spinner=False)
def load_episode_tables(data_dir_text: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir_text).expanduser()
    episode_df = add_derived_episode_metrics(read_table(data_dir, "episode_summary"))
    return {
        "episode": episode_df,
        "summary_by_enemy": read_optional_csv(data_dir / "summary_by_enemy.csv"),
        "selected_correlations": read_optional_csv(data_dir / "selected_correlations_by_enemy.csv"),
        "correlation_matrix": read_optional_csv(data_dir / "overall" / "correlation_matrix.csv"),
    }


@st.cache_data(show_spinner="Loading time series data...")
def load_timeseries_table(data_dir_text: str) -> pd.DataFrame:
    data_dir = Path(data_dir_text).expanduser()
    return read_table(data_dir, "timeseries")


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.error(f"{table_name} is missing required columns: {', '.join(missing)}")
        st.stop()


def apply_episode_filters(
    episode_df: pd.DataFrame,
    enemies: list[str],
    outcome: str,
    duration_range: tuple[int, int],
) -> pd.DataFrame:
    filtered = episode_df.copy()
    if enemies:
        filtered = filtered[filtered["enemy"].astype(str).isin(enemies)]
    if outcome == "Wins":
        filtered = filtered[filtered["win"] == 1.0]
    elif outcome == "Losses":
        filtered = filtered[filtered["win"] == 0.0]
    elif outcome == "Draws":
        filtered = filtered[filtered["win"] == 0.5]
    filtered = filtered[
        (filtered["duration"] >= duration_range[0])
        & (filtered["duration"] <= duration_range[1])
    ]
    return filtered


def aggregate_time_series(
    time_df: pd.DataFrame,
    metric: str,
    x_mode: str,
    cumulative: bool,
    bins: int,
) -> pd.DataFrame:
    df = time_df[["episode_id", "enemy", "t", "progress", metric]].copy()
    y_col = metric
    if cumulative:
        y_col = f"cumulative_{metric}"
        df[y_col] = df.groupby("episode_id", sort=False)[metric].cumsum()

    if x_mode == "Normalized progress":
        df["x"] = pd.cut(
            df["progress"].clip(0.0, 1.0),
            bins=np.linspace(0.0, 1.0, bins + 1),
            include_lowest=True,
            labels=False,
        )
        grouped = (
            df.groupby(["enemy", "x"], as_index=False)
            .agg(
                x_value=("progress", "mean"),
                mean=(y_col, "mean"),
                median=(y_col, "median"),
                std=(y_col, "std"),
                n=(y_col, "count"),
            )
            .dropna(subset=["x_value"])
        )
    else:
        grouped = (
            df.groupby(["enemy", "t"], as_index=False)
            .agg(
                x_value=("t", "mean"),
                mean=(y_col, "mean"),
                median=(y_col, "median"),
                std=(y_col, "std"),
                n=(y_col, "count"),
            )
        )

    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["n"].clip(lower=1))
    grouped["lower"] = grouped["mean"] - grouped["sem"]
    grouped["upper"] = grouped["mean"] + grouped["sem"]
    return grouped.sort_values(["enemy", "x_value"])


def plot_time_series(agg: pd.DataFrame, metric_label: str, x_mode: str) -> go.Figure:
    fig = go.Figure()
    x_title = "Normalized game progress" if x_mode == "Normalized progress" else "Timestep"
    for enemy, group in agg.groupby("enemy", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["x_value"],
                y=group["upper"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["x_value"],
                y=group["lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(72, 117, 168, 0.16)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["x_value"],
                y=group["mean"],
                mode="lines",
                name=str(enemy),
                hovertemplate=f"{x_title}: %{{x:.2f}}<br>{metric_label}: %{{y:.2f}}<extra>{enemy}</extra>",
            )
        )
    fig.update_layout(
        height=480,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        xaxis_title=x_title,
        yaxis_title=metric_label,
        legend_title_text="Enemy",
    )
    return fig


def render_metric_row(episode_df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Episodes", f"{len(episode_df):,}")
    col2.metric("Win rate", format_number(episode_df["win"].mean()))
    col3.metric("Mean duration", format_number(episode_df["duration"].mean()))
    col4.metric("Mean gathered", format_number(episode_df["resources_gathered_total"].mean()))
    if "resource_return_success_rate" in episode_df.columns:
        col5.metric("Return success", format_percent(episode_df["resource_return_success_rate"].mean()))
    else:
        col5.metric("Mean killed", format_number(episode_df["units_killed_total"].mean()))


def render_overview(episode_df: pd.DataFrame, summary_by_enemy: pd.DataFrame) -> None:
    render_metric_row(episode_df)

    left, right = st.columns([1.1, 1])
    with left:
        enemy_summary = (
            episode_df.groupby("enemy", as_index=False)
            .agg(
                episodes=("episode_id", "count"),
                win_rate=("win", "mean"),
                mean_duration=("duration", "mean"),
                resources_gathered=("resources_gathered_total", "mean"),
                return_success=("resource_return_success_rate", "mean"),
                units_killed=("units_killed_total", "mean"),
            )
            .sort_values("win_rate", ascending=False)
        )
        fig = px.bar(
            enemy_summary,
            x="enemy",
            y="win_rate",
            text="episodes",
            hover_data=["mean_duration", "resources_gathered", "return_success", "units_killed"],
            labels={"enemy": "Enemy", "win_rate": "Win rate"},
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=380, margin={"l": 8, "r": 8, "t": 28, "b": 8})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        outcomes = episode_df.assign(
            outcome=np.select(
                [episode_df["win"] == 1.0, episode_df["win"] == 0.5],
                ["Win", "Draw"],
                default="Loss",
            )
        )
        fig = px.histogram(
            outcomes,
            x="duration",
            color="outcome",
            nbins=30,
            labels={"duration": "Duration", "count": "Episodes"},
            barmode="overlay",
            opacity=0.72,
        )
        fig.update_layout(height=380, margin={"l": 8, "r": 8, "t": 28, "b": 8})
        st.plotly_chart(fig, use_container_width=True)

    if not summary_by_enemy.empty:
        st.dataframe(summary_by_enemy, use_container_width=True, hide_index=True)


def render_key_statistics(episode_df: pd.DataFrame) -> None:
    associations = build_outcome_associations(episode_df)
    separations = build_win_loss_separations(episode_df)
    metric_correlations = build_metric_correlations(episode_df)

    outcome_counts = episode_df["win"].value_counts()
    worker_only_share = (
        share_where(episode_df, episode_df["units_produced_total"] == episode_df["worker_produced_total"])
        if {"units_produced_total", "worker_produced_total"}.issubset(episode_df.columns)
        else float("nan")
    )
    combat_unit_cols = [col for col in ["light_produced_total", "heavy_produced_total", "ranged_produced_total"] if col in episode_df.columns]
    combat_unit_share = (
        share_where(episode_df, episode_df[combat_unit_cols].sum(axis=1) > 0)
        if combat_unit_cols
        else float("nan")
    )
    distant_foraging_share = (
        share_where(episode_df, episode_df["average_distance_to_resource"] > 4)
        if "average_distance_to_resource" in episode_df.columns
        else float("nan")
    )
    non_adjacent_resource_share = (
        share_where(episode_df, episode_df["shortest_distance_to_resource"] > 1)
        if "shortest_distance_to_resource" in episode_df.columns
        else float("nan")
    )

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Episodes", f"{len(episode_df):,}")
    kpi_cols[1].metric("Win score", format_number(episode_df["win"].mean()))
    kpi_cols[2].metric(
        "W / D / L",
        f"{int(outcome_counts.get(1.0, 0)):,} / {int(outcome_counts.get(0.5, 0)):,} / {int(outcome_counts.get(0.0, 0)):,}",
    )
    kpi_cols[3].metric("Worker-only production", format_percent(worker_only_share))
    kpi_cols[4].metric("Avg resource distance > 4", format_percent(distant_foraging_share))

    st.subheader("Strongest Associations With Outcome")
    if associations.empty:
        st.info("No outcome associations are available for the current filters.")
    else:
        top_associations = associations.head(12).copy()
        fig = px.bar(
            top_associations.sort_values("abs_pearson_r"),
            x="pearson_r",
            y="metric_label",
            orientation="h",
            color="pearson_r",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            labels={"pearson_r": "Pearson r with win score", "metric_label": "Metric"},
        )
        fig.update_layout(height=430, margin={"l": 8, "r": 8, "t": 28, "b": 8})
        st.plotly_chart(fig, use_container_width=True)

        association_display = associations[
            ["metric_label", "pearson_r", "spearman_rho", "win_mean", "draw_mean", "loss_mean", "n"]
        ].head(20)
        st.dataframe(
            association_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric_label": "Metric",
                "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
                "spearman_rho": st.column_config.NumberColumn("Spearman rho", format="%.3f"),
                "win_mean": st.column_config.NumberColumn("Win mean", format="%.3f"),
                "draw_mean": st.column_config.NumberColumn("Draw mean", format="%.3f"),
                "loss_mean": st.column_config.NumberColumn("Loss mean", format="%.3f"),
                "n": st.column_config.NumberColumn("n", format="%d"),
            },
        )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Largest Win/Loss Separations")
        if separations.empty:
            st.info("Select filters with at least one win and one loss to estimate win/loss separation.")
        else:
            st.dataframe(
                separations[
                    [
                        "metric_label",
                        "cliffs_delta",
                        "win_mean",
                        "loss_mean",
                        "mean_difference",
                        "win_median",
                        "loss_median",
                        "n",
                    ]
                ].head(15),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "metric_label": "Metric",
                    "cliffs_delta": st.column_config.NumberColumn("Cliff's delta", format="%.3f"),
                    "win_mean": st.column_config.NumberColumn("Win mean", format="%.3f"),
                    "loss_mean": st.column_config.NumberColumn("Loss mean", format="%.3f"),
                    "mean_difference": st.column_config.NumberColumn("Mean diff.", format="%.3f"),
                    "win_median": st.column_config.NumberColumn("Win median", format="%.3f"),
                    "loss_median": st.column_config.NumberColumn("Loss median", format="%.3f"),
                    "n": st.column_config.NumberColumn("n", format="%d"),
                },
            )

    with right:
        st.subheader("Strong Metric-Metric Correlations")
        if metric_correlations.empty:
            st.info("No metric correlations are available for the current filters.")
        else:
            st.dataframe(
                metric_correlations[["metric_a_label", "metric_b_label", "pearson_r"]].head(15),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "metric_a_label": "Metric A",
                    "metric_b_label": "Metric B",
                    "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
                },
            )

    st.subheader("Learned Behavior Evidence")
    evidence_cols = st.columns(3)
    evidence_cols[0].metric("Combat-unit production", format_percent(combat_unit_share))
    evidence_cols[1].metric("Nearest resource distance > 1", format_percent(non_adjacent_resource_share))
    return_success = (
        episode_df["resource_return_success_rate"].mean()
        if "resource_return_success_rate" in episode_df.columns
        else float("nan")
    )
    value_killed_r = (
        pearsonr_safe(episode_df["value_killed_total"], episode_df["win"])
        if {"value_killed_total", "win"}.issubset(episode_df.columns)
        else float("nan")
    )
    evidence_cols[2].metric("Resource return success", format_percent(return_success))
    st.caption(f"Value killed vs win score: r = {format_number(value_killed_r)}")

    outcome_df = add_outcome_label(episode_df)
    claim_metrics = [
        col
        for col in [
            "worker_produced_total",
            "max_worker",
            "final_worker",
            "resources_gathered_total",
            "resource_return_success_rate",
            "value_killed_total",
        ]
        if col in outcome_df.columns
    ]
    if claim_metrics:
        grouped = (
            outcome_df.groupby("outcome", as_index=False, observed=True)[claim_metrics]
            .mean()
            .melt(id_vars="outcome", var_name="metric", value_name="mean_value")
        )
        grouped["metric_label"] = grouped["metric"].map(metric_label)
        grouped["outcome"] = pd.Categorical(grouped["outcome"], categories=OUTCOME_ORDER, ordered=True)
        grouped = grouped.sort_values(["outcome", "metric_label"])
        fig = px.bar(
            grouped,
            x="metric_label",
            y="mean_value",
            color="outcome",
            barmode="group",
            category_orders={"outcome": OUTCOME_ORDER},
            labels={"metric_label": "Metric", "mean_value": "Mean value", "outcome": "Outcome"},
        )
        fig.update_layout(height=410, margin={"l": 8, "r": 8, "t": 28, "b": 8}, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    detail_left, detail_right = st.columns([1, 1])
    with detail_left:
        st.markdown("**Resource Foraging Distance Bins**")
        required_distance_cols = {
            "average_distance_to_resource",
            "resources_gathered_total",
            "resources_returned_total",
            "resource_return_success_rate",
            "units_produced_total",
        }
        if required_distance_cols.issubset(episode_df.columns):
            distance_df = episode_df.copy()
            distance_df["distance_bin"] = np.where(distance_df["average_distance_to_resource"] > 4, ">4", "<=4")
            distance_summary = (
                distance_df.groupby("distance_bin", as_index=False)
                .agg(
                    episodes=("episode_id", "count"),
                    win_score=("win", "mean"),
                    resources_gathered=("resources_gathered_total", "mean"),
                    resources_returned=("resources_returned_total", "mean"),
                    return_success_rate=("resource_return_success_rate", "mean"),
                    units_produced=("units_produced_total", "mean"),
                )
                .sort_values("distance_bin")
            )
            st.dataframe(
                distance_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "distance_bin": "Average distance",
                    "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                    "win_score": st.column_config.NumberColumn("Win score", format="%.3f"),
                    "resources_gathered": st.column_config.NumberColumn("Resources gathered", format="%.3f"),
                    "resources_returned": st.column_config.NumberColumn("Resources returned", format="%.3f"),
                    "return_success_rate": st.column_config.NumberColumn("Return success", format="%.3f"),
                    "units_produced": st.column_config.NumberColumn("Units produced", format="%.3f"),
                },
            )
        else:
            st.info("Resource distance columns are not available in the loaded episode summary.")

    with detail_right:
        st.markdown("**Survival/Draw Behavior Proxy**")
        if "duration" in outcome_df.columns:
            survival_aggs: dict[str, tuple[str, str]] = {
                "episodes": ("episode_id", "count"),
                "duration": ("duration", "mean"),
            }
            if "final_worker" in outcome_df.columns:
                survival_aggs["final_workers"] = ("final_worker", "mean")

            survival_summary = outcome_df.groupby("outcome", as_index=False, observed=True).agg(**survival_aggs)
            survival_summary["outcome"] = pd.Categorical(survival_summary["outcome"], categories=OUTCOME_ORDER, ordered=True)
            survival_summary = survival_summary.sort_values("outcome")
            column_config: dict[str, Any] = {
                "outcome": "Outcome",
                "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                "duration": st.column_config.NumberColumn("Duration", format="%.3f"),
            }
            if "final_workers" in survival_summary.columns:
                column_config["final_workers"] = st.column_config.NumberColumn("Final workers", format="%.3f")
            st.dataframe(
                survival_summary,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )
        else:
            st.info("Survival proxy columns are not available in the loaded episode summary.")

        draw_by_enemy = (
            outcome_df.groupby("enemy", as_index=False)
            .agg(episodes=("episode_id", "count"), draw_rate=("win", lambda values: float((values == 0.5).mean())))
            .sort_values("draw_rate", ascending=False)
        )
        st.dataframe(
            draw_by_enemy,
            use_container_width=True,
            hide_index=True,
            column_config={
                "enemy": "Enemy",
                "episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                "draw_rate": st.column_config.NumberColumn("Draw rate", format="%.3f"),
            },
        )


def render_time_series(time_df: pd.DataFrame) -> None:
    available_metrics = [metric for metric in METRIC_LABELS if metric in time_df.columns]
    controls = st.columns([1.2, 1, 1, 1])
    metric = controls[0].selectbox(
        "Metric",
        available_metrics,
        format_func=lambda value: METRIC_LABELS.get(value, clean_label(value)),
    )
    x_mode = controls[1].segmented_control(
        "X axis",
        ["Timesteps", "Normalized progress"],
        default="Timesteps",
    )
    cumulative = controls[2].toggle("Cumulative", value=metric.endswith(("gathered", "returned", "spent", "dealt", "taken", "produced", "lost", "killed")))
    bins = controls[3].slider("Progress bins", 10, 100, 50, disabled=x_mode == "Timesteps")

    metric_label = METRIC_LABELS.get(metric, clean_label(metric))
    agg = aggregate_time_series(time_df, metric, x_mode, cumulative, bins)
    st.plotly_chart(plot_time_series(agg, metric_label, x_mode), use_container_width=True)


def render_correlations(episode_df: pd.DataFrame, selected_correlations: pd.DataFrame, matrix_df: pd.DataFrame) -> None:
    left, middle, right = st.columns([1, 1, 1])
    numeric_cols = [col for col in EPISODE_SCATTER_COLUMNS if col in episode_df.columns]

    with left:
        x_col = st.selectbox("X metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], index=0)
    with middle:
        y_default = numeric_cols.index("win") if "win" in numeric_cols else min(1, len(numeric_cols) - 1)
        y_col = st.selectbox("Y metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], index=y_default)
    discrete_x = episode_df[x_col].nunique(dropna=True) <= min(12, max(3, len(episode_df) // 10))
    plot_options = ["Scatter", "Line", "Box", "Violin"]
    with right:
        plot_kind = st.segmented_control(
            "Plot type",
            plot_options,
            default="Box" if discrete_x else "Scatter",
        )

    bin_col = x_col
    use_binned_x = False
    bin_count = 8
    if plot_kind in {"Line", "Box", "Violin"}:
        bin_controls = st.columns([1, 1])
        use_binned_x = bin_controls[0].toggle("Bin X values", value=not discrete_x)
        bin_count = bin_controls[1].slider("X bins", 2, 30, 8, disabled=not use_binned_x)

    labels = {x_col: EPISODE_SCATTER_COLUMNS[x_col], y_col: EPISODE_SCATTER_COLUMNS[y_col]}
    hover_data = ["episode_id", "duration", "result"]
    plot_df = episode_df.copy()
    if plot_kind in {"Line", "Box", "Violin"}:
        if use_binned_x:
            bin_col = f"{x_col}_bin"
            clean_x = plot_df[x_col].replace([np.inf, -np.inf], np.nan)
            try:
                plot_df[bin_col] = pd.qcut(clean_x, q=bin_count, duplicates="drop")
            except ValueError:
                plot_df[bin_col] = pd.cut(clean_x, bins=bin_count, duplicates="drop")
            plot_df[bin_col] = plot_df[bin_col].astype("string")
            labels[bin_col] = f"{EPISODE_SCATTER_COLUMNS[x_col]} bin"
        else:
            bin_col = f"{x_col}_category"
            plot_df[bin_col] = plot_df[x_col].astype("string")
            labels[bin_col] = EPISODE_SCATTER_COLUMNS[x_col]

        if plot_kind == "Line":
            if use_binned_x:
                plot_df["_x_order"] = pd.Categorical(plot_df[bin_col], categories=plot_df[bin_col].dropna().unique(), ordered=True).codes
                grouped = (
                    plot_df.dropna(subset=[bin_col, y_col])
                    .groupby(["enemy", bin_col, "_x_order"], as_index=False, observed=True)
                    .agg(
                        y_mean=(y_col, "mean"),
                        y_median=(y_col, "median"),
                        y_std=(y_col, "std"),
                        n=(y_col, "count"),
                    )
                    .sort_values(["enemy", "_x_order"])
                )
                x_line_col = bin_col
            else:
                grouped = (
                    plot_df.dropna(subset=[x_col, y_col])
                    .groupby(["enemy", x_col], as_index=False)
                    .agg(
                        y_mean=(y_col, "mean"),
                        y_median=(y_col, "median"),
                        y_std=(y_col, "std"),
                        n=(y_col, "count"),
                    )
                    .sort_values(["enemy", x_col])
                )
                x_line_col = x_col

            grouped["sem"] = grouped["y_std"].fillna(0.0) / np.sqrt(grouped["n"].clip(lower=1))
            grouped["ci_low"] = grouped["y_mean"] - 1.96 * grouped["sem"]
            grouped["ci_high"] = grouped["y_mean"] + 1.96 * grouped["sem"]
            fig = px.line(
                grouped,
                x=x_line_col,
                y="y_mean",
                color="enemy",
                markers=True,
                hover_data=["y_median", "n"],
                labels={
                    x_line_col: labels.get(x_line_col, EPISODE_SCATTER_COLUMNS[x_col]),
                    "y_mean": f"Mean {EPISODE_SCATTER_COLUMNS[y_col]}",
                    "y_median": f"Median {EPISODE_SCATTER_COLUMNS[y_col]}",
                    "n": "Episodes",
                },
            )
            for enemy, group in grouped.groupby("enemy", sort=True):
                fig.add_trace(
                    go.Scatter(
                        x=group[x_line_col],
                        y=group["ci_high"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=group[x_line_col],
                        y=group["ci_low"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(128, 128, 128, 0.14)",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            if use_binned_x:
                fig.update_xaxes(type="category")
        elif plot_kind == "Box":
            fig = px.box(
                plot_df,
                x=bin_col,
                y=y_col,
                color="enemy",
                points="outliers",
                hover_data=hover_data,
                labels=labels,
            )
        else:
            fig = px.violin(
                plot_df,
                x=bin_col,
                y=y_col,
                color="enemy",
                box=True,
                points="outliers",
                hover_data=hover_data,
                labels=labels,
            )
        fig.update_xaxes(type="category")
    else:
        corr = pearsonr_safe(plot_df[x_col], plot_df[y_col])
        st.metric("Pearson r", format_number(corr))
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="enemy",
            hover_data=hover_data,
            labels=labels,
        )
        add_regression_lines(fig, plot_df, x_col, y_col)
    fig.update_layout(height=470, margin={"l": 8, "r": 8, "t": 28, "b": 8})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if not selected_correlations.empty:
            corr_plot_df = selected_correlations.copy()
            if "enemy" in corr_plot_df.columns:
                corr_plot_df = corr_plot_df[corr_plot_df["enemy"].astype(str).isin(["ALL", *episode_df["enemy"].astype(str).unique()])]
            fig = px.bar(
                corr_plot_df,
                x="pearson_r",
                y="metric",
                color="enemy" if "enemy" in corr_plot_df.columns else None,
                orientation="h",
                labels={"pearson_r": "Pearson r", "metric": "Metric"},
            )
            fig.update_layout(height=480, margin={"l": 8, "r": 8, "t": 28, "b": 8})
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not matrix_df.empty:
            matrix = matrix_df.set_index(matrix_df.columns[0]) if matrix_df.columns[0].startswith("Unnamed") else matrix_df.set_index(matrix_df.columns[0])
            fig = px.imshow(
                matrix,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                labels={"color": "Pearson r"},
            )
            fig.update_layout(height=480, margin={"l": 8, "r": 8, "t": 28, "b": 8})
            st.plotly_chart(fig, use_container_width=True)


def render_episode_explorer(episode_df: pd.DataFrame) -> None:
    cols = [col for col in EPISODE_SCATTER_COLUMNS if col in episode_df.columns]
    metric = st.selectbox("Distribution metric", cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value])
    fig = px.box(
        episode_df,
        x="enemy",
        y=metric,
        color="enemy",
        points="outliers",
        labels={"enemy": "Enemy", metric: EPISODE_SCATTER_COLUMNS[metric]},
    )
    fig.update_layout(height=430, margin={"l": 8, "r": 8, "t": 28, "b": 8}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        col
        for col in [
            "episode_id",
            "enemy",
            "result",
            "win",
            "duration",
            "resources_gathered_total",
            "resources_spent_total",
            "resource_return_success_rate",
            "units_produced_total",
            "units_killed_total",
            "units_lost_total",
            "damage_dealt_total",
            "damage_taken_total",
        ]
        if col in episode_df.columns
    ]
    st.dataframe(
        episode_df[display_cols].sort_values(["enemy", "episode_id"]),
        use_container_width=True,
        hide_index=True,
    )


def render_methods_reference() -> None:
    st.caption(
        "A plain-language reference for the statistical summaries, tests, effect sizes, and corrections used in this dashboard."
    )
    st.info(
        "Inference tests use the currently filtered data. Results describe associations in these runs; they do not prove causation."
    )

    for section in STAT_METHOD_SECTIONS:
        st.subheader(section["title"])
        for method in section["methods"]:
            with st.expander(method["name"], expanded=False):
                st.markdown(f"**Used for:** {method['used_for']}")
                st.write(method["explanation"])
                resource_links = " | ".join(
                    f"[{label}]({url})" for label, url in method["resources"]
                )
                st.markdown(f"**Further resources:** {resource_links}")


def render_trajectory_charts(
    trajectory_df: pd.DataFrame,
    result_df: pd.DataFrame,
    grouping: str,
) -> None:
    if trajectory_df.empty:
        st.warning("No trajectory summaries are available for the current filters.")
        return

    group_col = "outcome" if grouping == "Outcome" else "enemy"
    plot_df = trajectory_df.copy()
    if grouping == "Outcome":
        plot_df = plot_df[plot_df["outcome"].isin(["Win", "Loss"])]

    grouped = (
        plot_df.groupby([group_col, "progress_bin_index", "progress_bin"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n=("value", "count"),
        )
        .sort_values([group_col, "progress_bin_index"])
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["n"].clip(lower=1))
    grouped["lower"] = grouped["mean"] - 1.96 * grouped["sem"]
    grouped["upper"] = grouped["mean"] + 1.96 * grouped["sem"]

    fig = go.Figure()
    for group_name, group in grouped.groupby(group_col, sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["progress_bin"],
                y=group["upper"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["progress_bin"],
                y=group["lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.14)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["progress_bin"],
                y=group["mean"],
                mode="lines+markers",
                name=str(group_name),
                hovertemplate="Progress: %{x}<br>Mean: %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        height=390,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        xaxis_title="Normalized progress bin",
        yaxis_title="Mean summary value",
        legend_title_text=grouping,
    )
    st.plotly_chart(fig, use_container_width=True)

    valid_results = result_df[result_df["test"] != "Not run"].copy()
    if valid_results.empty:
        return

    left, right = st.columns(2)
    effect_fig = px.line(
        valid_results,
        x="progress_bin",
        y="effect_size",
        color="test",
        markers=True,
        labels={"progress_bin": "Progress bin", "effect_size": "Effect size"},
    )
    effect_fig.add_hline(y=0.0, line_dash="dash", opacity=0.5)
    effect_fig.update_layout(height=360, margin={"l": 8, "r": 8, "t": 28, "b": 8})
    left.plotly_chart(effect_fig, use_container_width=True)

    p_col = "p_value_corrected" if "p_value_corrected" in valid_results.columns else "p_value"
    p_fig = px.line(
        valid_results,
        x="progress_bin",
        y=p_col,
        color="test",
        markers=True,
        labels={"progress_bin": "Progress bin", p_col: "Corrected p-value" if p_col == "p_value_corrected" else "p-value"},
    )
    p_fig.add_hline(y=0.05, line_dash="dash", line_color="red", opacity=0.65)
    p_fig.update_yaxes(range=[0, 1])
    p_fig.update_layout(height=360, margin={"l": 8, "r": 8, "t": 28, "b": 8})
    right.plotly_chart(p_fig, use_container_width=True)


def render_episode_level_inference(episode_df: pd.DataFrame) -> None:
    numeric_cols = [col for col in EPISODE_SCATTER_COLUMNS if col in episode_df.columns]
    st.caption("Inference tests use filtered episode-level rows only. Timestep rows are not treated as independent observations.")

    test_family = st.selectbox(
        "Test family",
        [
            "Win rate by enemy",
            "Metric by outcome",
            "Metric by enemy",
            "Correlation test",
            "Logistic regression",
        ],
        key="inference_test_family",
    )

    effect_size_help = {
        "Win rate by enemy": (
            "Effect size is Cramer's V. It ranges from 0 to 1 and measures how strongly enemy identity is associated with win/loss outcome."
        ),
        "Metric by outcome": (
            "Welch t-test uses Cohen's d: positive means the metric is higher in wins, negative means higher in losses. "
            "Mann-Whitney U uses Cliff's delta, ranging from -1 to 1 with the same direction."
        ),
        "Metric by enemy": (
            "Kruskal-Wallis uses an epsilon-squared-like effect size, and ANOVA uses eta squared. "
            "Both indicate how much metric variation is explained by enemy group."
        ),
        "Correlation test": (
            "Effect size is the correlation coefficient. Values near -1 or 1 indicate stronger association; values near 0 indicate weak association."
        ),
        "Logistic regression": (
            "Effect size is an odds ratio. 1 means no estimated effect; above 1 means higher odds of winning; below 1 means lower odds. "
            "This changes odds, not percentage-point win probability."
        ),
    }
    st.info(effect_size_help[test_family])

    result_df = pd.DataFrame()
    support_df = pd.DataFrame()

    if test_family == "Win rate by enemy":
        if st.button("Run test", type="primary", key="inference_run_win_rate_by_enemy"):
            result_df, support_df = chi_square_win_by_enemy(episode_df)
    elif test_family == "Metric by outcome":
        metric = st.selectbox("Metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], key="inference_metric_by_outcome")
        if st.button("Run test", type="primary", key="inference_run_metric_by_outcome"):
            result_df = compare_metric_by_outcome(episode_df, metric)
            support_df = (
                episode_df[episode_df["win"].isin([0.0, 1.0])]
                .assign(outcome=lambda df: df["win"].map({0.0: "Loss", 1.0: "Win"}))
                .groupby("outcome", as_index=False)
                .agg(
                    n=(metric, "count"),
                    mean=(metric, "mean"),
                    median=(metric, "median"),
                    std=(metric, "std"),
                )
            )
            fig = px.violin(
                episode_df[episode_df["win"].isin([0.0, 1.0])].assign(outcome=lambda df: df["win"].map({0.0: "Loss", 1.0: "Win"})),
                x="outcome",
                y=metric,
                color="outcome",
                box=True,
                points="outliers",
                labels={"outcome": "Outcome", metric: EPISODE_SCATTER_COLUMNS[metric]},
            )
            fig.update_layout(height=380, margin={"l": 8, "r": 8, "t": 28, "b": 8}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    elif test_family == "Metric by enemy":
        metric = st.selectbox("Metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], key="inference_metric_by_enemy")
        if st.button("Run test", type="primary", key="inference_run_metric_by_enemy"):
            result_df = compare_metric_by_enemy(episode_df, metric)
            support_df = (
                episode_df.groupby("enemy", as_index=False)
                .agg(
                    n=(metric, "count"),
                    mean=(metric, "mean"),
                    median=(metric, "median"),
                    std=(metric, "std"),
                )
                .sort_values("enemy")
            )
            fig = px.box(
                episode_df,
                x="enemy",
                y=metric,
                color="enemy",
                points="outliers",
                labels={"enemy": "Enemy", metric: EPISODE_SCATTER_COLUMNS[metric]},
            )
            fig.update_layout(height=380, margin={"l": 8, "r": 8, "t": 28, "b": 8}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    elif test_family == "Correlation test":
        left, right = st.columns(2)
        x_col = left.selectbox("X metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], index=0, key="inference_correlation_x")
        y_default = numeric_cols.index("win") if "win" in numeric_cols else min(1, len(numeric_cols) - 1)
        y_col = right.selectbox("Y metric", numeric_cols, format_func=lambda value: EPISODE_SCATTER_COLUMNS[value], index=y_default, key="inference_correlation_y")
        if st.button("Run test", type="primary", key="inference_run_correlation"):
            result_df = correlation_tests(episode_df, x_col, y_col)
            support_df = pd.DataFrame(
                [
                    {
                        "x_metric": x_col,
                        "y_metric": y_col,
                        "complete_observations": len(episode_df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()),
                    }
                ]
            )
            fig = px.scatter(
                episode_df,
                x=x_col,
                y=y_col,
                color="enemy",
                hover_data=["episode_id", "duration", "result"],
                labels={x_col: EPISODE_SCATTER_COLUMNS[x_col], y_col: EPISODE_SCATTER_COLUMNS[y_col]},
            )
            add_regression_lines(fig, episode_df, x_col, y_col)
            fig.update_layout(height=380, margin={"l": 8, "r": 8, "t": 28, "b": 8})
            st.plotly_chart(fig, use_container_width=True)
    else:
        metric = st.selectbox(
            "Predictor metric",
            [col for col in numeric_cols if col != "win"],
            format_func=lambda value: EPISODE_SCATTER_COLUMNS[value],
            key="inference_logistic_metric",
        )
        if st.button("Run test", type="primary", key="inference_run_logistic"):
            result_df = logistic_regression_win(episode_df, metric)
            support_df = pd.DataFrame(
                [
                    {
                        "episodes": int(episode_df["win"].isin([0.0, 1.0]).sum()),
                        "predictor": metric,
                        "enemies": int(episode_df["enemy"].nunique()),
                        "draws_excluded": int((episode_df["win"] == 0.5).sum()),
                    }
                ]
            )

    if not result_df.empty:
        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results CSV",
            result_df.to_csv(index=False),
            file_name="inference_results.csv",
            mime="text/csv",
        )

    if not support_df.empty:
        st.subheader("Supporting table")
        st.dataframe(support_df, use_container_width=True, hide_index=True)


def render_trajectory_inference(time_df: pd.DataFrame) -> None:
    st.caption(
        "Trajectory inference summarizes each episode inside normalized progress bins before testing. "
        "Raw timestep rows are not treated as independent observations."
    )

    numeric_cols = [
        col
        for col in METRIC_LABELS
        if col in time_df.columns and pd.api.types.is_numeric_dtype(time_df[col])
    ]
    if not numeric_cols:
        st.warning("No numeric time-series metrics are available for trajectory inference.")
        return

    default_metric = numeric_cols.index("army_value") if "army_value" in numeric_cols else 0
    controls = st.columns([1.2, 0.9, 0.8, 0.9, 0.8])
    metric = controls[0].selectbox(
        "Metric",
        numeric_cols,
        index=default_metric,
        format_func=lambda value: METRIC_LABELS.get(value, clean_label(value)),
        key="trajectory_metric",
    )
    grouping = controls[1].selectbox("Group by", ["Outcome", "Enemy"], key="trajectory_grouping")
    bins = controls[2].slider("Progress bins", 5, 30, 10, key="trajectory_bins")
    summary = controls[3].selectbox("Bin summary", TRAJECTORY_SUMMARIES, key="trajectory_summary")
    apply_fdr = controls[4].toggle("FDR correction", value=True, key="trajectory_fdr")

    if st.button("Run trajectory inference", type="primary", key="trajectory_run"):
        try:
            trajectory_df = build_trajectory_summary(time_df, metric=metric, bins=bins, summary=summary)
        except Exception as e:
            st.error(f"Could not build trajectory summary: {e}")
            return

        result_df = run_trajectory_inference(
            trajectory_df,
            metric=metric,
            summary=summary,
            grouping=grouping,
            apply_fdr=apply_fdr,
        )

        st.subheader("Trajectory plots")
        render_trajectory_charts(trajectory_df, result_df, grouping)

        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download trajectory results CSV",
            result_df.to_csv(index=False),
            file_name="trajectory_inference_results.csv",
            mime="text/csv",
        )

        st.subheader("Episode-bin summaries")
        preview_cols = ["episode_id", "enemy", "outcome", "progress_bin", "metric", "summary", "value"]
        st.dataframe(trajectory_df[preview_cols].head(500), use_container_width=True, hide_index=True)


def render_inference(episode_df: pd.DataFrame, data_dir: str, episode_ids: set) -> None:
    inference_mode = st.segmented_control(
        "Inference mode",
        ["Episode-level", "Trajectory"],
        default="Episode-level",
        key="inference_mode",
    )
    if inference_mode == "Episode-level":
        render_episode_level_inference(episode_df)
    else:
        time_df = load_timeseries_table(data_dir)
        require_columns(time_df, ["episode_id", "enemy", "t", "progress"], "timeseries")
        time_df = time_df[time_df["episode_id"].isin(episode_ids)].copy()
        render_trajectory_inference(time_df)


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("OCGDT Stats Dashboard")

    with st.sidebar:
        data_dir = st.text_input("Precomputed stats directory", value=str(DEFAULT_DATA_DIR))
        st.caption("Reads Parquet when present, otherwise CSV. Raw episode logs are not parsed here.")

    try:
        tables = load_episode_tables(data_dir)
    except Exception as e:
        st.error(f"Could not load precomputed tables: {e}")
        st.stop()

    episode_df = tables["episode"]
    require_columns(episode_df, ["episode_id", "enemy", "win", "duration"], "episode_summary")

    with st.sidebar:
        enemies = sorted(episode_df["enemy"].astype(str).dropna().unique())
        selected_enemies = st.multiselect("Enemies", enemies, default=enemies)
        outcome = st.radio("Outcome", ["All", "Wins", "Losses", "Draws"], horizontal=True)
        duration_min = int(episode_df["duration"].min())
        duration_max = int(episode_df["duration"].max())
        duration_range = st.slider("Duration", duration_min, duration_max, (duration_min, duration_max))

    filtered_episode_df = apply_episode_filters(episode_df, selected_enemies, outcome, duration_range)
    episode_ids = set(filtered_episode_df["episode_id"])

    if filtered_episode_df.empty:
        st.warning("No episodes match the current filters.")
        st.stop()

    active_view = st.segmented_control(
        "View",
        ["Overview", "Key Statistics", "Time Series", "Correlations", "Inference", "Methods", "Episodes"],
        default="Overview",
        label_visibility="collapsed",
    )

    if active_view == "Overview":
        render_overview(filtered_episode_df, tables["summary_by_enemy"])
    elif active_view == "Key Statistics":
        render_key_statistics(filtered_episode_df)
    elif active_view == "Time Series":
        time_df = load_timeseries_table(data_dir)
        require_columns(time_df, ["episode_id", "enemy", "t", "progress"], "timeseries")
        filtered_time_df = time_df[time_df["episode_id"].isin(episode_ids)].copy()
        render_time_series(filtered_time_df)
    elif active_view == "Correlations":
        render_correlations(filtered_episode_df, tables["selected_correlations"], tables["correlation_matrix"])
    elif active_view == "Inference":
        render_inference(filtered_episode_df, data_dir, episode_ids)
    elif active_view == "Methods":
        render_methods_reference()
    elif active_view == "Episodes":
        render_episode_explorer(filtered_episode_df)


if __name__ == "__main__":
    main()
