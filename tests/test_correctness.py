from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import dashboard
import inference
import main


def test_damage_metrics_include_lethal_disappearing_units() -> None:
    previous_p1 = {1: {"id": 1, "owner": 0, "unit_type": 4, "hp": 2}}
    current_p1 = {}
    previous_p2 = {2: {"id": 2, "owner": 1, "unit_type": 6, "hp": 4}}
    current_p2 = {}

    assert main.damage_metrics(previous_p1, current_p1, previous_p2, current_p2) == {
        "damage_dealt": 4,
        "damage_taken": 2,
    }


def _episode(actions_by_timestep: list[list[dict]], resources: list[int] | None = None) -> tuple[str, Path, dict]:
    duration = len(actions_by_timestep)
    resources = resources or [5] * duration
    units = [
        [
            {"id": 10, "owner": 0, "unit_type": 2, "x": 0, "y": 0, "hp": 10, "resources": 0},
            {"id": 11, "owner": 0, "unit_type": 4, "x": 1, "y": 0, "hp": 1, "resources": min(t, 1)},
            {"id": 20, "owner": 1, "unit_type": 2, "x": 7, "y": 7, "hp": 10, "resources": 0},
        ]
        for t in range(duration)
    ]
    return (
        "episode",
        Path("episode.json"),
        {
            "result": 1,
            "enemy": "enemy",
            "duration": duration,
            "game_state": {
                "width": 8,
                "height": 8,
                "players": [
                    {"id": 0, "resources": resources},
                    {"id": 1, "resources": [5] * duration},
                ],
                "units_by_timestep": units,
                "actions_by_timestep": actions_by_timestep,
            },
        },
    )


def test_return_success_uses_return_actions_over_harvest_actions() -> None:
    _, episode_df = main.build_frames(
        [
            _episode(
                [
                    [{"action_type": 2, "unit_id": 11}],
                    [{"action_type": 2, "unit_id": 11}],
                    [{"action_type": 3, "unit_id": 11}],
                ]
            )
        ]
    )

    row = episode_df.iloc[0]
    assert row["action_harvest_total"] == 2
    assert row["action_return_total"] == 1
    assert row["resource_return_success_rate"] == 0.5


def test_return_success_is_nan_when_no_harvest_actions() -> None:
    _, episode_df = main.build_frames([_episode([[], []])])

    assert episode_df.iloc[0]["action_harvest_total"] == 0
    assert math.isnan(episode_df.iloc[0]["resource_return_success_rate"])


def test_true_win_rate_excludes_draws() -> None:
    values = pd.Series([1.0, 0.5, 0.0, 0.5])

    assert dashboard.true_win_rate(values) == 0.5
    assert values.mean() == 0.5


def test_normalized_time_series_is_episode_weighted() -> None:
    time_df = pd.DataFrame(
        {
            "episode_id": ["long"] * 4 + ["short"],
            "enemy": ["a"] * 5,
            "t": [0, 1, 2, 3, 0],
            "progress": [0.0, 0.25, 0.5, 1.0, 0.0],
            "metric": [100, 100, 100, 100, 0],
        }
    )

    agg = dashboard.aggregate_time_series(time_df, "metric", "Normalized progress", cumulative=False, bins=1)

    assert agg.iloc[0]["n"] == 2
    assert agg.iloc[0]["mean"] == 50


def test_same_metric_correlation_returns_not_run() -> None:
    result = inference.correlation_tests(pd.DataFrame({"x": [1, 2, 3]}), "x", "x")

    assert result.iloc[0]["test"] == "Not run"


def test_trajectory_auc_is_order_invariant() -> None:
    trajectory = pd.DataFrame(
        {
            "episode_id": ["e1"] * 4,
            "enemy": ["a"] * 4,
            "win": [1.0] * 4,
            "t": [0, 1, 2, 3],
            "progress": [0.0, 0.25, 0.5, 1.0],
            "metric": [0.0, 10.0, 0.0, 0.0],
        }
    )
    shuffled = trajectory.sample(frac=1.0, random_state=3).reset_index(drop=True)

    sorted_auc = inference.build_trajectory_summary(trajectory, "metric", bins=1, summary="AUC")["value"].iloc[0]
    shuffled_auc = inference.build_trajectory_summary(shuffled, "metric", bins=1, summary="AUC")["value"].iloc[0]

    assert sorted_auc == shuffled_auc == 2.5


def test_metric_by_enemy_uses_only_eligible_groups_for_n() -> None:
    df = pd.DataFrame({"enemy": ["a", "a", "b", "b", "c"], "metric": [1, 2, 10, 11, 1000]})

    result = inference.compare_metric_by_enemy(df, "metric")

    assert set(result["n"]) == {4}


def test_metric_by_enemy_all_identical_returns_not_run() -> None:
    df = pd.DataFrame({"enemy": ["a", "a", "b", "b"], "metric": [1, 1, 1, 1]})

    result = inference.compare_metric_by_enemy(df, "metric")

    assert result.iloc[0]["test"] == "Not run"


def test_generated_stats_are_parquet_by_default(tmp_path: Path) -> None:
    episode_df = pd.DataFrame(
        {
            "average_distance_to_resource": [2.0, 4.0],
            "shortest_distance_to_resource": [1.0, 2.0],
            "resources_gathered_total": [10.0, 20.0],
            "resource_return_success_rate": [0.5, 1.0],
            "resources_spent_total": [3.0, 6.0],
            "units_produced_total": [1.0, 2.0],
            "units_killed_total": [0.0, 1.0],
            "value_killed_total": [0.0, 5.0],
            "duration": [10.0, 20.0],
            "win": [0.0, 1.0],
        }
    )

    main.create_visualizations(pd.DataFrame(), episode_df, tmp_path, bins=8)

    assert (tmp_path / "selected_correlations.parquet").exists()
    assert (tmp_path / "correlation_matrix.parquet").exists()
    assert not (tmp_path / "selected_correlations.csv").exists()
    assert not (tmp_path / "correlation_matrix.csv").exists()
    assert "metric" in pd.read_parquet(tmp_path / "selected_correlations.parquet").columns
    assert pd.read_parquet(tmp_path / "correlation_matrix.parquet").index.name == "metric"


def test_dashboard_optional_tables_prefer_parquet(tmp_path: Path) -> None:
    pd.DataFrame({"source": ["csv"]}).to_csv(tmp_path / "summary_by_enemy.csv", index=False)
    pd.DataFrame({"source": ["parquet"]}).to_parquet(tmp_path / "summary_by_enemy.parquet", index=False)

    loaded = dashboard.read_optional_table(tmp_path, "summary_by_enemy")

    assert loaded["source"].iloc[0] == "parquet"
