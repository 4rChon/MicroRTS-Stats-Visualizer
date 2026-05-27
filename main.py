from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **_: Any):  # type: ignore[no-redef]
        return iterable if iterable is not None else []


PLAYER_1 = 0
PLAYER_2 = 1
NEUTRAL = -1

UNIT_TYPE_NAMES = {
    0: "none",
    1: "resource",
    2: "base",
    3: "barracks",
    4: "worker",
    5: "light",
    6: "heavy",
    7: "ranged",
}

ACTION_TYPE_NAMES = {
    0: "noop",
    1: "move",
    2: "harvest",
    3: "return",
    4: "produce",
    5: "attack",
}

UNIT_COSTS = {
    0: 0,   # none
    1: 0,   # resource
    2: 10,  # base
    3: 5,   # barracks
    4: 1,   # worker
    5: 2,   # light
    6: 4,   # heavy
    7: 2,   # ranged
}

TRACKED_UNIT_TYPES = [2, 3, 4, 5, 6, 7]
COMBAT_UNIT_TYPES = [5, 6, 7]
ARMY_VALUE_TYPES = [5, 6, 7]
ECONOMY_VALUE_TYPES = [4]
INFRASTRUCTURE_VALUE_TYPES = [2, 3]


def safe_filename(value: Any) -> str:
    text = str(value).strip() or "unknown_enemy"
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        elif char.isspace() or char in {"/", "\\", ":", "."}:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe or "unknown_enemy"


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def open_json_or_gzip(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_rich_episode_schema(data: dict[str, Any]) -> bool:
    game_state = data.get("game_state")
    if not isinstance(game_state, dict):
        return False

    return all(
        key in data
        for key in ("result", "enemy", "duration")
    ) and all(
        key in game_state
        for key in ("players", "units_by_timestep", "actions_by_timestep")
    )


def load_episode_jsons(input_dir: Path, pattern: str = "*.json*") -> list[tuple[str, Path, dict[str, Any]]]:
    paths = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    episodes: list[tuple[str, Path, dict[str, Any]]] = []

    for path in tqdm(paths, desc="Loading episode JSONs", unit="file"):
        try:
            data = open_json_or_gzip(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"Skipping unreadable JSON: {path} ({e})")
            continue

        if not is_rich_episode_schema(data):
            print(f"Skipping non-rich episode JSON: {path}")
            continue

        episode_id = path.name
        if episode_id.endswith(".json.gz"):
            episode_id = episode_id[:-8]
        elif episode_id.endswith(".json"):
            episode_id = episode_id[:-5]

        episodes.append((episode_id, path, data))

    if not episodes:
        raise FileNotFoundError(f"No rich episode JSON files found in {input_dir} using pattern {pattern!r}")

    return episodes


def get_unit_type(unit: dict[str, Any]) -> int:
    return int(unit.get("unit_type", unit.get("type", 0)))


def get_action_type(action: dict[str, Any]) -> int:
    return int(action.get("action_type", action.get("type", 0)))


def get_action_unit_type(action: dict[str, Any]) -> int:
    return int(action.get("unit_type", action.get("unitType", 0)))


def get_owner(unit: dict[str, Any]) -> int:
    return int(unit["owner"])


def get_unit_id(unit: dict[str, Any]) -> int:
    return int(unit["id"])


def units_by_id(units: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {get_unit_id(unit): unit for unit in units}


def owned_units_by_id(units: list[dict[str, Any]], owner: int) -> dict[int, dict[str, Any]]:
    return {get_unit_id(unit): unit for unit in units if get_owner(unit) == owner}


def count_units(units: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts = {unit_type: 0 for unit_type in TRACKED_UNIT_TYPES}
    for unit in units:
        unit_type = get_unit_type(unit)
        if unit_type in counts:
            counts[unit_type] += 1
    return counts


def unit_value(unit: dict[str, Any]) -> int:
    return UNIT_COSTS.get(get_unit_type(unit), 0)


def total_value(units: Iterable[dict[str, Any]], unit_types: Iterable[int]) -> int:
    allowed = set(unit_types)
    return sum(unit_value(unit) for unit in units if get_unit_type(unit) in allowed)


def resources_carried(units: Iterable[dict[str, Any]]) -> int:
    return sum(int(unit.get("resources", 0)) for unit in units)


def resource_nodes_remaining(units: Iterable[dict[str, Any]]) -> int:
    return sum(int(unit.get("resources", 0)) for unit in units if get_unit_type(unit) == 1)


def position_distance(unit_a: dict[str, Any], unit_b: dict[str, Any]) -> int:
    return abs(int(unit_a["x"]) - int(unit_b["x"])) + abs(int(unit_a["y"]) - int(unit_b["y"]))


def min_distance_to_units(
    source_units: Iterable[dict[str, Any]],
    target_units: Iterable[dict[str, Any]],
) -> float:
    sources = list(source_units)
    targets = list(target_units)
    if not sources or not targets:
        return float("nan")

    return float(min(position_distance(source, target) for source in sources for target in targets))


def get_player_payload(game_state: dict[str, Any], player_id: int) -> dict[str, Any]:
    players = game_state.get("players", [])
    if len(players) > player_id and int(players[player_id].get("id", player_id)) == player_id:
        return players[player_id]

    for player in players:
        if int(player.get("id")) == player_id:
            return player

    raise ValueError(f"Could not find player payload for player id {player_id}")


def scalar_or_timestep_value(value: Any, t: int, default: float = float("nan")) -> float:
    if isinstance(value, list):
        if not value:
            return default
        if t < len(value):
            return float(value[t])
        return float(value[-1])
    if value is None:
        return default
    return float(value)


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0 or math.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def player_metric(player_payload: dict[str, Any], key: str, t: int) -> float:
    return scalar_or_timestep_value(player_payload.get(key), t)


def get_resource_bank(player_payload: dict[str, Any], t: int) -> float:
    return scalar_or_timestep_value(player_payload.get("resources"), t, default=0.0)


def timestep_items(timesteps: Any, t: int) -> list[dict[str, Any]]:
    if isinstance(timesteps, list) and t < len(timesteps) and isinstance(timesteps[t], list):
        return timesteps[t]
    return []


def action_counts_for_player(
    actions: list[dict[str, Any]],
    current_units: dict[int, dict[str, Any]],
    previous_units: dict[int, dict[str, Any]],
    player_id: int,
) -> dict[str, int]:
    counts = {f"action_{name}": 0 for name in ACTION_TYPE_NAMES.values()}

    for action in actions:
        unit_id = int(action.get("unit_id", -1))
        acting_unit = current_units.get(unit_id) or previous_units.get(unit_id)
        if acting_unit is None or get_owner(acting_unit) != player_id:
            continue

        action_type = get_action_type(action)
        action_name = ACTION_TYPE_NAMES.get(action_type, f"unknown_{action_type}")
        counts[f"action_{action_name}"] = counts.get(f"action_{action_name}", 0) + 1

    return counts


def produced_lost_killed_metrics(
    previous_p1: dict[int, dict[str, Any]],
    current_p1: dict[int, dict[str, Any]],
    previous_p2: dict[int, dict[str, Any]],
    current_p2: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    produced_ids = set(current_p1) - set(previous_p1)
    lost_ids = set(previous_p1) - set(current_p1)
    killed_ids = set(previous_p2) - set(current_p2)

    produced_units = [current_p1[unit_id] for unit_id in produced_ids]
    lost_units = [previous_p1[unit_id] for unit_id in lost_ids]
    killed_units = [previous_p2[unit_id] for unit_id in killed_ids]

    metrics: dict[str, Any] = {
        "units_produced": len(produced_units),
        "units_lost": len(lost_units),
        "units_killed": len(killed_units),
        "value_produced": sum(unit_value(unit) for unit in produced_units),
        "value_lost": sum(unit_value(unit) for unit in lost_units),
        "value_killed": sum(unit_value(unit) for unit in killed_units),
    }

    for unit_type in TRACKED_UNIT_TYPES:
        name = UNIT_TYPE_NAMES[unit_type]
        metrics[f"produced_{name}"] = sum(1 for unit in produced_units if get_unit_type(unit) == unit_type)
        metrics[f"lost_{name}"] = sum(1 for unit in lost_units if get_unit_type(unit) == unit_type)
        metrics[f"killed_{name}"] = sum(1 for unit in killed_units if get_unit_type(unit) == unit_type)

    return metrics


def damage_metrics(
    previous_p1: dict[int, dict[str, Any]],
    current_p1: dict[int, dict[str, Any]],
    previous_p2: dict[int, dict[str, Any]],
    current_p2: dict[int, dict[str, Any]],
) -> dict[str, int]:
    damage_taken = 0
    for unit_id in set(previous_p1) & set(current_p1):
        damage_taken += max(0, int(previous_p1[unit_id]["hp"]) - int(current_p1[unit_id]["hp"]))

    damage_dealt = 0
    for unit_id in set(previous_p2) & set(current_p2):
        damage_dealt += max(0, int(previous_p2[unit_id]["hp"]) - int(current_p2[unit_id]["hp"]))

    return {
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
    }


def harvested_resources(
    previous_p1: dict[int, dict[str, Any]],
    current_p1: dict[int, dict[str, Any]],
) -> int:
    harvested = 0

    for unit_id, current_unit in current_p1.items():
        current_resources = int(current_unit.get("resources", 0))
        previous_unit = previous_p1.get(unit_id)
        previous_resources = int(previous_unit.get("resources", 0)) if previous_unit is not None else 0
        harvested += max(0, current_resources - previous_resources)

    return harvested


def first_nonzero_t(df: pd.DataFrame, column: str) -> float:
    matches = df.index[df[column] > 0].tolist()
    if not matches:
        return float("nan")
    return float(df.loc[matches[0], "t"])


def build_frames(
    episodes: Iterable[tuple[str, Path, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_rows: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []

    episode_list = list(episodes)
    for episode_id, path, data in tqdm(episode_list, desc="Building data frames", unit="episode"):
        game_state = data["game_state"]
        player_1_payload = get_player_payload(game_state, PLAYER_1)

        units_by_timestep = game_state.get("units_by_timestep", [])
        actions_by_timestep = game_state.get("actions_by_timestep", [])
        if not isinstance(units_by_timestep, list):
            print(f"Skipping episode with invalid units_by_timestep: {path}")
            continue
        if not isinstance(actions_by_timestep, list):
            actions_by_timestep = []

        duration = int(data.get("duration", len(units_by_timestep)))
        duration = min(duration, len(units_by_timestep))

        if duration <= 0:
            print(f"Skipping empty episode: {path}")
            continue

        result = float(data.get("result", 0.0))
        win = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
        enemy = str(data.get("enemy", ""))

        rows: list[dict[str, Any]] = []

        previous_all: dict[int, dict[str, Any]] = {}
        previous_p1: dict[int, dict[str, Any]] = {}
        previous_p2: dict[int, dict[str, Any]] = {}
        previous_bank = get_resource_bank(player_1_payload, 0)

        for t in range(duration):
            units_t = timestep_items(units_by_timestep, t)
            actions_t = timestep_items(actions_by_timestep, t)

            current_all = units_by_id(units_t)
            current_p1 = owned_units_by_id(units_t, PLAYER_1)
            current_p2 = owned_units_by_id(units_t, PLAYER_2)

            p1_units = list(current_p1.values())
            p2_units = list(current_p2.values())
            all_units = list(current_all.values())

            bank = get_resource_bank(player_1_payload, t)
            bank_delta = 0.0 if t == 0 else bank - previous_bank

            current_counts = count_units(p1_units)
            opponent_counts = count_units(p2_units)

            action_counts = action_counts_for_player(
                actions_t,
                current_units=current_all,
                previous_units=previous_all,
                player_id=PLAYER_1,
            )

            if t == 0:
                event_metrics = {
                    "units_produced": 0,
                    "units_lost": 0,
                    "units_killed": 0,
                    "value_produced": 0,
                    "value_lost": 0,
                    "value_killed": 0,
                }
                for unit_type in TRACKED_UNIT_TYPES:
                    name = UNIT_TYPE_NAMES[unit_type]
                    event_metrics[f"produced_{name}"] = 0
                    event_metrics[f"lost_{name}"] = 0
                    event_metrics[f"killed_{name}"] = 0
                damage = {"damage_dealt": 0, "damage_taken": 0}
                gathered = 0
            else:
                event_metrics = produced_lost_killed_metrics(
                    previous_p1=previous_p1,
                    current_p1=current_p1,
                    previous_p2=previous_p2,
                    current_p2=current_p2,
                )
                damage = damage_metrics(
                    previous_p1=previous_p1,
                    current_p1=current_p1,
                    previous_p2=previous_p2,
                    current_p2=current_p2,
                )
                gathered = harvested_resources(previous_p1, current_p1)

            resources_returned = max(bank_delta, 0.0)
            resources_spent = max(-bank_delta, 0.0)

            workers = [unit for unit in p1_units if get_unit_type(unit) == 4]
            combat_units = [unit for unit in p1_units if get_unit_type(unit) in COMBAT_UNIT_TYPES]
            enemy_units = [unit for unit in p2_units if get_unit_type(unit) != 1]
            enemy_base_units = [unit for unit in p2_units if get_unit_type(unit) == 2]
            resource_units = [unit for unit in all_units if get_unit_type(unit) == 1]

            p1_total_value = total_value(p1_units, TRACKED_UNIT_TYPES)
            p2_total_value = total_value(p2_units, TRACKED_UNIT_TYPES)

            row: dict[str, Any] = {
                "episode_id": episode_id,
                "source_file": str(path),
                "enemy": enemy,
                "t": t,
                "duration": duration,
                "progress": t / max(duration - 1, 1),
                "result": result,
                "win": win,
                "map_width": int(game_state.get("width", 0)),
                "map_height": int(game_state.get("height", 0)),
                "resource_bank": bank,
                "resource_bank_delta": bank_delta,
                "resources_gathered": gathered,
                "resources_returned": resources_returned,
                "resources_spent": resources_spent,
                "resources_carried": resources_carried(p1_units),
                "resource_nodes_remaining": resource_nodes_remaining(all_units),
                "units_total": sum(current_counts.values()),
                "opponent_units_total": sum(opponent_counts.values()),
                "army_value": total_value(p1_units, ARMY_VALUE_TYPES),
                "economy_value": total_value(p1_units, ECONOMY_VALUE_TYPES),
                "infrastructure_value": total_value(p1_units, INFRASTRUCTURE_VALUE_TYPES),
                "total_unit_value": p1_total_value,
                "opponent_total_unit_value": p2_total_value,
                "unit_value_advantage": p1_total_value - p2_total_value,
                "min_worker_distance_to_resource": min_distance_to_units(workers, resource_units),
                "min_combat_distance_to_enemy_base": min_distance_to_units(combat_units, enemy_base_units),
                "min_unit_distance_to_enemy": min_distance_to_units(p1_units, enemy_units),
                "average_distance_to_resource": player_metric(player_1_payload, "average_distance_to_resource", t),
                "shortest_distance_to_resource": player_metric(player_1_payload, "shortest_distance_to_resource", t),
                "distance_to_enemy_base": player_metric(player_1_payload, "distance_to_enemy_base", t),
            }

            for unit_type in TRACKED_UNIT_TYPES:
                name = UNIT_TYPE_NAMES[unit_type]
                row[name] = current_counts[unit_type]
                row[f"opponent_{name}"] = opponent_counts[unit_type]

            row.update(action_counts)
            row.update(event_metrics)
            row.update(damage)
            rows.append(row)

            previous_all = current_all
            previous_p1 = current_p1
            previous_p2 = current_p2
            previous_bank = bank

        episode_time_df = pd.DataFrame(rows)
        time_rows.append(episode_time_df)
        resources_gathered_total = episode_time_df["resources_gathered"].sum()
        resources_returned_total = episode_time_df["resources_returned"].sum()

        episode_summary: dict[str, Any] = {
            "episode_id": episode_id,
            "source_file": str(path),
            "enemy": enemy,
            "result": result,
            "win": win,
            "duration": duration,
            "map_width": int(game_state.get("width", 0)),
            "map_height": int(game_state.get("height", 0)),
            "average_distance_to_resource": episode_time_df["average_distance_to_resource"].mean(),
            "shortest_distance_to_resource": episode_time_df["shortest_distance_to_resource"].min(),
            "distance_to_enemy_base": episode_time_df["distance_to_enemy_base"].mean(),
            "initial_resource_bank": episode_time_df["resource_bank"].iloc[0],
            "final_resource_bank": episode_time_df["resource_bank"].iloc[-1],
            "max_resource_bank": episode_time_df["resource_bank"].max(),
            "resources_gathered_total": resources_gathered_total,
            "resources_returned_total": resources_returned_total,
            "resource_return_success_rate": safe_rate(resources_returned_total, resources_gathered_total),
            "resources_spent_total": episode_time_df["resources_spent"].sum(),
            "units_produced_total": episode_time_df["units_produced"].sum(),
            "units_lost_total": episode_time_df["units_lost"].sum(),
            "units_killed_total": episode_time_df["units_killed"].sum(),
            "damage_dealt_total": episode_time_df["damage_dealt"].sum(),
            "damage_taken_total": episode_time_df["damage_taken"].sum(),
            "value_produced_total": episode_time_df["value_produced"].sum(),
            "value_lost_total": episode_time_df["value_lost"].sum(),
            "value_killed_total": episode_time_df["value_killed"].sum(),
            "max_army_value": episode_time_df["army_value"].max(),
            "max_total_unit_value": episode_time_df["total_unit_value"].max(),
            "final_total_unit_value": episode_time_df["total_unit_value"].iloc[-1],
            "mean_unit_value_advantage": episode_time_df["unit_value_advantage"].mean(),
            "final_unit_value_advantage": episode_time_df["unit_value_advantage"].iloc[-1],
            "time_to_first_harvest": first_nonzero_t(episode_time_df, "action_harvest"),
            "time_to_first_return": first_nonzero_t(episode_time_df, "action_return"),
            "time_to_first_produce": first_nonzero_t(episode_time_df, "action_produce"),
            "time_to_first_attack": first_nonzero_t(episode_time_df, "action_attack"),
            "time_to_first_kill": first_nonzero_t(episode_time_df, "units_killed"),
            "time_to_first_loss": first_nonzero_t(episode_time_df, "units_lost"),
            "time_to_first_combat_unit": first_nonzero_t(
                episode_time_df.assign(
                    combat_produced=(
                        episode_time_df["produced_light"]
                        + episode_time_df["produced_heavy"]
                        + episode_time_df["produced_ranged"]
                    )
                ),
                "combat_produced",
            ),
        }

        for unit_type in TRACKED_UNIT_TYPES:
            name = UNIT_TYPE_NAMES[unit_type]
            episode_summary[f"{name}_produced_total"] = episode_time_df[f"produced_{name}"].sum()
            episode_summary[f"{name}_lost_total"] = episode_time_df[f"lost_{name}"].sum()
            episode_summary[f"{name}_killed_total"] = episode_time_df[f"killed_{name}"].sum()
            episode_summary[f"max_{name}"] = episode_time_df[name].max()
            episode_summary[f"final_{name}"] = episode_time_df[name].iloc[-1]

        for action_name in ACTION_TYPE_NAMES.values():
            episode_summary[f"action_{action_name}_total"] = episode_time_df[f"action_{action_name}"].sum()

        episode_rows.append(episode_summary)

    if not time_rows:
        raise ValueError("No valid episode rows could be built.")

    time_df = pd.concat(time_rows, ignore_index=True)
    episode_df = pd.DataFrame(episode_rows)
    return time_df, episode_df


def save_csvs(time_df: pd.DataFrame, episode_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    time_df.to_csv(output_dir / "timeseries.csv", index=False)
    episode_df.to_csv(output_dir / "episode_summary.csv", index=False)


def save_parquets(time_df: pd.DataFrame, episode_df: pd.DataFrame, output_dir: Path) -> bool:
    ensure_output_dir(output_dir)
    try:
        time_df.to_parquet(output_dir / "timeseries.parquet", index=False)
        episode_df.to_parquet(output_dir / "episode_summary.parquet", index=False)
    except (ImportError, ValueError, ModuleNotFoundError) as e:
        print(f"Skipping Parquet output in {output_dir}: {e}")
        return False
    return True


def aggregate_over_time(time_df: pd.DataFrame, value: str, cumulative: bool = True) -> pd.DataFrame:
    df = time_df.copy()
    y_col = f"cumulative_{value}" if cumulative else value
    if cumulative:
        df[y_col] = df.groupby("episode_id", sort=False)[value].cumsum()

    grouped = (
        df.groupby("t", as_index=False)
        .agg(
            mean=(y_col, "mean"),
            median=(y_col, "median"),
            n=(y_col, "count"),
            std=(y_col, "std"),
        )
        .sort_values("t")
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))
    return grouped


def plot_time_series(
    time_df: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    output_path: Path,
    cumulative: bool = True,
) -> None:
    if time_df.empty or value not in time_df.columns:
        return

    agg = aggregate_over_time(time_df, value, cumulative=cumulative)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = agg["t"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    sem = agg["sem"].fillna(0.0).to_numpy(dtype=float)

    ax.plot(x, y)
    ax.fill_between(x, y - sem, y + sem, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_normalized_time_series(
    time_df: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    output_path: Path,
    cumulative: bool = True,
    bins: int = 50,
) -> None:
    if time_df.empty or value not in time_df.columns:
        return

    df = time_df.copy()
    y_col = f"cumulative_{value}" if cumulative else value
    if cumulative:
        df[y_col] = df.groupby("episode_id", sort=False)[value].cumsum()

    df["progress_bin"] = pd.cut(
        df["progress"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        labels=False,
    )

    grouped = (
        df.groupby("progress_bin", as_index=False)
        .agg(
            progress=("progress", "mean"),
            mean=(y_col, "mean"),
            n=(y_col, "count"),
            std=(y_col, "std"),
        )
        .dropna(subset=["progress"])
        .sort_values("progress")
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["n"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = grouped["progress"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy(dtype=float)
    sem = grouped["sem"].fillna(0.0).to_numpy(dtype=float)

    ax.plot(x, y)
    ax.fill_between(x, y - sem, y + sem, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Normalized game progress")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def pearsonr_safe(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))


def add_regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2:
        return

    x_vals = pair.iloc[:, 0].to_numpy(dtype=float)
    y_vals = pair.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x_vals)) < 2:
        return

    slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, slope * xs + intercept, linestyle="--", linewidth=1.5)


def plot_scatter_correlation(
    episode_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> float:
    if episode_df.empty or x_col not in episode_df.columns or y_col not in episode_df.columns:
        return float("nan")

    corr = pearsonr_safe(episode_df[x_col], episode_df[y_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(episode_df[x_col], episode_df[y_col], alpha=0.75)
    add_regression_line(ax, episode_df[x_col], episode_df[y_col])

    ax.set_title(f"{title}\\nPearson r = {corr:.3f}" if not math.isnan(corr) else title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return corr


def make_quantile_bins(series: pd.Series, bins: int) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.nunique() < 2:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")

    usable_bins = min(bins, clean.nunique())
    try:
        return pd.qcut(series, q=usable_bins, duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=usable_bins, duplicates="drop")


def plot_binned_win_rate(
    episode_df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 8,
) -> float:
    if episode_df.empty or x_col not in episode_df.columns:
        return float("nan")

    df = episode_df[[x_col, "win"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    corr = pearsonr_safe(df[x_col], df["win"])
    if df.empty:
        return corr

    df["bin"] = make_quantile_bins(df[x_col], bins=bins)
    grouped = (
        df.groupby("bin", observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            win_rate=("win", "mean"),
            n=("win", "count"),
        )
        .reset_index(drop=True)
        .sort_values("x_mean")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(grouped["x_mean"], grouped["win_rate"], marker="o")

    for _, row in grouped.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["x_mean"], row["win_rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_title(f"{title}\\nPoint-biserial/Pearson r = {corr:.3f}" if not math.isnan(corr) else title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Win-rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return corr


def plot_enemy_summary(episode_df: pd.DataFrame, output_path: Path) -> None:
    if episode_df.empty or episode_df["enemy"].nunique() < 1:
        return

    grouped = (
        episode_df.groupby("enemy", as_index=False)
        .agg(
            episodes=("episode_id", "count"),
            win_rate=("win", "mean"),
            mean_duration=("duration", "mean"),
            resources_gathered=("resources_gathered_total", "mean"),
            units_killed=("units_killed_total", "mean"),
            units_lost=("units_lost_total", "mean"),
        )
        .sort_values("win_rate", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(grouped))
    ax.bar(x, grouped["win_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["enemy"])
    ax.set_title("Win-rate by enemy")
    ax.set_xlabel("Enemy")
    ax.set_ylabel("Win-rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)

    for idx, row in grouped.iterrows():
        ax.annotate(
            f"n={int(row['episodes'])}",
            (idx, row["win_rate"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_correlation_matrix(episode_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    candidate_cols = [
        "resources_gathered_total",
        "resources_returned_total",
        "resource_return_success_rate",
        "resources_spent_total",
        "units_produced_total",
        "units_lost_total",
        "units_killed_total",
        "damage_dealt_total",
        "damage_taken_total",
        "value_produced_total",
        "value_lost_total",
        "value_killed_total",
        "max_army_value",
        "mean_unit_value_advantage",
        "average_distance_to_resource",
        "shortest_distance_to_resource",
        "distance_to_enemy_base",
        "duration",
        "win",
    ]
    cols = [col for col in candidate_cols if col in episode_df.columns]
    corr_df = episode_df[cols].corr(method="pearson") if cols else pd.DataFrame()

    if corr_df.empty:
        return corr_df

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_df.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr_df.iloc[i, j]
            if not math.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Episode-level Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return corr_df


def create_visualizations(
    time_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    output_dir: Path,
    bins: int,
) -> pd.DataFrame:
    ensure_output_dir(output_dir)

    raw_time_dir = output_dir / "raw_time"
    normalized_time_dir = output_dir / "normalized_time"
    ensure_output_dir(raw_time_dir)
    ensure_output_dir(normalized_time_dir)

    time_series_tasks = [
        ("resources_gathered", "Resources gathered over time", "Mean cumulative resources gathered", True),
        ("resources_returned", "Resources returned to bank over time", "Mean cumulative resources returned", True),
        ("resources_spent", "Resources spent over time", "Mean cumulative resources spent", True),
        ("units_produced", "Units produced over time", "Mean cumulative units produced", True),
        ("units_lost", "Units lost over time", "Mean cumulative units lost", True),
        ("units_killed", "Units killed over time", "Mean cumulative units killed", True),
        ("damage_dealt", "Damage dealt over time", "Mean cumulative damage dealt", True),
        ("damage_taken", "Damage taken over time", "Mean cumulative damage taken", True),
        ("resource_bank", "Resource bank over time", "Mean resource bank", False),
        ("resources_carried", "Resources carried over time", "Mean resources carried", False),
        ("army_value", "Army value over time", "Mean army value", False),
        ("total_unit_value", "Total unit value over time", "Mean total unit value", False),
        ("unit_value_advantage", "Unit value advantage over time", "Mean unit value advantage", False),
    ]

    for value, title, ylabel, cumulative in tqdm(
        time_series_tasks,
        desc=f"Plotting time series: {output_dir.name}",
        unit="plot",
        leave=False,
    ):
        plot_time_series(
            time_df,
            value=value,
            title=title,
            ylabel=ylabel,
            output_path=raw_time_dir / f"{value}_over_time.png",
            cumulative=cumulative,
        )
        plot_normalized_time_series(
            time_df,
            value=value,
            title=title.replace("over time", "over normalized game progress"),
            ylabel=ylabel,
            output_path=normalized_time_dir / f"{value}_over_normalized_time.png",
            cumulative=cumulative,
        )

    correlation_tasks = [
        (
            "resources_gathered_vs_average_distance_to_resource",
            "scatter",
            {
                "x_col": "average_distance_to_resource",
                "y_col": "resources_gathered_total",
                "title": "Resources gathered vs average distance to resources",
                "xlabel": "Average base-to-resource distance",
                "ylabel": "Total resources gathered",
                "output_path": output_dir / "corr_resources_gathered_vs_avg_resource_distance.png",
            },
        ),
        (
            "resources_gathered_vs_shortest_distance_to_resource",
            "scatter",
            {
                "x_col": "shortest_distance_to_resource",
                "y_col": "resources_gathered_total",
                "title": "Resources gathered vs shortest distance to resources",
                "xlabel": "Shortest base-to-resource distance",
                "ylabel": "Total resources gathered",
                "output_path": output_dir / "corr_resources_gathered_vs_shortest_resource_distance.png",
            },
        ),
        (
            "resources_gathered_vs_win",
            "binned_win_rate",
            {
                "x_col": "resources_gathered_total",
                "title": "Win-rate vs resources gathered",
                "xlabel": "Total resources gathered",
                "output_path": output_dir / "corr_resources_gathered_vs_win_rate.png",
            },
        ),
        (
            "resource_return_success_rate_vs_win",
            "binned_win_rate",
            {
                "x_col": "resource_return_success_rate",
                "title": "Win-rate vs resource return success rate",
                "xlabel": "Returned / harvested resources",
                "output_path": output_dir / "corr_resource_return_success_rate_vs_win_rate.png",
            },
        ),
        (
            "resources_spent_vs_win",
            "binned_win_rate",
            {
                "x_col": "resources_spent_total",
                "title": "Win-rate vs resources spent",
                "xlabel": "Total resources spent",
                "output_path": output_dir / "corr_resources_spent_vs_win_rate.png",
            },
        ),
        (
            "average_distance_to_resource_vs_win",
            "binned_win_rate",
            {
                "x_col": "average_distance_to_resource",
                "title": "Win-rate vs average distance to resources",
                "xlabel": "Average base-to-resource distance",
                "output_path": output_dir / "corr_avg_resource_distance_vs_win_rate.png",
            },
        ),
        (
            "shortest_distance_to_resource_vs_win",
            "binned_win_rate",
            {
                "x_col": "shortest_distance_to_resource",
                "title": "Win-rate vs shortest distance to resources",
                "xlabel": "Shortest base-to-resource distance",
                "output_path": output_dir / "corr_shortest_resource_distance_vs_win_rate.png",
            },
        ),
        (
            "units_produced_vs_win",
            "binned_win_rate",
            {
                "x_col": "units_produced_total",
                "title": "Win-rate vs units produced",
                "xlabel": "Total units produced",
                "output_path": output_dir / "corr_units_produced_vs_win_rate.png",
            },
        ),
        (
            "units_killed_vs_win",
            "binned_win_rate",
            {
                "x_col": "units_killed_total",
                "title": "Win-rate vs units killed",
                "xlabel": "Total units killed",
                "output_path": output_dir / "corr_units_killed_vs_win_rate.png",
            },
        ),
        (
            "value_killed_vs_win",
            "binned_win_rate",
            {
                "x_col": "value_killed_total",
                "title": "Win-rate vs value killed",
                "xlabel": "Total value killed",
                "output_path": output_dir / "corr_value_killed_vs_win_rate.png",
            },
        ),
        (
            "duration_vs_win",
            "binned_win_rate",
            {
                "x_col": "duration",
                "title": "Win-rate over game duration",
                "xlabel": "Game duration / episode length",
                "output_path": output_dir / "win_rate_over_game_duration.png",
            },
        ),
    ]

    correlations: dict[str, float] = {}
    for metric, kind, kwargs in tqdm(
        correlation_tasks,
        desc=f"Plotting correlations: {output_dir.name}",
        unit="plot",
        leave=False,
    ):
        if kind == "scatter":
            correlations[metric] = plot_scatter_correlation(episode_df, **kwargs)
        elif kind == "binned_win_rate":
            correlations[metric] = plot_binned_win_rate(episode_df, bins=bins, **kwargs)
        else:
            raise ValueError(f"Unknown correlation plot kind: {kind}")

    for _ in tqdm(range(1), desc=f"Plotting matrix: {output_dir.name}", unit="plot", leave=False):
        corr_matrix = plot_correlation_matrix(episode_df, output_dir / "correlation_matrix.png")
        corr_matrix.to_csv(output_dir / "correlation_matrix.csv")

    correlations_df = pd.DataFrame(
        [{"metric": metric, "pearson_r": value} for metric, value in correlations.items()]
    )
    correlations_df.to_csv(output_dir / "selected_correlations.csv", index=False)
    return correlations_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read rich episode-statistics JSON files and generate player-1 analysis plots."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing episode statistics .json or .json.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("episode_stats_plots"),
        help="Directory where plots and CSV summaries will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json*",
        help="Glob pattern used recursively under input_dir. Default: *.json*",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of quantile bins for win-rate plots. Default: 8.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip optional Parquet tables. CSV tables are always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    episodes = load_episode_jsons(args.input_dir, pattern=args.pattern)
    time_df, episode_df = build_frames(episodes)

    save_csvs(time_df, episode_df, args.output_dir)
    wrote_parquet = False
    if not args.no_parquet:
        wrote_parquet = save_parquets(time_df, episode_df, args.output_dir)

    all_correlation_frames: list[pd.DataFrame] = []

    overall_dir = args.output_dir / "overall"
    ensure_output_dir(overall_dir)
    plot_enemy_summary(episode_df, overall_dir / "win_rate_by_enemy.png")

    overall_correlations_df = create_visualizations(
        time_df=time_df,
        episode_df=episode_df,
        output_dir=overall_dir,
        bins=args.bins,
    )
    overall_correlations_df.insert(0, "enemy", "ALL")
    all_correlation_frames.append(overall_correlations_df)

    enemy_root = args.output_dir / "by_enemy"
    ensure_output_dir(enemy_root)

    enemy_groups = list(episode_df.groupby("enemy", dropna=False))
    for enemy, enemy_episode_df in tqdm(enemy_groups, desc="Creating per-enemy outputs", unit="enemy"):
        enemy_time_df = time_df[time_df["enemy"] == enemy].copy()
        enemy_dir = enemy_root / safe_filename(enemy)
        ensure_output_dir(enemy_dir)

        save_csvs(enemy_time_df, enemy_episode_df.copy(), enemy_dir)
        if not args.no_parquet and wrote_parquet:
            save_parquets(enemy_time_df, enemy_episode_df.copy(), enemy_dir)
        enemy_correlations_df = create_visualizations(
            time_df=enemy_time_df,
            episode_df=enemy_episode_df.copy(),
            output_dir=enemy_dir,
            bins=args.bins,
        )
        enemy_correlations_df.insert(0, "enemy", str(enemy) if str(enemy).strip() else "unknown_enemy")
        all_correlation_frames.append(enemy_correlations_df)

    correlations_df = pd.concat(all_correlation_frames, ignore_index=True)
    correlations_df.to_csv(args.output_dir / "selected_correlations_by_enemy.csv", index=False)

    summary_by_enemy = (
        episode_df.groupby("enemy", as_index=False)
        .agg(
            episodes=("episode_id", "count"),
            win_rate=("win", "mean"),
            mean_duration=("duration", "mean"),
            mean_resources_gathered=("resources_gathered_total", "mean"),
            mean_resource_return_success_rate=("resource_return_success_rate", "mean"),
            mean_resources_spent=("resources_spent_total", "mean"),
            mean_units_produced=("units_produced_total", "mean"),
            mean_units_killed=("units_killed_total", "mean"),
            mean_units_lost=("units_lost_total", "mean"),
            mean_damage_dealt=("damage_dealt_total", "mean"),
            mean_damage_taken=("damage_taken_total", "mean"),
        )
        .sort_values("enemy")
    )
    summary_by_enemy.to_csv(args.output_dir / "summary_by_enemy.csv", index=False)

    print(f"Loaded episodes: {len(episodes)}")
    print(f"Timeseries rows: {len(time_df)}")
    print(f"Episode rows: {len(episode_df)}")
    print(f"Enemies: {', '.join(sorted(str(e) for e in episode_df['enemy'].dropna().unique()))}")
    print(f"Wrote outputs to: {args.output_dir.resolve()}")
    print(f"Parquet tables: {'written' if wrote_parquet else 'not written'}")
    print("Summary by enemy:")
    print(summary_by_enemy.to_string(index=False))
    print("Selected correlations by enemy:")
    print(correlations_df.to_string(index=False))


if __name__ == "__main__":
    main()
