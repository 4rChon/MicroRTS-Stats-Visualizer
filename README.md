# OCGDT Stats Visualizer

Generate precomputed statistics from rich per-episode `.json` or `.json.gz`
logs, then inspect them in a Streamlit + Plotly dashboard.

## Build Tables

```powershell
uv run python main.py data --output-dir stats_viz
```

The visualizer writes Parquet tables by default. Add `--write-csv` when you
also want CSV exports for inspection or sharing.

## Run Dashboard

```powershell
uv run streamlit run dashboard.py
```

The dashboard reads `stats_viz` tables from Parquet when present, otherwise
CSV. It does not parse raw episode logs on UI interactions.

The Inference tab runs statistical tests on filtered episode-level summary rows.
It does not treat timestep rows as independent observations.

The Trajectory mode in the Inference tab summarizes each episode inside
normalized progress bins before testing differences over time.

Win score uses the encoded outcome value where wins are `1.0`, draws are `0.5`,
and losses are `0.0`. Win rate excludes draws. Resource return success is
computed from issued commands as return actions divided by harvest actions.
