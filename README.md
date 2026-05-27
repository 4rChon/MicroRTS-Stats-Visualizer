# OCGDT Stats Visualizer

Generate precomputed statistics from rich per-episode `.json` or `.json.gz`
logs, then inspect them in a Streamlit + Plotly dashboard.

## Build Tables And Plots

```powershell
uv run python main.py data --output-dir stats_viz
```

The visualizer writes CSV summaries, Parquet tables when `pyarrow` is
available, and PNG plots split into overall and per-enemy outputs.

## Run Dashboard

```powershell
uv run streamlit run dashboard.py
```

The dashboard reads `stats_viz/episode_summary` and `stats_viz/timeseries`
from Parquet when present, otherwise CSV. It does not parse raw episode logs on
UI interactions.

The Inference tab runs statistical tests on filtered episode-level summary rows.
It does not treat timestep rows as independent observations.

The Trajectory mode in the Inference tab summarizes each episode inside
normalized progress bins before testing differences over time.
