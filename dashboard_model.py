"""Small, explicit data layer for measured experiment points."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dashboard_data" / "experiments.json"
METRICS = {
    "current": "Current finite-blocklength sum-rate",
    "effective": "Effective sum-rate · original model",
    "modeled": "Budget-adjusted rate · sensitivity model",
}


def load_experiments(path=DATA_PATH):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or not payload.get("scenarios"):
        raise ValueError("Unsupported schema or empty experiment results.")
    for scenario in payload["scenarios"]:
        data = frame(scenario)
        if not np.isfinite(scenario["T_ms"]) or scenario["T_ms"] <= 0:
            raise ValueError("T must be positive and finite")
        required = ["t1", "t2", *scenario["metrics"]]
        if data.empty or not np.isfinite(data[required].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric results: {scenario['id']}")
        if ((data[["t1", "t2"]] < 0).any().any()
                or (data["total"] > 100 + 1e-8).any()
                or data.duplicated(["t1", "t2"]).any()):
            raise ValueError(f"Invalid or duplicate allocations: {scenario['id']}")
        if scenario["experiment"] == "fixed_bank":
            grid = data.pivot(index="t1", columns="t2", values="current").sort_index().sort_index(axis=1).to_numpy()
            if (not np.isfinite(grid).all() or (np.diff(grid, axis=0) < -1e-8).any()
                    or (np.diff(grid, axis=1) < -1e-8).any()):
                raise ValueError(f"Incomplete or nonmonotone fixed-candidate grid: {scenario['id']}")
    return payload["scenarios"]


def frame(scenario):
    data = pd.DataFrame(scenario["points"])
    data["t1"] = data["t1"].round(8)
    data["t2"] = data["t2"].round(8)
    data["total"] = (data["t1"] + data["t2"]).round(8)
    return data


def select_point(data, t1, t2):
    selected = data[np.isclose(data.t1, t1) & np.isclose(data.t2, t2)]
    if len(selected) != 1:
        raise ValueError("No saved experiment point exists for this allocation.")
    return selected.iloc[0]


def best_point(data, metric, budget=None):
    feasible = data if budget is None else data[data.total <= budget + 1e-8]
    if feasible.empty:
        raise ValueError("No saved points are feasible within this budget.")
    return feasible.sort_values([metric, "total", "t1"], ascending=[False, True, True]).iloc[0]
