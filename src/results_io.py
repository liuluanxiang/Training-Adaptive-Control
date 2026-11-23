"""Unified experiment-result I/O for the flat ``experiment/method`` layout.

Archive and training outputs use::

    {results_dir}/{experiment}/{method}/history.csv
    {results_dir}/{experiment}/{method}/summary.json

There is no ``seed_*`` directory level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def infer_experiment_type(experiment_name):
    """Infer experiment type from legacy names; new runs should store it explicitly."""

    name = str(experiment_name or "").lower()
    if name.startswith("quick"):
        return "quick"
    if name.startswith("main"):
        return "main"
    if name.startswith("supp"):
        return "supplement"
    if (
        name.startswith("ablation")
        or name.startswith("controller_ablation")
        or name.startswith("pd_component")
    ):
        return "ablation"
    if name.startswith("sens") or name.startswith("sensitivity") or name.startswith("sigma"):
        return "sensitivity"
    return "unspecified"


def ensure_experiment_type(df):
    """Fill ``experiment_type`` without mutating the caller DataFrame."""

    if df is None or df.empty:
        return df
    out = df.copy()
    if "experiment_type" not in out.columns:
        out["experiment_type"] = None
    if "experiment" in out.columns:
        missing = out["experiment_type"].isna() | (out["experiment_type"].astype(str).str.strip() == "")
        out.loc[missing, "experiment_type"] = out.loc[missing, "experiment"].map(infer_experiment_type)
    else:
        out["experiment_type"] = out["experiment_type"].fillna("unspecified")
    return out


def filter_quick(df, include_quick=False):
    """By default exclude quick smoke runs from paper-facing tables."""

    out = ensure_experiment_type(df)
    if out is None or out.empty or include_quick:
        return out
    return out[out["experiment_type"] != "quick"].copy()


def collect_histories(results_dir, include_quick=False):
    """Aggregate per-epoch histories from ``{experiment}/{method}/history.csv``."""

    frames = []
    for path in Path(results_dir).glob("*/*/history.csv"):
        method_dir = path.parent
        exp_dir = method_dir.parent
        # Skip any accidental nested seed directories if present on disk.
        if method_dir.name.startswith("seed_"):
            continue
        try:
            df = pd.read_csv(path)
            if "experiment" not in df.columns:
                df["experiment"] = exp_dir.name
            if "method" not in df.columns:
                df["method"] = method_dir.name
            if "experiment_type" not in df.columns:
                df["experiment_type"] = infer_experiment_type(df["experiment"].iloc[0])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return filter_quick(pd.concat(frames, ignore_index=True), include_quick=include_quick)


def collect_summaries(results_dir, include_quick=False):
    """Prefer the master summary table; otherwise rebuild from flat ``summary.json`` files."""

    results_dir = Path(results_dir)
    master = results_dir / "master_summary.csv"
    if master.exists():
        try:
            return filter_quick(pd.read_csv(master), include_quick=include_quick)
        except Exception:
            pass

    rows = []
    for path in results_dir.glob("*/*/summary.json"):
        method_dir = path.parent
        exp_dir = method_dir.parent
        if method_dir.name.startswith("seed_"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row.setdefault("experiment", exp_dir.name)
            row.setdefault("method", method_dir.name)
            row.setdefault("experiment_type", infer_experiment_type(row.get("experiment")))
            rows.append(row)
        except Exception:
            continue
    return filter_quick(pd.DataFrame(rows), include_quick=include_quick)
