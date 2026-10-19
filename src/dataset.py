"""Canonical processed dataset adapter. All models and the dashboard use master_dataset.csv."""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "processed" / "master_dataset.csv"


def load_dataset():
    raw = pd.read_csv(DATASET)
    required = ["year", "Y_ratio_pct", "brent_usd_per_bbl", "oil_incl_condensate_mln_t"]
    missing = [col for col in required if col not in raw]
    if missing:
        raise ValueError(f"Missing columns in canonical dataset: {missing}")
    if raw[required].isna().any().any() or raw["year"].duplicated().any():
        raise ValueError("Canonical dataset has missing values or duplicate years")
    return raw.rename(columns={
        "year": "Year",
        "Y_ratio_pct": "Y (%)",
        "brent_usd_per_bbl": "Brent annual avg (USD/bbl)",
        "oil_incl_condensate_mln_t": "Oil incl condensate (million t)",
    }).sort_values("Year").reset_index(drop=True)
