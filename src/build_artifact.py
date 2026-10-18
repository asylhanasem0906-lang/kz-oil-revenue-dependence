
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan
)

project = Path(__file__).resolve().parent.parent

file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year").copy()

df["dY"] = df["Y (%)"].diff()
df["dBrent"] = df["Brent annual avg (USD/bbl)"].diff()

reg = df.dropna(subset=["dY", "dBrent"])

X = sm.add_constant(reg[["dBrent"]])
y = reg["dY"]

model = sm.OLS(y, X).fit()

bg = acorr_breusch_godfrey(model, nlags=1)
bp = het_breuschpagan(model.resid, X)

# Load previously generated forecast results
forecast_file = project / "outputs" / "difference_backtest.csv"
backtest = pd.read_csv(forecast_file)

def metrics(name):
    errors = backtest["Actual"] - backtest[name]
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2)))
    }

artifact = {
    "model_name": "First-Difference Brent Regression",
    "indicator_name": "Oil Tax Revenue-to-State Budget Income Ratio",
    "indicator_unit": "percent",
    "model_target": "annual change in indicator, percentage points",
    "frequency": "annual",
    "sample_start": int(reg["Year"].min()),
    "sample_end": int(reg["Year"].max()),
    "n_observations": int(len(reg)),
    "estimated_at_utc": datetime.now(timezone.utc).isoformat(),
    "coefficients": {
        k: float(v) for k, v in model.params.items()
    },
    "standard_errors": {
        k: float(v) for k, v in model.bse.items()
    },
    "covariance_matrix": model.cov_params().values.tolist(),
    "r_squared": float(model.rsquared),
    "diagnostics": {
        "breusch_godfrey_p": float(bg[1]),
        "breusch_pagan_p": float(bp[1]),
        "durbin_watson": float(
            sm.stats.stattools.durbin_watson(model.resid)
        )
    },
    "forecast_test": {
        "test_years": [2022, 2023, 2024, 2025],
        "naive": metrics("Naive"),
        "difference_model_lagged_brent": metrics("Difference Model")
    },
    "limitations": [
        "19 annual observations in the original dataset.",
        "The indicator covers National Fund oil taxes relative to state budget income.",
        "The regression describes same-year associations, not causality.",
        "The naive benchmark has lower out-of-sample RMSE."
    ]
}

out_dir = project / "model"
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "artifact.json"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2, ensure_ascii=False)

print("\nMODEL ARTIFACT CREATED!")
print("File:", out_file)
print("R-squared:", round(model.rsquared, 4))
print("Brent coefficient:", round(model.params["dBrent"], 4))
print("Benchmark RMSE:", round(metrics("Naive")["rmse"], 4))
