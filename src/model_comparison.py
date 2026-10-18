
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load dataset
project = Path(__file__).resolve().parent.parent
file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year").reset_index(drop=True)

Y = "Y (%)"
BRENT = "Brent annual avg (USD/bbl)"
OIL = "Oil incl condensate (million t)"

# Use previous year's information for forecasting
df["Y_lag1"] = df[Y].shift(1)
df["Brent_lag1"] = df[BRENT].shift(1)
df["Oil_lag1"] = df[OIL].shift(1)

features = ["Brent_lag1", "Oil_lag1"]
dynamic_features = ["Y_lag1", "Brent_lag1", "Oil_lag1"]

results = []

# Expanding-window backtest, 2022-2025
for year in range(2022, 2026):
    train = df[df["Year"] < year].dropna(
        subset=[Y] + dynamic_features
    )
    test = df[df["Year"] == year]

    actual = float(test[Y].iloc[0])
    naive = float(test["Y_lag1"].iloc[0])

    # OLS with lagged economic factors
    X_train = sm.add_constant(
        train[features], has_constant="add"
    )
    X_test = sm.add_constant(
        test[features], has_constant="add"
    )

    ols = sm.OLS(train[Y], X_train).fit()
    ols_pred = float(ols.predict(X_test).iloc[0])

    # Dynamic OLS with previous year's Y
    X_dyn_train = sm.add_constant(
        train[dynamic_features], has_constant="add"
    )
    X_dyn_test = sm.add_constant(
        test[dynamic_features], has_constant="add"
    )

    dynamic = sm.OLS(train[Y], X_dyn_train).fit()
    dynamic_pred = float(
        dynamic.predict(X_dyn_test).iloc[0]
    )

    # GLSAR with lagged economic factors
    glsar = sm.GLSAR(train[Y], X_train, rho=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        glsar_result = glsar.iterative_fit(maxiter=20)

    glsar_pred = float(
        glsar_result.predict(X_test).iloc[0]
    )

    results.append({
        "Year": year,
        "Actual": actual,
        "Naive": naive,
        "OLS": ols_pred,
        "Dynamic OLS": dynamic_pred,
        "GLSAR": glsar_pred
    })

results_df = pd.DataFrame(results)

print("\nOUT-OF-SAMPLE FORECASTS (2022-2025)")
print(results_df.round(3).to_string(index=False))

print("\nFORECAST ACCURACY")
scores = []

for name in ["Naive", "OLS", "Dynamic OLS", "GLSAR"]:
    errors = results_df["Actual"] - results_df[name]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    scores.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

scores_df = pd.DataFrame(scores)
naive_rmse = float(
    scores_df.loc[
        scores_df["Model"] == "Naive", "RMSE"
    ].iloc[0]
)

scores_df["RMSE / Naive"] = (
    scores_df["RMSE"] / naive_rmse
)

print(scores_df.round(4).to_string(index=False))

# Save results for the website and technical note
outputs = project / "outputs"
outputs.mkdir(exist_ok=True)

results_df.to_csv(
    outputs / "backtest_forecasts.csv", index=False
)
scores_df.to_csv(
    outputs / "backtest_scores.csv", index=False
)

print("\nResults saved in outputs/")
print("Note: This compares models using lagged predictors.")
