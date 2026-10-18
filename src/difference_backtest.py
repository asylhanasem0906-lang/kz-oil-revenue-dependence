
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

project = Path(__file__).resolve().parent.parent

file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year").copy()

Y = "Y (%)"
BRENT = "Brent annual avg (USD/bbl)"

df["dY"] = df[Y].diff()
df["dBrent"] = df[BRENT].diff()

# Use previous year's Brent change
df["dBrent_lag1"] = df["dBrent"].shift(1)
df["Y_lag1"] = df[Y].shift(1)

results = []

for year in range(2022, 2026):
    train = df[df["Year"] < year].dropna(
        subset=["dY", "dBrent_lag1"]
    )

    test = df[df["Year"] == year]

    X_train = sm.add_constant(
        train[["dBrent_lag1"]], has_constant="add"
    )

    model = sm.OLS(train["dY"], X_train).fit()

    X_test = sm.add_constant(
        test[["dBrent_lag1"]], has_constant="add"
    )

    predicted_change = float(
        model.predict(X_test).iloc[0]
    )

    previous_y = float(test["Y_lag1"].iloc[0])
    predicted_y = previous_y + predicted_change

    results.append({
        "Year": year,
        "Actual": float(test[Y].iloc[0]),
        "Naive": previous_y,
        "Difference Model": predicted_y
    })

results = pd.DataFrame(results)

print("\nFIRST-DIFFERENCE FORECAST BACKTEST")
print(results.round(4).to_string(index=False))

print("\nFORECAST ERRORS")

for name in ["Naive", "Difference Model"]:
    errors = results["Actual"] - results[name]

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

outputs = project / "outputs"
outputs.mkdir(exist_ok=True)

results.to_csv(
    outputs / "difference_backtest.csv",
    index=False
)

print("\nResults saved successfully.")
