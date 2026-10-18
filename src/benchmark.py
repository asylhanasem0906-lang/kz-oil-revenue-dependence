
from pathlib import Path
import pandas as pd
import numpy as np

project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file).sort_values("Year").copy()

# Benchmark: next year's Y equals previous year's Y
df["Naive_Forecast"] = df["Y (%)"].shift(1)

# Test period: last four years
test = df[df["Year"].between(2022, 2025)].copy()

actual = test["Y (%)"].to_numpy()
forecast = test["Naive_Forecast"].to_numpy()

mae = np.mean(np.abs(actual - forecast))
rmse = np.sqrt(np.mean((actual - forecast) ** 2))

print("\nNAIVE BENCHMARK — 2022–2025")
print(test[["Year", "Y (%)", "Naive_Forecast"]].round(4).to_string(index=False))
print(f"\nMAE: {mae:.4f} percentage points")
print(f"RMSE: {rmse:.4f} percentage points")
