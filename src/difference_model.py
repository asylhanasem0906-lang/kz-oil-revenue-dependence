
from pathlib import Path
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

# Calculate annual changes
df["dY"] = df["Y (%)"].diff()
df["dBrent"] = df["Brent annual avg (USD/bbl)"].diff()

df = df.dropna(subset=["dY", "dBrent"])

# Estimate model in first differences
X = sm.add_constant(df[["dBrent"]])
y = df["dY"]

model = sm.OLS(y, X).fit()

print("\nFIRST-DIFFERENCE REGRESSION")
print(model.summary())

# Residual diagnostics
bg = acorr_breusch_godfrey(model, nlags=1)
bp = het_breuschpagan(model.resid, X)

print("\nDIAGNOSTICS")
print("Breusch-Godfrey p-value:", round(bg[1], 4))
print("Breusch-Pagan p-value:", round(bp[1], 4))
print(
    "Durbin-Watson:",
    round(sm.stats.stattools.durbin_watson(model.resid), 4)
)
