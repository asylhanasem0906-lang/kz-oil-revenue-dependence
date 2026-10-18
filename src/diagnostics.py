
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file).sort_values("Year")

y = df["Y (%)"]

X = df[
    [
        "Brent annual avg (USD/bbl)",
        "Oil incl condensate (million t)"
    ]
]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

# 1. Autocorrelation test
bg_stat, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=1)

# 2. Heteroskedasticity test
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)

# 3. Multicollinearity
vif = [
    variance_inflation_factor(X.values, i)
    for i in range(1, X.shape[1])
]

print("\nMODEL DIAGNOSTICS")
print(f"Breusch-Godfrey p-value: {bg_p:.4f}")
print(f"Breusch-Pagan p-value: {bp_p:.4f}")

for name, value in zip(X.columns[1:], vif):
    print(f"VIF for {name}: {value:.2f}")

print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(model.resid):.3f}")
