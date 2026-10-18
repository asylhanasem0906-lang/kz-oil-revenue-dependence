
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Load data
project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file).sort_values("Year").copy()

# Previous year's oil-dependence indicator
df["Y_lag1"] = df["Y (%)"].shift(1)

# Only remove rows missing variables required by this model
variables = [
    "Y (%)",
    "Y_lag1",
    "Brent annual avg (USD/bbl)",
    "Oil incl condensate (million t)"
]
df = df.dropna(subset=variables)

# Set up regression
y = df["Y (%)"]
X = sm.add_constant(
    df[
        [
            "Y_lag1",
            "Brent annual avg (USD/bbl)",
            "Oil incl condensate (million t)"
        ]
    ]
)

model = sm.OLS(y, X).fit()

print("\nDYNAMIC REGRESSION RESULTS")
print(model.summary())

# Autocorrelation test
bg_result = acorr_breusch_godfrey(
    model, nlags=1, result_object=True
)

print("\nMODEL CHECK")
print("Observations:", len(df))
print("R-squared:", round(model.rsquared, 4))
print("Breusch-Godfrey p-value:", round(bg_result.lmpval, 4))
print("Durbin-Watson:", round(
    sm.stats.stattools.durbin_watson(model.resid), 4
))
