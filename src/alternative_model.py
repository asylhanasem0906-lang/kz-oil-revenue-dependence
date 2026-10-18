
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

project = Path(__file__).resolve().parent.parent

file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year")

y = df["Y (%)"]

# Original specification
X1 = sm.add_constant(df[
    [
        "Brent annual avg (USD/bbl)",
        "Oil incl condensate (million t)"
    ]
])

# Alternative specification: add USD/KZT
X2 = sm.add_constant(df[
    [
        "Brent annual avg (USD/bbl)",
        "Oil incl condensate (million t)",
        "USD/KZT annual avg"
    ]
])

model1 = sm.OLS(y, X1).fit()
model2 = sm.OLS(y, X2).fit()

print("\nALTERNATIVE SPECIFICATION CHECK")

for name, model in [
    ("Original Model", model1),
    ("Alternative Model + USD/KZT", model2)
]:
    print(f"\n{name}")
    print("R-squared:", round(model.rsquared, 4))
    print("Adjusted R-squared:", round(model.rsquared_adj, 4))
    print("AIC:", round(model.aic, 2))

    print("\nCoefficients:")
    print(model.params.round(4).to_string())

    print("\nP-values:")
    print(model.pvalues.round(4).to_string())
