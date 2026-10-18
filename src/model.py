
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

# Locate project and dataset
project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file).sort_values("Year")

# Variables
y = df["Y (%)"]
X = df[
    [
        "Brent annual avg (USD/bbl)",
        "Oil incl condensate (million t)"
    ]
]

# Add intercept
X = sm.add_constant(X)

# Fit ordinary least squares regression
model = sm.OLS(y, X).fit()

# Display results
print("\nOIL-REVENUE DEPENDENCE REGRESSION")
print(model.summary())
