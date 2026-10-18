
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson

project = Path(__file__).resolve().parent.parent

file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year")

y = df["Y (%)"]

X = sm.add_constant(df[
    [
        "Brent annual avg (USD/bbl)",
        "Oil incl condensate (million t)"
    ]
])

# Estimate a regression with AR(1) errors
model = sm.GLSAR(y, X, rho=1)
results = model.iterative_fit(maxiter=20)

print("\nGLSAR(1) RESULTS")
print(results.summary())

print("\nEstimated rho:", model.rho[0])

# Check the approximate innovations, rather than
# treating the raw GLSAR residuals as independent.
rho = float(model.rho[0])
resid = results.resid.to_numpy()
innovations = resid[1:] - rho * resid[:-1]

print("Durbin-Watson (innovations):",
      round(durbin_watson(innovations), 4))

print("\nIMPORTANT:")
print("Residual diagnostics require further validation.")
print("GLSAR is not automatically the final model.")
