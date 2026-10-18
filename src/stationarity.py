
from pathlib import Path
import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

project = Path(__file__).resolve().parent.parent
file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year")

variables = {
    "Oil Dependence Y": "Y (%)",
    "Brent": "Brent annual avg (USD/bbl)",
    "Oil Production": "Oil incl condensate (million t)",
    "USD/KZT": "USD/KZT annual avg"
}

print("\nSTATIONARITY TESTS")
print("ADF H0: Unit root (non-stationary)")
print("KPSS H0: Stationary around a constant")
print("Significance level: 5%")

for name, column in variables.items():
    series = df[column].dropna().astype(float)

    print(f"\n{name}")

    try:
        adf_result = adfuller(
            series, regression="c", autolag="AIC"
        )
        print("ADF p-value:", round(adf_result[1], 4))
        print("ADF lags:", adf_result[2])
    except Exception as e:
        print("ADF failed:", str(e))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(
                series, regression="c", nlags="auto"
            )
        print("KPSS p-value:", round(kpss_result[1], 4))
    except Exception as e:
        print("KPSS failed:", str(e))

print("\nIMPORTANT:")
print("Small annual samples make these tests uncertain.")
print("We must also check trends and structural breaks.")
