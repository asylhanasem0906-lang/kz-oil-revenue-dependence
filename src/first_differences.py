
from pathlib import Path
import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

project = Path(__file__).resolve().parent.parent
file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year").copy()

variables = {
    "Oil Dependence Y": "Y (%)",
    "Brent": "Brent annual avg (USD/bbl)",
    "Oil Production": "Oil incl condensate (million t)",
    "USD/KZT": "USD/KZT annual avg"
}

print("\nFIRST-DIFFERENCE STATIONARITY TESTS")
print("ADF H0: Unit root")
print("KPSS H0: Stationary")

for name, column in variables.items():
    # Change from the previous year
    changes = df[column].astype(float).diff().dropna()

    print(f"\n{name}")
    print("Observations:", len(changes))

    try:
        # Fixed short lag because the sample is small
        adf = adfuller(
            changes, maxlag=1, autolag=None,
            regression="c"
        )
        print("ADF p-value:", round(adf[1], 4))
    except Exception as e:
        print("ADF error:", e)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kp = kpss(
                changes, regression="c", nlags=1
            )
        print("KPSS p-value:", round(kp[1], 4))
    except Exception as e:
        print("KPSS error:", e)

print("\nNOTE: Results are preliminary with 18 observations.")
