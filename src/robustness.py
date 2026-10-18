
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

project = Path(__file__).resolve().parent.parent

file = (
    project / "data" / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

df = pd.read_excel(file).sort_values("Year")

Y = "Y (%)"
features = [
    "Brent annual avg (USD/bbl)",
    "Oil incl condensate (million t)"
]

def fit_model(data, label):
    X = sm.add_constant(data[features])
    y = data[Y]

    model = sm.OLS(y, X).fit()

    print(f"\n{label}")
    print("Observations:", len(data))
    print("R-squared:", round(model.rsquared, 4))

    for name in features:
        print(name, ":", round(model.params[name], 4))

# Full period
fit_model(df, "FULL SAMPLE: 2007-2025")

# Excluding COVID year
fit_model(
    df[df["Year"] != 2020],
    "EXCLUDING 2020"
)

# Later starting year
fit_model(
    df[df["Year"] >= 2012],
    "SAMPLE: 2012-2025"
)
