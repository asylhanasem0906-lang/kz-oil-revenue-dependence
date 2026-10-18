
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f

project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file).sort_values("Year")

features = [
    "Brent annual avg (USD/bbl)",
    "Oil incl condensate (million t)"
]
target = "Y (%)"

def fit(data):
    X = sm.add_constant(data[features], has_constant="add")
    y = data[target]
    return sm.OLS(y, X).fit()

full = fit(df)
k = len(full.params)

print("STRUCTURAL BREAK TESTS")

for year in [2015, 2020, 2022]:
    before = df[df["Year"] < year]
    after = df[df["Year"] >= year]

    if min(len(before), len(after)) <= k:
        print(f"{year}: insufficient observations")
        continue

    m1 = fit(before)
    m2 = fit(after)

    rss_full = np.sum(full.resid ** 2)
    rss_split = np.sum(m1.resid ** 2) + np.sum(m2.resid ** 2)

    n = len(df)
    df2 = n - 2 * k

    F = ((rss_full - rss_split) / k) / (rss_split / df2)
    p = f.sf(F, k, df2)

    print(f"{year}: F={F:.4f}, p-value={p:.4f}")
    print("Possible break" if p < 0.05 else "No significant break")
