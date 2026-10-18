
from pathlib import Path
import pandas as pd

# Open our Excel dataset
project = Path(__file__).resolve().parent.parent
file = project / "data" / "processed" / "master_macro_dataset_2007_2025_UPDATED.xlsx"

df = pd.read_excel(file)

# Calculate Y using our formula
df["Y_calculated"] = (
    df["OilTaxNF (thousand KZT)"]
    / df["StateIncome (thousand KZT)"]
    * 100
)

# Compare our calculations with Excel
df["Difference"] = df["Y_calculated"] - df["Y (%)"]

print("\nFORMULA CHECK")
print(
    df[["Year", "Y (%)", "Y_calculated", "Difference"]]
    .round(4)
    .to_string(index=False)
)

# Check if the differences are small
max_difference = df["Difference"].abs().max()

if max_difference < 0.01:
    print("\nSUCCESS: Formula is correct!")
else:
    print("\nWARNING: Formula differences found!")
    print("Maximum difference:", max_difference)
