
from pathlib import Path
import pandas as pd

# Find our project folder
project_folder = Path(__file__).resolve().parent.parent

# Excel file containing our macroeconomic data
excel_file = (
    project_folder
    / "data"
    / "processed"
    / "master_macro_dataset_2007_2025_UPDATED.xlsx"
)

# Check that the file exists
if not excel_file.exists():
    print("ERROR: Excel file not found!")
    print("Expected location:", excel_file)
else:
    # Load Excel data
    data = pd.read_excel(excel_file)

    print("DATASET LOADED SUCCESSFULLY!")
    print("\nNumber of rows and columns:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nFirst five rows:")
    print(data.head())

    print("\nMissing values:")
    print(data.isna().sum())
