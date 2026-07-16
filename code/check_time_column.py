"""
check_time_column.py

Diagnostic: scans all files in data/known_lightcurves/ and reports
how many are missing a usable 'time' column (the bug from the
original download script before the reset_index() fix).

Author: Ray's Exoplanet AI Project
"""

import os
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "known_lightcurves")
CATALOG_FOLDER = os.path.join(SCRIPT_DIR, "..", "data", "catalogs")
REPORT_PATH = os.path.join(CATALOG_FOLDER, "time_column_check.csv")

files = [f for f in os.listdir(RAW_FOLDER) if f.endswith(".csv")]
print(f"Checking {len(files)} files in {RAW_FOLDER}...\n")

results = []
missing_time = 0
has_time = 0
unreadable = 0

for filename in tqdm(files, desc="Checking"):
    path = os.path.join(RAW_FOLDER, filename)
    try:
        header = pd.read_csv(path, nrows=0).columns
        if "time" in header:
            has_time += 1
            results.append([filename, "OK", len(header)])
        else:
            missing_time += 1
            results.append([filename, "MISSING time", len(header)])
    except Exception as e:
        unreadable += 1
        results.append([filename, f"Unreadable: {e}", None])

report_df = pd.DataFrame(results, columns=["File", "Status", "Column Count"])
report_df.to_csv(REPORT_PATH, index=False)

print("\n===================================")
print("Check Finished")
print("===================================")
print(f"Total files checked: {len(files)}")
print(f"Files with valid 'time' column: {has_time}")
print(f"Files MISSING 'time' column: {missing_time}")
print(f"Unreadable files: {unreadable}")
print(f"\nFull report saved to: {REPORT_PATH}")

if missing_time > 0:
    pct = round(100 * missing_time / len(files), 1)
    print(f"\n{pct}% of your files need to be re-downloaded with the fixed script.")