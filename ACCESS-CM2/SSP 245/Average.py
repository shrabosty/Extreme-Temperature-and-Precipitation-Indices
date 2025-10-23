import pandas as pd
import numpy as np
from glob import glob
import os

IN_DIR  = r"C:\Ovi\SSP 245 2015-2039"   
OUT_CSV = os.path.join(IN_DIR, "AllSat_Average_Indices_2015_2039.csv")

LONLAT_DECIMALS = 3   

# 1) Load all CSVs
files = sorted(glob(os.path.join(IN_DIR, "*.csv")))
if not files:
    raise FileNotFoundError(f"No CSV files found in: {IN_DIR}")

dfs = []
for f in files:
    df = pd.read_csv(f)

    if not {"lon", "lat"}.issubset(df.columns):
        raise ValueError(f"'lon'/'lat' columns missing in {os.path.basename(f)}")

    if LONLAT_DECIMALS is not None:
        df["lon"] = df["lon"].round(LONLAT_DECIMALS)
        df["lat"] = df["lat"].round(LONLAT_DECIMALS)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = ["lon", "lat"] + [c for c in num_cols if c not in ("lon", "lat")]
    dfs.append(df[keep_cols])

all_cols = set().union(*[set(d.columns) for d in dfs]) - {"lon", "lat"}
all_cols = ["lon", "lat"] + sorted(all_cols)

stacked = pd.concat([d.reindex(columns=all_cols) for d in dfs], ignore_index=True, axis=0)

index_cols = [c for c in all_cols if c not in ("lon", "lat")]
avg_df = (
    stacked
    .groupby(["lon", "lat"], as_index=False)[index_cols]
    .mean()   
)

avg_df = avg_df.sort_values(["lat", "lon"]).reset_index(drop=True)
avg_df.to_csv(OUT_CSV, index=False)

print(f"Averaged indices saved to:\n{OUT_CSV}")
print("Preview:")
print(avg_df.head())
