import pandas as pd
import numpy as np
from glob import glob
import os


IN_DIR  = r"C:\Ovi\New folder"  
OUT_CSV = r"C:\Ovi\New folder\AllSat_Average_Indices_1985_2014.csv"

LONLAT_DECIMALS = 3   

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

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = ["lon", "lat"] + [c for c in numeric_cols if c not in ("lon", "lat")]
    df = df[keep_cols]

    dfs.append(df)

all_cols = set().union(*[set(d.columns) for d in dfs]) - {"lon", "lat"}
all_cols = ["lon", "lat"] + sorted(all_cols)

stacked = pd.concat(
    [d.reindex(columns=all_cols) for d in dfs],
    axis=0,
    ignore_index=True
)

index_cols = [c for c in all_cols if c not in ("lon", "lat")]
avg_df = (
    stacked
    .groupby(["lon", "lat"], as_index=False)[index_cols]
    .mean()  
)

avg_df = avg_df.sort_values(["lat", "lon"]).reset_index(drop=True)
avg_df.to_csv(OUT_CSV, index=False)

print(f"✅ Averaged indices saved to:\n{OUT_CSV}")
print("Preview:")
print(avg_df.head())
