#!/usr/bin/env python
# coding: utf-8

# # Statistical Test Value for all Temperature index

# In[48]:


import pandas as pd
import numpy as np
from scipy import stats
hist_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\Hist.csv"
future_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\SSP 245 NF.csv"
df_hist = pd.read_csv(hist_path)
df_fut  = pd.read_csv(future_path)
variables = [
    "TXx", "TXn", "TNx", "TNn",
    "TX90p", "TN90p", "TX10p", "TN10p",
    "TXm", "TNm", "TMm", "DTR", 
    "SU25", "TR20", "CSDI5", "WSDI5"
]
lat_col = df_hist.columns[0]
lon_col = df_hist.columns[1]
for df in (df_hist, df_fut):
    df["lat_round"] = df[lat_col].round(4)
    df["lon_round"] = df[lon_col].round(4)
results = []

for var in variables:
    print(f"\nVariable: {var}")

    if var not in df_hist.columns or var not in df_fut.columns:
        print(f"'{var}' not found in one of the files — skipping.")
        continue

    hist = df_hist[["lat_round", "lon_round", var]].copy()
    hist.columns = ["lat", "lon", "value_hist"]

    fut = df_fut[["lat_round", "lon_round", var]].copy()
    fut.columns = ["lat", "lon", "value_fut"]

    df = pd.merge(hist, fut, on=["lat", "lon"], how="inner")
    df.dropna(subset=["value_hist", "value_fut"], inplace=True)

    n = len(df)
    if n < 5:
        print(f"Not enough valid data for {var}. Skipping.")
        continue
    d = df["value_fut"].to_numpy() - df["value_hist"].to_numpy()
    mean_diff   = np.mean(d)
    median_diff = np.median(d)
    sd_diff     = np.std(d, ddof=1)
    alpha = 0.05
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - tcrit * sd_diff / np.sqrt(n)
    ci_upper = mean_diff + tcrit * sd_diff / np.sqrt(n)
    t_stat, p_ttest = stats.ttest_rel(df["value_fut"], df["value_hist"])
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p_wilcoxon = np.nan
    cohens_dz = mean_diff / sd_diff if sd_diff != 0 else np.nan
    results.append({
        "Variable": var,
        "N": n,
        "Mean Change": mean_diff,
        "Median Change": median_diff,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "t-test p": p_ttest,
        "Wilcoxon p": p_wilcoxon,
        "Cohen's dz": cohens_dz
    })
    print(f"Sample size: {n}")
    print(f"Mean change: {mean_diff:.4f} (95% CI: {ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Median change: {median_diff:.4f}")
    print(f"Paired t-test p = {p_ttest:.6g}")
    print(f"Wilcoxon p      = {p_wilcoxon:.6g}")
    print(f"Cohen’s dz       = {cohens_dz:.3f}")


# In[47]:


results_df = pd.DataFrame(results)
print("\nSUMMARY TABLE")
print(results_df.to_string(index=False))


# # Correlation and Heatmap for all Temperature Index

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
hist_path  = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\Hist.csv"
future_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\SSP 245 NF.csv"  # or MF/FF/SSP585
df_hist = pd.read_csv(hist_path)
df_fut  = pd.read_csv(future_path)
variables = [
    "TXx", "TXn", "TNx", "TNn",
    "TX90p", "TN90p", "TX10p", "TN10p",
    "TXm", "TNm", "TMm", "DTR",
    "SU25", "TR20", "CSDI5", "WSDI5"
]
lat_col = df_hist.columns[0]
lon_col = df_hist.columns[1]
for df in (df_hist, df_fut):
    df["lat_round"] = df[lat_col].round(4)
    df["lon_round"] = df[lon_col].round(4)
merged = pd.merge(
    df_hist[["lat_round", "lon_round"] + variables],
    df_fut[["lat_round", "lon_round"] + variables],
    on=["lat_round", "lon_round"],
    suffixes=("_hist", "_fut"),
    how="inner"
)
for var in variables:
    merged[f"{var}_diff"] = merged[f"{var}_fut"] - merged[f"{var}_hist"]

diff_cols = [f"{v}_diff" for v in variables]
corr_matrix = merged[diff_cols].corr(method="pearson")
print(corr_matrix.round(3))


# In[36]:


plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    cbar_kws={'label': 'Pearson r'},
    square=True
)
plt.title("Spatial Correlation Between Temperature Index Changes (Future - Historical)\nSSP245 Near Future", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


# # Statistical Test Value for all Precipitation index

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
hist_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\Hist.csv"
future_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\SSP 245 NF.csv"
out_csv = r"C:\New folder\Puran Laptop\Paper With Bou\Difference\SSP 245\Stats_Summary_Precip.csv"
df_hist = pd.read_csv(hist_path)
df_fut  = pd.read_csv(future_path)
variables = [
    "CDD", "CWD", "PRCPTOT", "SDII",
    "R10mm", "R20mm", "R30mm",
    "R95p", "R95PTOT", "R99p", "R99PTOT",
    "Rx1day", "Rx5day"
]
lat_col = df_hist.columns[0]
lon_col = df_hist.columns[1]
for df in (df_hist, df_fut):
    df["lat_round"] = df[lat_col].round(4)
    df["lon_round"] = df[lon_col].round(4)
results = []
for var in variables:
    print(f"\nVariable: {var}")
    if var not in df_hist.columns or var not in df_fut.columns:
        print(f"'{var}' not found in one of the datasets — skipping.")
        continue
    hist = df_hist[["lat_round", "lon_round", var]].copy()
    hist.columns = ["lat", "lon", "value_hist"]

    fut = df_fut[["lat_round", "lon_round", var]].copy()
    fut.columns = ["lat", "lon", "value_fut"]

    df = pd.merge(hist, fut, on=["lat", "lon"], how="inner")
    df.dropna(subset=["value_hist", "value_fut"], inplace=True)

    n = len(df)
    if n < 5:
        print(f"Not enough matched points for {var} (n={n}). Skipping.")
        continue
    d = df["value_fut"].to_numpy() - df["value_hist"].to_numpy()
    mean_diff = np.mean(d)
    median_diff = np.median(d)
    sd_diff = np.std(d, ddof=1)
    alpha = 0.05
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - tcrit * sd_diff / np.sqrt(n)
    ci_upper = mean_diff + tcrit * sd_diff / np.sqrt(n)
    t_stat, p_ttest = stats.ttest_rel(df["value_fut"], df["value_hist"])
    try:
        _, p_wilcoxon = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p_wilcoxon = np.nan
    cohens_dz = mean_diff / sd_diff if sd_diff != 0 else np.nan
    results.append({
        "Variable": var,
        "N": n,
        "Mean Change": mean_diff,
        "Median Change": median_diff,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "t-test p": p_ttest,
        "Wilcoxon p": p_wilcoxon,
        "Cohen's dz": cohens_dz
    })
    print(f"Sample size: {n}")
    print(f"Mean change: {mean_diff:.4f} (95% CI: {ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Median change: {median_diff:.4f}")
    print(f"Paired t-test p = {p_ttest:.6g}")
    print(f"Wilcoxon p      = {p_wilcoxon:.6g}")
    print(f"Cohen’s dz       = {cohens_dz:.3f}")


# In[2]:


results_df = pd.DataFrame(results)
print("\nSUMMARY TABLE")
print(results_df.to_string(index=False))


# # Correlation and Heatmap for all Precipitation Index

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
hist_path  = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\Hist.csv"
future_path = r"C:\New folder\Puran Laptop\Paper With Bou\Raw Excel\SSP 245 NF.csv"
df_hist = pd.read_csv(hist_path)
df_fut  = pd.read_csv(future_path)
variables = [
    "CDD", "CWD", "PRCPTOT", "SDII",
    "R10mm", "R20mm", "R30mm",
    "R95p", "R95PTOT", "R99p", "R99PTOT",
    "Rx1day", "Rx5day"
]
lat_col = df_hist.columns[0]
lon_col = df_hist.columns[1]
for df in (df_hist, df_fut):
    df["lat_round"] = df[lat_col].round(4)
    df["lon_round"] = df[lon_col].round(4)
merged = pd.merge(
    df_hist[["lat_round", "lon_round"] + variables],
    df_fut[["lat_round", "lon_round"] + variables],
    on=["lat_round", "lon_round"],
    suffixes=("_hist", "_fut"),
    how="inner"
)
for var in variables:
    merged[f"{var}_diff"] = merged[f"{var}_fut"] - merged[f"{var}_hist"]
diff_cols = [f"{v}_diff" for v in variables]
corr_matrix = merged[diff_cols].corr(method="pearson")
print("\nSpatial Correlation Between Precipitation Index Changes (Future - Historical) [Pearson]")
print(corr_matrix.round(3))


# In[10]:


plt.figure(figsize=(10, 8))
sns.set(font_scale=0.9)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Pearson Correlation Coefficient"}
)

plt.title("Spatial Correlation of Precipitation Index Changes (SSP2-4.5 NF vs Historical)", fontsize=12, pad=12)
plt.tight_layout()
plt.show()

