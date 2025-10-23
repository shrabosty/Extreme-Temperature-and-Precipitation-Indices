import pandas as pd
import numpy as np


TMAX_PATH = r"C:\Ovi\Bangladesh\ACCESS-CM2\historical\TMaxData"
TMIN_PATH = r"C:\Ovi\Bangladesh\ACCESS-CM2\historical\TMinData"
PRCP_PATH = r"C:\Ovi\Bangladesh\ACCESS-CM2\historical\PrecipData"

OUT_CSV   = r"C:\Ovi\Bangladesh\ACCESS-CM2\historical\All_Indices_1985_2014.csv"

BASE_START, BASE_END = 1985, 2014
WETDAY_THRESHOLD = 1.0  # mm (ETCCDI)
SU25_THRESHOLD   = 25.0 # °C
TR20_THRESHOLD   = 20.0 # °C
RUN_LEN          = 5    # for WSDI5/CSDI5
ROLL_WIN_RX5     = 5    # days for Rx5day


def load_daily_grid(path):
    raw = pd.read_csv(path, sep=r'\s+', header=None, engine='python', skip_blank_lines=True)
    lons = pd.to_numeric(raw.iloc[0, 3:], errors='coerce').to_numpy()
    lats = pd.to_numeric(raw.iloc[1, 3:], errors='coerce').to_numpy()
    df = raw.iloc[2:, :].reset_index(drop=True)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # year
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # month
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # day
    df = df.sort_values([df.columns[0], df.columns[1], df.columns[2]]).reset_index(drop=True)
    point_cols = [f"p{i}" for i in range(len(lons))]
    df.columns = ["year", "month", "day"] + point_cols
    vals = df.loc[:, point_cols].apply(pd.to_numeric, errors='coerce')
    return df, vals, lons, lats, point_cols

def clean_temp(df_vals):
    vals = df_vals.replace([-9999, -999, -99, 99, 99.0, 99.9, 32766], np.nan)
    vals = vals.mask((vals < -50) | (vals > 60))
    return vals

def clean_prcp(df_vals):
    vals = df_vals.replace([-9999, -999, -99, 99.9, 32766], np.nan)
    vals = vals.mask(vals < 0)
    return vals

def max_run_length(bool_array: np.ndarray) -> int:
    if bool_array.size == 0:
        return 0
    a = np.asarray(bool_array, dtype=bool)
    padded = np.r_[False, a, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    run_lengths = ends - starts
    return int(run_lengths.max()) if run_lengths.size else 0

def total_days_in_runs_ge_k(bool_array: np.ndarray, k: int) -> int:
    if bool_array.size == 0:
        return 0
    a = np.asarray(bool_array, dtype=bool)
    padded = np.r_[False, a, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    lengths = ends - starts
    return int(lengths[lengths >= k].sum()) if lengths.size else 0

# Tmax
dfx, tx_raw, lons, lats, point_cols = load_daily_grid(TMAX_PATH)
tx = clean_temp(tx_raw)

# Tmin
dfn, tn_raw, _, _, _ = load_daily_grid(TMIN_PATH)
tn = clean_temp(tn_raw)

# Precip
dfp, pr_raw, _, _, _ = load_daily_grid(PRCP_PATH)
pr = clean_prcp(pr_raw)


mask_tx = (dfx["year"] >= BASE_START) & (dfx["year"] <= BASE_END)
mask_tn = (dfn["year"] >= BASE_START) & (dfn["year"] <= BASE_END)
mask_pr = (dfp["year"] >= BASE_START) & (dfp["year"] <= BASE_END)

tx_b = tx.loc[mask_tx].copy()
tn_b = tn.loc[mask_tn].copy()
pr_b = pr.loc[mask_pr].copy()

years_tx = dfx.loc[mask_tx, "year"].to_numpy()
years_tn = dfn.loc[mask_tn, "year"].to_numpy()
years_pr = dfp.loc[mask_pr, "year"].to_numpy()

tx_p90 = tx_b.quantile(0.90, axis=0)
tx_p10 = tx_b.quantile(0.10, axis=0)
tn_p90 = tn_b.quantile(0.90, axis=0)
tn_p10 = tn_b.quantile(0.10, axis=0)

wet_base = pr_b.where(pr_b >= WETDAY_THRESHOLD)
pr_p95 = wet_base.quantile(0.95, axis=0)
pr_p99 = wet_base.quantile(0.99, axis=0)

TXm = tx_b.mean(axis=0, skipna=True)
TNm = tn_b.mean(axis=0, skipna=True)
TMm = ((tx_b + tn_b) / 2.0).mean(axis=0, skipna=True)

tx_tn_diff = (tx.loc[mask_tx].to_numpy() - tn.loc[mask_tn].to_numpy())
dtr_yearly = []
start = 0
for y in np.unique(years_tx):
    idx = np.where(years_tx == y)[0]
    if idx.size == 0: 
        continue
    dtr_y = np.nanmean(tx_tn_diff[idx, :], axis=0)
    dtr_yearly.append(dtr_y)
DTR = pd.DataFrame(dtr_yearly).mean(axis=0, skipna=True)
DTR.index = point_cols

def annual_agg(vals_df, years_vec, func):
    out = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        agg = func(vals_df.loc[idx], axis=0)
        agg.name = y
        out.append(agg)
    out = pd.DataFrame(out)
    return out.mean(axis=0, skipna=True)

TXx = annual_agg(tx_b, years_tx, pd.DataFrame.max)
TNx = annual_agg(tn_b, years_tn, pd.DataFrame.max)
TXn = annual_agg(tx_b, years_tx, pd.DataFrame.min)
TNn = annual_agg(tn_b, years_tn, pd.DataFrame.min)

TN10p = (tn_b.lt(tn_p10, axis=1).sum(axis=0) / tn_b.count(axis=0)) * 100.0
TX10p = (tx_b.lt(tx_p10, axis=1).sum(axis=0) / tx_b.count(axis=0)) * 100.0
TN90p = (tn_b.gt(tn_p90, axis=1).sum(axis=0) / tn_b.count(axis=0)) * 100.0
TX90p = (tx_b.gt(tx_p90, axis=1).sum(axis=0) / tx_b.count(axis=0)) * 100.0

def mean_wsdi_like_per_point(temps_b, years_vec, threshold_series, cmp="gt"):
    totals = {c: 0 for c in temps_b.columns}
    n_years = {c: 0 for c in temps_b.columns}
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        year_vals = temps_b.loc[idx, :]
        if cmp == "gt":
            is_ext = year_vals.gt(threshold_series, axis=1)
        else:
            is_ext = year_vals.lt(threshold_series, axis=1)
        for c in temps_b.columns:
            arr = is_ext[c].to_numpy()
            if np.isfinite(year_vals[c]).sum() == 0:
                continue
            days = total_days_in_runs_ge_k(arr, RUN_LEN)
            totals[c] += days
            n_years[c] += 1
    mean_per_year = pd.Series({c: (totals[c] / n_years[c]) if n_years[c] > 0 else np.nan for c in temps_b.columns})
    return mean_per_year

WSDI5 = mean_wsdi_like_per_point(tx_b, years_tx, tx_p90, cmp="gt")
CSDI5 = mean_wsdi_like_per_point(tn_b, years_tn, tn_p10, cmp="lt")

def mean_annual_count(vals_b, years_vec, threshold, cmp="gt"):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        v = vals_b.loc[idx, :]
        if cmp == "gt":
            cnt = (v.gt(threshold)).sum(axis=0)
        else:
            cnt = (v.ge(threshold)).sum(axis=0)  # not used here
        cnt.name = y
        yearly.append(cnt)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

SU25 = mean_annual_count(tx_b, years_tx, SU25_THRESHOLD, "gt")
TR20 = mean_annual_count(tn_b, years_tn, TR20_THRESHOLD, "gt")

Rx1day = annual_agg(pr_b, years_pr, pd.DataFrame.max)

def rx5day_mean(pr_b, years_vec, win=5):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        roll = g.rolling(window=win, min_periods=win).sum()
        max5 = roll.max(axis=0)
        max5.name = y
        yearly.append(max5)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

Rx5day = rx5day_mean(pr_b, years_pr, ROLL_WIN_RX5)

def rXXp_mean(pr_b, years_vec, pthr_series, use_strict=True):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        mask = g.gt(pthr_series, axis=1) if use_strict else g.ge(pthr_series, axis=1)
        s = g.where(mask).sum(axis=0, min_count=1)
        s.name = y
        yearly.append(s)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

R95p = rXXp_mean(pr_b, years_pr, pr_p95, use_strict=True)
R99p = rXXp_mean(pr_b, years_pr, pr_p99, use_strict=True)

def rNmm_mean(pr_b, years_vec, threshold_mm):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        c = (g >= threshold_mm).sum(axis=0)
        c.name = y
        yearly.append(c)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

R10mm = rNmm_mean(pr_b, years_pr, 10.0)
R20mm = rNmm_mean(pr_b, years_pr, 20.0)
R30mm = rNmm_mean(pr_b, years_pr, 30.0)

def mean_annual_max_run(pr_b, years_vec, threshold, mode="dry"):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        if mode == "dry":
            bool_df = g.lt(threshold) & g.notna()
        else:  # wet
            bool_df = g.ge(threshold) & g.notna()
        maxlens = bool_df.apply(lambda col: max_run_length(col.to_numpy()), axis=0)
        maxlens.name = y
        yearly.append(maxlens)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

CDD = mean_annual_max_run(pr_b, years_pr, WETDAY_THRESHOLD, mode="dry")
CWD = mean_annual_max_run(pr_b, years_pr, WETDAY_THRESHOLD, mode="wet")

def prcptot_mean(pr_b, years_vec, wet_thr=1.0):
    yearly = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        wet_sum = g.where(g >= wet_thr).sum(axis=0, min_count=1)
        wet_sum.name = y
        yearly.append(wet_sum)
    return pd.DataFrame(yearly).mean(axis=0, skipna=True)

PRCPTOT = prcptot_mean(pr_b, years_pr, WETDAY_THRESHOLD)

def rXXpTOT_mean_percent(rXXp_series_mean_annual_fn, pr_b, years_vec, pthr_series, wet_thr=1.0, strict=True):
    
    pct_years = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        prcptot_y = g.where(g >= wet_thr).sum(axis=0, min_count=1)
        mask = g.gt(pthr_series, axis=1) if strict else g.ge(pthr_series, axis=1)
        rxxp_y = g.where(mask).sum(axis=0, min_count=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (rxxp_y / prcptot_y) * 100.0
        pct.name = y
        pct_years.append(pct)
    return pd.DataFrame(pct_years).mean(axis=0, skipna=True)

R95PTOT = rXXpTOT_mean_percent(rXXp_mean, pr_b, years_pr, pr_p95, WETDAY_THRESHOLD, strict=True)
R99PTOT = rXXpTOT_mean_percent(rXXp_mean, pr_b, years_pr, pr_p99, WETDAY_THRESHOLD, strict=True)

def sdii_mean(pr_b, years_vec, wet_thr=1.0):
    sdii_years = []
    for y in np.unique(years_vec):
        idx = (years_vec == y)
        g = pr_b.loc[idx, :]
        wet_mask = g >= wet_thr
        prcptot_y = g.where(wet_mask).sum(axis=0, min_count=1)
        wetcount_y = wet_mask.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            sdii_y = prcptot_y / wetcount_y
        sdii_y.name = y
        sdii_years.append(sdii_y)
    return pd.DataFrame(sdii_years).mean(axis=0, skipna=True)

SDII = sdii_mean(pr_b, years_pr, WETDAY_THRESHOLD)

out = pd.DataFrame({
    "lon": lons,
    "lat": lats,
    "TXm": TXm.values,
    "TNm": TNm.values,
    "TMm": TMm.values,
    "TXx": TXx.values,
    "TNx": TNx.values,
    "TXn": TXn.values,
    "TNn": TNn.values,
    "TN10p": TN10p.values,
    "TX10p": TX10p.values,
    "TN90p": TN90p.values,
    "TX90p": TX90p.values,
    "WSDI5": WSDI5.values,
    "CSDI5": CSDI5.values,
    "DTR": DTR.values,
    "SU25": SU25.values,
    "TR20": TR20.values,
    "Rx1day": Rx1day.values,
    "Rx5day": Rx5day.values,
    "R95p": R95p.values,
    "R99p": R99p.values,
    "R10mm": R10mm.values,
    "R20mm": R20mm.values,
    "R30mm": R30mm.values,
    "CDD": CDD.values,
    "CWD": CWD.values,
    "PRCPTOT": PRCPTOT.values,
    "R95PTOT": R95PTOT.values,
    "R99PTOT": R99PTOT.values,
    "SDII": SDII.values,
})

out = out.sort_values(["lat", "lon"]).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
print("Saved all indices to:", OUT_CSV)
print(out.head())
