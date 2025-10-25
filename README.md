[README.md](https://github.com/user-attachments/files/23143544/README.md)
# Extreme Temperature and Precipitation Indices

Reproducible code and helper scripts to compute **ETCCDI** and **ET-SCI** climate indices from daily **Tmax / Tmin / Precipitation** gridded data (historical and scenarios like **SSP245**).  
It supports your common text format where:

- **Row 0** = longitudes for each grid point  
- **Row 1** = latitudes for each grid point  
- **Rows 2+** = `year  month  day` followed by values for each grid point

> The scripts calculate indices per–grid point, aggregate **annually**, and (by default) export **mean values across a target period** (e.g., 1985–2014, 2015–2039, 2040–2069, 2070–2100).

---

## Table of contents
- [What’s inside](#whats-inside)
- [Supported indices](#supported-indices)
- [Data format](#data-format)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [A) Historical indices (1985–2014)](#a-historical-indices-1985–2014)
  - [B) Scenario indices (SSP245, three future periods)](#b-scenario-indices-ssp245-three-future-periods)
  - [C) Averaging across multiple satellites](#c-averaging-across-multiple-satellites)
- [Notes & conventions](#notes--conventions)
- [Citing definitions](#citing-definitions)
- [Contributing](#contributing)
- [License](#license)

---

## What’s inside

- `ACCESS-CM2/` — model-specific notebooks / scripts (e.g., historical + SSP245 runs)
- `Stat Test + Corr Matrix.ipynb` — example analysis notebook
- Python scripts (in notebooks or `.py` form) that:
  - parse the input grid format,
  - compute ETCCDI / ET-SCI indices per point,
  - export tidy CSVs,
  - optionally aggregate across multiple satellite/model CSVs.

---

## Supported indices

**Temperature (°C)**  
- `TXm` Mean daily maximum temperature  
- `TNm` Mean daily minimum temperature  
- `TMm` Mean daily mean temperature  
- `TXx` Annual maximum of daily Tmax *(mean of annual maxima over the period)*  
- `TNx` Annual maximum of daily Tmin *(mean of annual maxima)*  
- `TXn` Annual minimum of daily Tmax *(mean of annual minima)*  
- `TNn` Annual minimum of daily Tmin *(mean of annual minima)*  
- `DTR` Mean annual diurnal temperature range (TX − TN)  
- `TN10p` % of days with TN < 10th percentile (base: period)  
- `TX10p` % of days with TX < 10th percentile (base: period)  
- `TN90p` % of days with TN > 90th percentile (base: period)  
- `TX90p` % of days with TX > 90th percentile (base: period)  
- `WSDI5` Warm spell duration index (≥5 consecutive days with TX > 90th percentile)  
- `CSDI5` Cold spell duration index (≥5 consecutive days with TN < 10th percentile)

**Precipitation (mm / days)**  
- `Rx1day` Annual maximum 1-day precipitation *(mean of annual maxima)*  
- `Rx5day` Annual maximum consecutive 5-day precipitation *(mean of annual maxima)*  
- `R95p` Annual total precipitation from days > 95th percentile (wet-day distribution)  
- `R99p` Annual total precipitation from days > 99th percentile (wet-day distribution)  
- `R10mm` Annual count of days with precip ≥ 10 mm  
- `R20mm` Annual count of days with precip ≥ 20 mm  
- `R30mm` Annual count of days with precip ≥ 30 mm  
- `CDD` Mean annual **max** run length of dry days (precip < 1 mm)  
- `CWD` Mean annual **max** run length of wet days (precip ≥ 1 mm)  
- `PRCPTOT` Annual total precipitation on wet days (precip ≥ 1 mm)  
- `R95PTOT` 100 × R95p / PRCPTOT (annual %, then averaged)  
- `R99PTOT` 100 × R99p / PRCPTOT (annual %, then averaged)  
- `SDII` Simple Daily Intensity Index = PRCPTOT / (# wet days)

---

## Data format

Each input file is plain text, space-separated:

```
<ignored> <ignored> <ignored>  lon1   lon2   lon3 ...
<ignored> <ignored> <ignored>  lat1   lat2   lat3 ...
year  month  day     v1_1  v1_2  v1_3 ...
year  month  day     v2_1  v2_2  v2_3 ...
...
```

- First row: **longitudes** for all grid points (from column 4 onward)  
- Second row: **latitudes** for all grid points (from column 4 onward)  
- Subsequent rows: `year month day` + one column per grid point

---

## Quick start

### Requirements
- Python 3.9+  
- `pandas`, `numpy`

Install:
```bash
pip install -U pandas numpy
```

### Clone
```bash
git clone https://github.com/shrabosty/Extreme-Temperature-and-Precipitation-Indices.git
cd Extreme-Temperature-and-Precipitation-Indices
```

---

## Usage

### A) Historical indices (1985–2014)

Run the all-in-one historical script to generate `All_Indices_1985_2014.csv` with one row per (lon, lat) and columns for each index.

### B) Scenario indices (SSP245, three future periods)

A single script computes/saves indices for all **three** SSP245 periods:

- `2015–2039` → `All_Indices_2015_2039.csv`  
- `2040–2069` → `All_Indices_2040_2069.csv`  
- `2070–2100` → `All_Indices_2070_2100.csv`

By default, percentile thresholds (TX/TN 10th & 90th, precip 95th & 99th) are computed **within each period**.  

### C) Averaging across multiple satellites

If you have multiple CSVs (e.g., **13 satellites** for 2015–2039) and want a **per-point average** of each index across satellites:
```python
IN_DIR  = r"C:\Ovi\SSP 245 2015-2039"
OUT_CSV = r"C:\Ovi\SSP 245 2015-2039\AllSat_Average_Indices_2015_2039.csv"
```

---

## Notes & conventions

- **Wet day**: precip ≥ 1 mm (ETCCDI)  
- **R95p/R99p thresholds**: computed from wet-day distribution  
- **WSDI/CSDI**: ≥5 consecutive days (user-defined window)  
- **DTR**: mean of yearly mean TX−TN  
- **Sentinels**: −9999, 32766, etc. are treated as NaN  
- **Outputs**: CSV with `lon, lat, <indices...>`

---

## Citing definitions

- ETCCDI official index list  
- Climdex / Climpact documentation (WSDI, CSDI, percentile logic)
- ET-SCI / ClimPACT2 user guide

---

## Contributing

Issues and pull requests are welcome. Please:
1. Open an issue describing your change or feature.  
2. Keep functions modular and documented.  
3. Note any deviations from ETCCDI standards.

