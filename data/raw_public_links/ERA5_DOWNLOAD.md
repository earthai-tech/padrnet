# ERA5 Reanalysis Data — Download Instructions

ERA5 data are provided by the Copernicus Climate Change Service (C3S) and
cannot be bundled in this archive. They must be downloaded individually
through the CDS (Climate Data Store).

---

## Licence

ERA5 data are distributed under the **Copernicus licence**:
https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf

You are free to use and redistribute ERA5 data for research, education, and
non-commercial purposes provided you acknowledge the source:

> Hersbach, H. et al. (2020). The ERA5 global reanalysis.
> *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
> https://doi.org/10.1002/qj.3803

---

## Variables used in this study

| Variable | CDS name | Level type |
|---|---|---|
| Total precipitation | `total_precipitation` | single |
| 2-m temperature | `2m_temperature` | single |
| Volumetric soil water layer 1 | `volumetric_soil_water_layer_1` | single |
| Surface runoff | `surface_runoff` | single |
| Evaporation | `evaporation` | single |
| 10-m u-component of wind | `10m_u_component_of_wind` | single |
| 10-m v-component of wind | `10m_v_component_of_wind` | single |

**Temporal coverage:** 2000-01-01 to 2024-12-31, hourly  
**Spatial resolution:** 0.25 degrees  
**Study regions (bounding boxes):**

| Region | lat_min | lon_min | lat_max | lon_max |
|---|---|---|---|---|
| West Africa (Niger/Benue) | 4.0 | -12.0 | 15.0 | 15.0 |
| East Africa (Nile headwaters) | -4.0 | 28.0 | 16.0 | 40.0 |
| Southern Africa (Limpopo/Zambezi) | -27.0 | 20.0 | -8.0 | 37.0 |

---

## How to download

### Step 1 — Register

Create a free account at https://cds.climate.copernicus.eu/user/register

### Step 2 — Install the CDS API client

```bash
pip install cdsapi
```

### Step 3 — Save your credentials

After logging in, copy your UID and API key from
https://cds.climate.copernicus.eu/api-how-to and save them to
`~/.cdsapirc` (Linux/macOS) or `%USERPROFILE%\.cdsapirc` (Windows):

```
url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR_UID>:<YOUR_API_KEY>
```

### Step 4 — Run the download script

```bash
python code/scripts/download_era5.py --out-dir data/raw/reanalysis/era5
```

Downloaded files will be placed as:

```
data/raw/reanalysis/era5/
  west_africa_niger_benue/
    era5_hourly_2000.nc
    era5_hourly_2001.nc
    ...
  east_africa_nile_headwaters/
    ...
  southern_africa_limpopo_zambezi/
    ...
```

After download, run `code/scripts/03_build_era5_covariates.py` to extract
the event-level feature table (`data/processed/era5_covariates.csv`).
That pre-computed file is already included in the archive.

---

## Note on pre-computed features

If you only want to reproduce the model results (Tables 1–2, Figures 4–12),
you do **not** need to download ERA5. The extracted feature table
`data/processed/era5_covariates.csv` is already included in the archive
and is the direct input to `code/scripts/04_padrnet_training.py`.
