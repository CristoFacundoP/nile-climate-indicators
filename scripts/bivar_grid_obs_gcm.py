#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, argparse, warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
    _HAS_GPD = True
except Exception:
    _HAS_GPD = False

# ---------------- helpers (from your script) ----------------
def _rename_latlon(ds):
    ren = {}
    if "latitude"  in ds.coords: ren["latitude"]  = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    return ds.rename(ren) if ren else ds

def _preprocess_for_mf(ds):
    ds = _rename_latlon(ds)
    if "time" in ds.coords:
        t = ds["time"].values
        order = np.argsort(t)
        if not np.all(order == np.arange(t.size)):
            ds = ds.isel(time=order)
        t2 = ds["time"].values
        _, keep_idx = np.unique(t2, return_index=True)
        if keep_idx.size != ds.sizes["time"]:
            ds = ds.isel(time=np.sort(keep_idx))
    return ds

def _open_mf(pattern, engine="h5netcdf", chunks=None):
    if chunks is None: chunks = {"time": 365}
    try:
        ds = xr.open_mfdataset(
            pattern, engine=engine, combine="by_coords", decode_times=True,
            chunks=chunks, preprocess=_preprocess_for_mf, parallel=False
        )
    except Exception:
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files for pattern: {pattern}")
        parts = []
        for fp in files:
            d = xr.open_dataset(fp, engine=engine, decode_times=True)
            d = _preprocess_for_mf(d)
            parts.append(d)
        ds = xr.concat(parts, dim="time", join="override", combine_attrs="override")
    ds = _rename_latlon(ds)
    return ds.sortby("time") if "time" in ds.coords else ds

def _slice_dir(coord, vmin, vmax):
    c = coord.values
    return slice(vmin, vmax) if c[0] <= c[-1] else slice(vmax, vmin)

def _apply_bbox(ds, bbox):
    if not bbox or ("lat" not in ds.coords) or ("lon" not in ds.coords):
        return ds
    lonmin, lonmax, latmin, latmax = bbox
    return ds.sel(lon=_slice_dir(ds["lon"], lonmin, lonmax),
                  lat=_slice_dir(ds["lat"], latmin, latmax))

def _var_first_match(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    for v in ds.data_vars:
        if "lat" in ds[v].dims and "lon" in ds[v].dims:
            return v
    raise KeyError(f"None of {candidates} found in {list(ds.data_vars)}")

def annual_precip_mean(ds_pr, start, end):
    vname = _var_first_match(ds_pr, ["precip", "pr", "precipitation"])
    da = ds_pr[vname]
    units = str(da.attrs.get("units", "")).lower()
    if ("kg" in units) and ("s-1" in units or "s^" in units):
        da = da * 86400.0  # kg m-2 s-1 -> mm/day
    sub = da.sel(time=slice(start, end))
    return sub.resample(time="YE").sum().mean("time")

def annual_tmean_mean(ds_tmin, ds_tmax, start, end):
    vmin = _var_first_match(ds_tmin, ["Tmin","tmin","tasmin"])
    vmax = _var_first_match(ds_tmax, ["Tmax","tmax","tasmax"])
    Tn = ds_tmin[vmin].sel(time=slice(start, end))
    Tx = ds_tmax[vmax].sel(time=slice(start, end))
    return ((Tx + Tn) / 2.0).resample(time="YE").mean().mean("time")

def intersect_no_interp(daA, daB):
    latA, latB = daA["lat"].values, daB["lat"].values
    lonA, lonB = daA["lon"].values, daB["lon"].values
    common_lat = np.intersect1d(latA, latB)
    common_lon = np.intersect1d(lonA, lonB)
    if common_lat.size == 0 or common_lon.size == 0:
        raise ValueError("No overlapping coordinates between inputs.")
    return (daA.reindex(lat=common_lat, lon=common_lon),
            daB.reindex(lat=common_lat, lon=common_lon))

def digitize_safe(da, edges):
    idx = np.digitize(da.values, edges, right=False) - 1
    return np.clip(idx, 0, len(edges)-2)

# -------- anchored palette (ASCII-safe) --------
def _hex(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)

def bivar_palette_anchored(
    nP: int, nT: int,
    lowlow="#EAEAEA",     # lowest P & lowest T  -> gray
    highT_lowP="#E874AD", # highest T & lowest P -> pink
    lowT_highP="#9FD3E6", # lowest T & highest P -> light blue
    high_high="#6E2B73"   # highest T & highest P -> dark magenta
):
    """Build nP x nT color grid via bilinear blend of the four corner anchors."""
    LL = _hex(lowlow)       # (T low, P low)
    LR = _hex(highT_lowP)   # (T high, P low)
    UL = _hex(lowT_highP)   # (T low, P high)
    UR = _hex(high_high)    # (T high, P high)

    grid = np.zeros((nP, nT, 3), dtype=float)
    for i in range(nP):          # precip: low -> high
        p = i / (nP - 1) if nP > 1 else 0.0
        for j in range(nT):      # temp:   low -> high
            t = j / (nT - 1) if nT > 1 else 0.0
            grid[i, j, :] = np.clip(
                (1 - t) * (1 - p) * LL + t * (1 - p) * LR +
                (1 - t) * p * UL     + t * p * UR,
                0, 1
            )
    return grid

def make_color_grid(nP, nT, which="colombia"):
    # keep the CLI flag, but use the anchored palette we defined
    if which == "colombia":
        return bivar_palette_anchored(nP, nT)
    raise ValueError("Unsupported palette")


def plot_panel(ax, rgb, lon, lat, title=None, overlay=None, lw=0.8, la=1.0,
               show_xlabel=False, show_ylabel=False):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(rgb, origin="lower", extent=extent, aspect="auto", interpolation="nearest")

    if overlay is not None:
        try:
            import geopandas as gpd  # ensure available in this scope
            gdf = gpd.read_file(overlay)
            if gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "epsg: 4326"):
                gdf = gdf.to_crs("EPSG:4326")
            gdf.boundary.plot(ax=ax, color="k", linewidth=lw, alpha=la)
        except Exception as e:
            print(f"[WARN] overlay failed: {e}")

    # tidy ticks/labels
    ax.tick_params(length=2, pad=1, labelsize=8)

    if show_xlabel:
        ax.set_xlabel("lon", fontsize=10)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("lat", fontsize=10)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    if title:
        ax.set_title(title, fontsize=10, pad=2)


# -------- square legend with inequality ticks (ASCII-safe) --------
def draw_legend(ax, color_grid, Tedges, Pedges):
    nP, nT = color_grid.shape[:2]

    # Draw exact grid with square cells
    ax.imshow(
        color_grid, origin="lower",
        extent=[-0.5, nT - 0.5, -0.5, nP - 0.5],
        aspect="equal", interpolation="nearest"
    )
    ax.set_xlim(-0.5, nT - 0.5)
    ax.set_ylim(-0.5, nP - 0.5)

    # Put ticks on bin BOUNDARIES (n+1 ticks) and label with inequalities
    ax.set_xticks(np.linspace(-0.5, nT - 0.5, nT + 1))
    ax.set_yticks(np.linspace(-0.5, nP - 0.5, nP + 1))

    def _ineq_labels(edges):
        # e.g., =16, 20, 24, 28, 32, =36  (Unicode escapes)
        labs = [f"\u2264{int(edges[0])}"]               # =
        labs += [f"{int(v)}" for v in edges[1:-1]]
        labs += [f"\u2265{int(edges[-1])}"]             # =
        return labs

    ax.set_xticklabels(_ineq_labels(Tedges), rotation=45, fontsize=8)
    ax.set_yticklabels(_ineq_labels(Pedges), fontsize=8)

    ax.set_xlabel("Temperature (\u00B0C)", fontsize=9)   # °C
    ax.set_ylabel("Precipitation (mm/yr)", fontsize=9)

    # Optional: faint grid lines on boundaries for readability
    for x in range(nT):
        ax.axvline(x - 0.5, color="0.75", lw=0.4, alpha=0.5)
    for y in range(nP):
        ax.axhline(y - 0.5, color="0.75", lw=0.4, alpha=0.5)

# -------------- grid compute --------------
def compute_rgb(pr_glob, tmin_glob, tmax_glob, engine, bbox, start, end, Tedges, Pedges, color_grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_pr   = _open_mf(pr_glob,   engine=engine)
        ds_tmin = _open_mf(tmin_glob, engine=engine)
        ds_tmax = _open_mf(tmax_glob, engine=engine)
    if bbox:
        ds_pr   = _apply_bbox(ds_pr,   bbox)
        ds_tmin = _apply_bbox(ds_tmin, bbox)
        ds_tmax = _apply_bbox(ds_tmax, bbox)
    P = annual_precip_mean(ds_pr, start, end)
    T = annual_tmean_mean(ds_tmin, ds_tmax, start, end)
    # ensure common grid between P & T
    P2, T2 = intersect_no_interp(P, T)
    lat = P2["lat"].values; lon = P2["lon"].values
    if lat[0] > lat[-1]:
        P2 = P2.isel(lat=slice(None, None, -1))
        T2 = T2.isel(lat=slice(None, None, -1))
        lat = lat[::-1]
    Ti = digitize_safe(T2, Tedges); Pi = digitize_safe(P2, Pedges)
    rgb = color_grid[Pi, Ti, :]
    return rgb, lon, lat

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # OBS inputs + window
    ap.add_argument("--obs_pr_glob", required=True)
    ap.add_argument("--obs_tmin_glob", required=True)
    ap.add_argument("--obs_tmax_glob", required=True)
    ap.add_argument("--obs_start", required=True)
    ap.add_argument("--obs_end", required=True)

    # GCM templates: use {model} and {scenario}
    ap.add_argument("--gcm_pr_tpl", required=True, help="e.g., /gcm/{model}/{scenario}/pr_day_*.nc")
    ap.add_argument("--gcm_tmin_tpl", required=True)
    ap.add_argument("--gcm_tmax_tpl", required=True)

    # Lists
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["historical","ssp126","ssp245","ssp585"])

    # Per-scenario periods as JSON string
    ap.add_argument("--scenario_periods_json", required=True)

    # Common options
    ap.add_argument("--edges_json", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN","LON_MAX","LAT_MIN","LAT_MAX"))
    ap.add_argument("--engine", default="h5netcdf")
    ap.add_argument("--overlay_shp", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--palette", choices=["colombia"], default="colombia")
    ap.add_argument("--right_col_width", type=float, default=0.9, help="relative width for OBS+legend column")
    ap.add_argument(
        "--title",
        default="Northeast Africa \u2014 Bivariate Annual Temperature (\u00B0C) \u00D7 Precipitation (mm/yr)"
    )
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.edges_json, "r") as f:
        cfg = json.load(f)
    Tedges = np.asarray(cfg.get("Tedges") or cfg.get("Tedges_degC"), float)
    Pedges = np.asarray(cfg.get("Pedges") or cfg.get("Pedges_mm_yr"), float)
    nT = len(Tedges) - 1; nP = len(Pedges) - 1
    color_grid = make_color_grid(nP, nT, which=a.palette)

    scen_periods = json.loads(a.scenario_periods_json)  # dict: scen -> "YYYY-MM-DD,YYYY-MM-DD"

    # Precompute OBS rgb (baseline)
    obs_rgb, obs_lon, obs_lat = compute_rgb(
        a.obs_pr_glob, a.obs_tmin_glob, a.obs_tmax_glob, a.engine, a.bbox,
        a.obs_start, a.obs_end, Tedges, Pedges, color_grid
    )

    # Figure: grid of (rows=models, cols=scenarios) + right column
    n_rows = len(a.models); n_cols = len(a.scenarios) + 1  # +1 for right column
    # width ratios: scenarios = 1 each, right col = a.right_col_width
    width_ratios = [1]*len(a.scenarios) + [a.right_col_width]
    fig = plt.figure(figsize=(3.2*n_cols + 2.5, 3.2*n_rows + 1.5))
    fig.subplots_adjust(top=0.90)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=width_ratios, wspace=0.07, hspace=0.14)

    # Right column sub-grid: top (OBS) : bottom (legend)
    ax_obs = fig.add_subplot(gs[:, -1])  # allocate entire right column then split
    ax_obs.axis("off") 
    sub = gs[:, -1].subgridspec(3, 1, height_ratios=[2.0, 0.1, 1.0], hspace=0.25)
    ax_obs_map = fig.add_subplot(sub[0,0])
    ax_blank   = fig.add_subplot(sub[1,0]); ax_blank.axis("off")
    ax_legend  = fig.add_subplot(sub[2,0])

    # Plot OBS map
    plot_panel(
        ax_obs_map, obs_rgb, obs_lon, obs_lat,
        title=f"Observed (CHIRPS + CHIRTS)\n{a.obs_start} \u2192 {a.obs_end}",  # ?
        overlay=a.overlay_shp if a.overlay_shp else None,
        show_xlabel=True, show_ylabel=True
    )

    # Legend
    draw_legend(ax_legend, color_grid, Tedges, Pedges)

    # Column titles (scenarios)
    for j, scen in enumerate(a.scenarios):
        ax = fig.add_subplot(gs[0, j]); ax.axis("off")
        ax.set_title(scen, fontsize=11, pad=2)

    # Row labels (models) and the grid
    for i, model in enumerate(a.models):
        # left margin label via an invisible axis spanning first cell
        lab_ax = fig.add_subplot(gs[i, 0]); lab_ax.axis("off")
        lab_ax.text(-0.15, 0.5, model, va="center", ha="right", rotation=90, fontsize=11, transform=lab_ax.transAxes)

        for j, scen in enumerate(a.scenarios):
            start_end = scen_periods.get(scen, None)
            if start_end:
                s, e = start_end.split(",")
            else:
                s = a.obs_start; e = a.obs_end  # fallback

            # build globs from templates
            pr_glob   = a.gcm_pr_tpl.format(model=model, scenario=scen)
            tmin_glob = a.gcm_tmin_tpl.format(model=model, scenario=scen)
            tmax_glob = a.gcm_tmax_tpl.format(model=model, scenario=scen)

            ax = fig.add_subplot(gs[i, j])

            try:
                rgb, lon, lat = compute_rgb(pr_glob, tmin_glob, tmax_glob, a.engine, a.bbox, s, e,
                                            Tedges, Pedges, color_grid)
                plot_panel(
                    ax, rgb, lon, lat, title=None,
                    overlay=a.overlay_shp if a.overlay_shp else None,
                    show_xlabel=(i == len(a.models) - 1),   # bottom row
                    show_ylabel=(j == 0)                    # first column
                )
            except Exception as ex:
                # draw a placeholder "missing" panel
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color="0.6", fontsize=10, transform=ax.transAxes)
                ax.set_facecolor("white")
                print(f"[WARN] {model}/{scen} failed: {ex}")

    # Titles
    fig.suptitle(a.title, fontsize=16, y=0.980)
    if a.subtitle:
        fig.text(0.5, 0.940, a.subtitle, ha="center", va="top", fontsize=11)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("[DONE] wrote", a.out)

if __name__ == "__main__":
    main()
