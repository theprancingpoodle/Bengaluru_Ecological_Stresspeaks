"""
Build checkpoint assets for Bengaluru Ecological Stresspeaks.

This script uses existing local evidence only:
- Landsat LST / anomaly / NDVI / MNDWI / NDBI rasters in data/landsat
- ward_master.geojson in outputs/tables

It creates sample-only ecological stress assets and three checkpoint visuals:
- 2D ecological stress index ward map
- bivariate heat gain x vegetation loss map
- contour / DEM-style ecological stress map

The 3D render is handled by scripts/19_render_ecological_stresspeaks_blender.py
using the OBJ and drape JSON written here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, zoom


ROOT = Path(__file__).resolve().parents[1]
LANDSAT = ROOT / "data" / "landsat"
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "stresspeaks"
OUT_MAPS = ROOT / "outputs" / "ward_maps"

BG = "#ece9e3"
PANEL = "#f5f1e8"
INK = "#282421"
MUTE = "#756e66"
RUST = "#9c3b25"

FORMULA = {
    "lst_gain_1995": 0.30,
    "present_lst_anomaly": 0.20,
    "ndvi_loss_1995": 0.20,
    "water_decline_1995": 0.15,
    "ndbi_gain_1995": 0.15,
}

PALETTE_A = ["#f4eadb", "#dfc78d", "#c98743", "#8c3e2e", "#421d1d"]
PALETTE_B = ["#dfe8df", "#d9ca96", "#bf6f43", "#7c2830", "#2b1718"]
PALETTE_C = ["#b9cdc6", "#e6dac2", "#d18a48", "#8c2f36", "#251816"]


def read_raster(name: str):
    path = LANDSAT / name
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True).astype("float32").filled(np.nan)
        profile = ds.profile
        bounds = ds.bounds
    return data, profile, bounds


def zscore(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask & np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype="float32")
    if vals.size == 0:
        return out
    med = float(np.nanmedian(vals))
    sd = float(np.nanstd(vals))
    if not sd or not np.isfinite(sd):
        sd = 1.0
    z = (arr - med) / sd
    z = np.clip(z, -3.0, 3.0)
    out[mask] = z[mask]
    return out


def robust_norm(arr: np.ndarray, mask: np.ndarray, lo_pct=2, hi_pct=98) -> tuple[np.ndarray, tuple[float, float]]:
    vals = arr[mask & np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype="float32")
    if vals.size == 0:
        return out, (0.0, 1.0)
    lo, hi = np.nanpercentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out[mask] = np.clip((arr[mask] - lo) / (hi - lo), 0, 1)
    return out, (float(lo), float(hi))


def write_raster(path: Path, arr: np.ndarray, profile: dict):
    prof = dict(profile)
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr.astype("float32"), 1)


def write_png16(path: Path, arr01: np.ndarray):
    h = np.clip(np.nan_to_num(arr01, nan=0.0), 0, 1)
    Image.fromarray((h * 65535 + 0.5).astype("uint16"), mode="I;16").save(path)


def cmap(hexes, name="stress"):
    return LinearSegmentedColormap.from_list(name, hexes)


def compute_pixel_stress():
    lst_present, profile, bounds = read_raster("lst_present_median.tif")
    lst_1990s, _, _ = read_raster("lst_1990s_median.tif")
    anom_present, _, _ = read_raster("anomaly_present_median.tif")
    ndvi_present, _, _ = read_raster("ndvi_present.tif")
    ndvi_1990s, _, _ = read_raster("ndvi_1990s.tif")
    mndwi_present, _, _ = read_raster("mndwi_present.tif")
    mndwi_1990s, _, _ = read_raster("mndwi_1990s.tif")
    ndbi_present, _, _ = read_raster("ndbi_present.tif")
    ndbi_1990s, _, _ = read_raster("ndbi_1990s.tif")

    components = {
        "lst_gain_1995": lst_present - lst_1990s,
        "present_lst_anomaly": anom_present,
        "ndvi_loss_1995": ndvi_1990s - ndvi_present,
        "water_decline_1995": mndwi_1990s - mndwi_present,
        "ndbi_gain_1995": ndbi_present - ndbi_1990s,
    }
    mask = np.ones(lst_present.shape, dtype=bool)
    for arr in components.values():
        mask &= np.isfinite(arr)
    wards = gpd.read_file(TABLES / "ward_master.geojson").to_crs(profile["crs"])
    boundary_mask = rasterize(
        [(wards.geometry.union_all(), 1)],
        out_shape=lst_present.shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    mask &= boundary_mask

    stress = np.zeros(lst_present.shape, dtype="float32")
    comp_z = {}
    for key, weight in FORMULA.items():
        comp_z[key] = zscore(components[key], mask)
        stress += weight * np.nan_to_num(comp_z[key], nan=0.0)
    stress[~mask] = np.nan
    stress_norm, scale = robust_norm(stress, mask)

    OUT.mkdir(parents=True, exist_ok=True)
    write_raster(OUT / "ecological_stress_index_pixel.tif", stress, profile)
    write_raster(OUT / "ecological_stress_index_pixel_norm.tif", stress_norm, profile)

    meta = {
        "title": "Bengaluru Ecological Stress Index",
        "formula": FORMULA,
        "component_definitions": {
            "lst_gain_1995": "LST present median minus LST 1990s median",
            "present_lst_anomaly": "present LST anomaly relative to present city mean",
            "ndvi_loss_1995": "NDVI 1990s minus NDVI present",
            "water_decline_1995": "MNDWI 1990s minus MNDWI present",
            "ndbi_gain_1995": "NDBI present minus NDBI 1990s",
        },
        "zscore": "median-centered standard deviation z-score, clipped to [-3, 3]",
        "normalization_for_heightmap": f"robust percentile scale 2-98 = {scale}",
        "note": "Height represents composite ecological stress, not real elevation.",
    }
    (OUT / "ecological_stress_index_method.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return stress, stress_norm, components, profile, bounds, mask


def classify_categories(row):
    high_heat = row["lst_gain_1995"] >= 0.75
    high_anom = row["present_lst_anomaly"] >= 0.75
    high_veg = row["ndvi_loss_1995"] >= 0.75
    high_water = row["water_decline_1995"] >= 0.75
    high_built = row["ndbi_gain_1995"] >= 0.75
    n = sum([high_heat or high_anom, high_veg, high_water, high_built])
    if n >= 3:
        return "Extreme combined stress"
    if (high_heat or high_anom) and high_veg and high_water:
        return "Heat + vegetation + water stress"
    if (high_heat or high_anom) and high_veg:
        return "Heat + vegetation loss"
    if (high_heat or high_anom) and high_water:
        return "Heat + water decline"
    if (high_heat or high_anom) and high_built:
        return "Heat + built-up expansion"
    if high_veg or high_water or high_built:
        return "Ecological loss signal"
    return "Lower relative stress"


def compute_ward_stress():
    gdf = gpd.read_file(TABLES / "ward_master.geojson")
    numeric = [
        "lst_change_1995_to_present",
        "lst_anom_mean_present",
        "ndvi_change_1995_to_present",
        "water_change_1995_to_present",
        "ndbi_mean_1990s",
        "ndbi_mean_present",
    ]
    for col in numeric:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    gdf["lst_gain_1995_raw"] = gdf["lst_change_1995_to_present"]
    gdf["present_lst_anomaly_raw"] = gdf["lst_anom_mean_present"]
    gdf["ndvi_loss_1995_raw"] = -gdf["ndvi_change_1995_to_present"]
    gdf["water_decline_1995_raw"] = -gdf["water_change_1995_to_present"]
    gdf["ndbi_gain_1995_raw"] = gdf["ndbi_mean_present"] - gdf["ndbi_mean_1990s"]

    score = np.zeros(len(gdf), dtype="float64")
    for key, weight in FORMULA.items():
        raw = f"{key}_raw"
        vals = pd.to_numeric(gdf[raw], errors="coerce")
        med = vals.median()
        sd = vals.std(ddof=0)
        if not sd or not np.isfinite(sd):
            sd = 1.0
        z = ((vals - med) / sd).clip(-3, 3)
        gdf[f"{key}_z"] = z
        score += weight * z.fillna(0)
        # percentile rank for categories and maps
        gdf[key] = vals.rank(pct=True)

    gdf["ecological_stress_index"] = score
    gdf["ecological_stress_rank"] = gdf["ecological_stress_index"].rank(ascending=False, method="min").astype(int)
    gdf["ecological_stress_category"] = gdf.apply(classify_categories, axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MAPS.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_csv(TABLES / "ecological_stress_index.csv", index=False)
    gdf.to_file(TABLES / "ecological_stress_index.geojson", driver="GeoJSON")
    return gdf


def make_heightmaps(stress_norm, mask, profile, bounds):
    variants = {
        "subtle": (1.05, 0.18),
        "balanced": (1.28, 0.10),
        "dramatic": (1.55, 0.06),
    }
    pal = cmap(PALETTE_B, "stress_b")
    for name, (exp, base) in variants.items():
        # Light smoothing removes raster noise but keeps local stress ridges.
        filled = np.nan_to_num(stress_norm, nan=0.0)
        smooth = gaussian_filter(filled, sigma=2.0)
        h = np.where(mask, base + (1 - base) * np.power(np.clip(smooth, 0, 1), exp), 0.0)
        write_png16(OUT / f"ecological_stress_heightmap_{name}.png", h)
        preview = np.where(mask, h, np.nan)
        fig, ax = plt.subplots(figsize=(7, 7), dpi=160, facecolor=BG)
        ax.imshow(preview, cmap=pal, origin="upper")
        ax.set_title(f"Ecological stress heightmap - {name}", fontsize=12, color=INK)
        ax.set_axis_off()
        fig.savefig(OUT / f"preview_heightmap_{name}.png", bbox_inches="tight", pad_inches=0.08, facecolor=BG)
        plt.close(fig)

    meta = {
        "crs": str(profile["crs"]),
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "formula": FORMULA,
    }
    (OUT / "ecological_stress_heightmap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_obj_mesh(stress_norm, mask, bounds, mesh_max=520):
    h0, w0 = stress_norm.shape
    scale = mesh_max / max(h0, w0)
    hh = max(2, int(round(h0 * scale)))
    ww = max(2, int(round(w0 * scale)))
    height = zoom(np.nan_to_num(stress_norm, nan=0.0), (hh / h0, ww / w0), order=1)
    mm = zoom(mask.astype("float32"), (hh / h0, ww / w0), order=1) > 0.55
    height = gaussian_filter(height, sigma=1.2)
    z = np.where(mm, 0.045 + 0.84 * np.power(np.clip(height, 0, 1), 1.25), np.nan)

    real_w = bounds.right - bounds.left
    real_h = bounds.top - bounds.bottom
    sx = 2.0
    sy = 2.0 * real_h / real_w

    verts = []
    vid = np.full((hh, ww), -1, dtype=int)
    for i in range(hh):
        y = sy * (0.5 - i / (hh - 1))
        for j in range(ww):
            if not mm[i, j]:
                continue
            x = sx * (j / (ww - 1) - 0.5)
            vid[i, j] = len(verts) + 1
            verts.append((x, y, float(z[i, j])))

    faces = []
    cell = np.zeros((hh - 1, ww - 1), dtype=bool)
    for i in range(hh - 1):
        for j in range(ww - 1):
            if mm[i, j] and mm[i + 1, j] and mm[i, j + 1] and mm[i + 1, j + 1]:
                faces.append((vid[i, j], vid[i, j + 1], vid[i + 1, j + 1], vid[i + 1, j]))
                cell[i, j] = True

    def add_base_vertex(top_id):
        x, y, _ = verts[top_id - 1]
        verts.append((x, y, 0.0))
        return len(verts)

    for i in range(hh - 1):
        for j in range(ww - 1):
            if not cell[i, j]:
                continue
            # north, east, south, west edges
            edges = [
                ((i, j), (i, j + 1), i == 0 or not cell[i - 1, j]),
                ((i, j + 1), (i + 1, j + 1), j == ww - 2 or not cell[i, j + 1]),
                ((i + 1, j + 1), (i + 1, j), i == hh - 2 or not cell[i + 1, j]),
                ((i + 1, j), (i, j), j == 0 or not cell[i, j - 1]),
            ]
            for a, b, boundary in edges:
                if not boundary:
                    continue
                va, vb = vid[a], vid[b]
                ba, bb = add_base_vertex(va), add_base_vertex(vb)
                faces.append((va, vb, bb, ba))

    obj_path = OUT / "ecological_stress_mesh_balanced.obj"
    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Bengaluru Ecological Stresspeaks mesh - generated from stress index\n")
        for x, y, zz in verts:
            f.write(f"v {x:.6f} {y:.6f} {zz:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(v) for v in face) + "\n")

    mesh_meta = {"sx": sx, "sy": sy, "mesh_height": hh, "mesh_width": ww, "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top]}
    (OUT / "ecological_stress_mesh_meta.json").write_text(json.dumps(mesh_meta, indent=2), encoding="utf-8")
    return obj_path, mesh_meta


def xy_to_scene(x, y, bounds, sx, sy):
    u = (x - bounds.left) / (bounds.right - bounds.left)
    v = (bounds.top - y) / (bounds.top - bounds.bottom)
    return sx * (u - 0.5), sy * (0.5 - v)


def height_sampler(stress_norm, mask, bounds):
    h, w = stress_norm.shape
    smooth = gaussian_filter(np.nan_to_num(stress_norm, nan=0.0), sigma=2.0)

    def sample(x, y):
        u = (x - bounds.left) / (bounds.right - bounds.left)
        v = (bounds.top - y) / (bounds.top - bounds.bottom)
        col = int(np.clip(round(u * (w - 1)), 0, w - 1))
        row = int(np.clip(round(v * (h - 1)), 0, h - 1))
        if not mask[row, col]:
            best = None
            best_dist = None
            for radius in range(1, 9):
                r0, r1 = max(0, row - radius), min(h, row + radius + 1)
                c0, c1 = max(0, col - radius), min(w, col + radius + 1)
                ys, xs = np.where(mask[r0:r1, c0:c1])
                if ys.size:
                    ys = ys + r0
                    xs = xs + c0
                    dist = (ys - row) ** 2 + (xs - col) ** 2
                    idx = int(np.argmin(dist))
                    best = (int(ys[idx]), int(xs[idx]))
                    best_dist = float(dist[idx])
                    break
            if best is None or best_dist is None:
                return 0.0
            row, col = best
        return 0.045 + 0.84 * float(np.power(np.clip(smooth[row, col], 0, 1), 1.25))

    return sample


def make_drape_json(gdf, stress_norm, mask, bounds, mesh_meta):
    sx, sy = mesh_meta["sx"], mesh_meta["sy"]
    sample_h = height_sampler(stress_norm, mask, bounds)
    g = gdf.to_crs("EPSG:32643").copy()
    g["geometry"] = g.geometry.simplify(85, preserve_topology=True)
    top_names = set(g.nsmallest(8, "ecological_stress_rank")["ward_name"].astype(str))

    def ring(poly):
        pts = []
        for x, y in poly.exterior.coords:
            px, py = xy_to_scene(x, y, bounds, sx, sy)
            pts.append([round(px, 5), round(py, 5), round(sample_h(x, y) + 0.015, 5)])
        return pts

    all_lines, top_lines, outline_lines = [], [], []
    for _, row in g.iterrows():
        geom = row.geometry
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        ward_lines = [ring(p) for p in polys if not p.is_empty]
        all_lines.extend(ward_lines)
        if str(row["ward_name"]) in top_names:
            top_lines.extend(ward_lines)

    outline = g.geometry.union_all()
    polys = outline.geoms if outline.geom_type == "MultiPolygon" else [outline]
    outline_lines.extend([ring(p) for p in polys if not p.is_empty])

    anchors = []
    for _, row in g.nsmallest(8, "ecological_stress_rank").iterrows():
        pt = row.geometry.representative_point()
        px, py = xy_to_scene(pt.x, pt.y, bounds, sx, sy)
        anchors.append({
            "name": str(row["ward_name"]),
            "rank": int(row["ecological_stress_rank"]),
            "score": float(row["ecological_stress_index"]),
            "x": round(px, 5),
            "y": round(py, 5),
            "z": round(sample_h(pt.x, pt.y) + 0.04, 5),
        })

    data = {
        "all_polylines": all_lines,
        "top_polylines": top_lines,
        "outline_polylines": outline_lines,
        "anchors": anchors,
        "sx": sx,
        "sy": sy,
    }
    path = OUT / "ecological_stress_drape.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def add_scalebar(ax, length_km=10, xfrac=0.11, yfrac=0.08):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xb = x0 + (x1 - x0) * xfrac
    yb = y0 + (y1 - y0) * yfrac
    ax.plot([xb, xb + length_km * 1000], [yb, yb], color=INK, lw=2.4, solid_capstyle="butt")
    ax.text(xb + length_km * 500, yb + (y1 - y0) * 0.02, f"{length_km} km",
            ha="center", va="top", fontsize=8.5, color=INK)


def add_north(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - (x1 - x0) * 0.075
    y = y1 - (y1 - y0) * 0.18
    ax.add_patch(FancyArrowPatch((x, y - (y1 - y0) * 0.075), (x, y),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.6, color=INK))
    ax.text(x, y - (y1 - y0) * 0.10, "N", ha="center", va="center", fontsize=10, weight="bold", color=INK)


def setup_map_ax(gdf, fig):
    ax = fig.add_axes([0.045, 0.155, 0.91, 0.655], facecolor=PANEL)
    ax.set_axis_off()
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - (maxx - minx) * 0.52, maxx + (maxx - minx) * 0.54)
    ax.set_ylim(miny - (maxy - miny) * 0.16, maxy + (maxy - miny) * 0.18)
    return ax


def external_labels(ax, gdf, names):
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    midx = (minx + maxx) / 2

    side_hint = {
        "J P Park": "left",
        "Nilasandra": "left",
        "Gurappanapalya": "left",
        "Gali Anjenaya Temple ward": "left",
        "T Dasarahalli": "left",
        "Peenya Industrial Area": "left",
        "Kushal Nagar": "left",
        "Muneshwara Nagar": "left",
        "Varthuru": "right",
        "Marathahalli": "right",
        "Kadugondanahalli": "right",
        "Ejipura": "right",
    }

    grouped = {"left": [], "right": []}
    for order, name in enumerate(names):
        row = gdf[gdf["ward_name"].astype(str) == name]
        if row.empty:
            continue
        row = row.iloc[0]
        pt = row.geometry.representative_point()
        side = side_hint.get(name, "right" if pt.x >= midx else "left")
        grouped[side].append((order, name, row, pt))

    label_x = {"left": minx - width * 0.28, "right": maxx + width * 0.28}
    elbow_x = {"left": minx - width * 0.055, "right": maxx + width * 0.055}
    leader_end_x = {"left": label_x["left"] + width * 0.025, "right": label_x["right"] - width * 0.025}

    for side, rows in grouped.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda item: item[3].y, reverse=True)
        top_y = maxy - height * 0.02
        bottom_y = miny + height * 0.06
        slots = np.linspace(top_y, bottom_y, len(rows)) if len(rows) > 1 else np.array([(top_y + bottom_y) / 2])
        for slot_y, (_order, name, row, pt) in zip(slots, rows):
            ha = "right" if side == "left" else "left"
            text = f"{name}\n#{int(row['ecological_stress_rank'])} ecological stress"
            ax.plot(
                [pt.x, elbow_x[side], leader_end_x[side]],
                [pt.y, slot_y, slot_y],
                color="#26201c",
                lw=1.05,
                alpha=0.88,
                linestyle=(0, (1.1, 2.6)),
                solid_capstyle="round",
                zorder=9,
                clip_on=False,
            )
            ax.scatter([pt.x], [pt.y], s=94, facecolor=BG, edgecolor="#211916",
                       linewidth=1.15, zorder=12, clip_on=False)
            ax.scatter([pt.x], [pt.y], s=34, facecolor="#2a1717", edgecolor="#2a1717",
                       linewidth=0.6, zorder=13, clip_on=False)
            ax.text(
                label_x[side],
                slot_y,
                text,
                ha=ha,
                va="center",
                fontsize=8.1,
                color=INK,
                linespacing=1.12,
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.23", fc=BG, ec="#9f9488", lw=0.62, alpha=0.97),
                clip_on=False,
            )


def add_title_block(fig, title, subtitle_lines, note=None):
    title_lines = title.count("\n") + 1
    fig.text(0.058, 0.958, title, color=RUST, fontsize=22.8,
             family="serif", weight="bold", va="top")
    y = 0.918 - (title_lines - 1) * 0.038
    for line in subtitle_lines:
        fig.text(0.059, y, line, color=INK, fontsize=12.8, weight="bold", va="top")
        y -= 0.024
    if note:
        fig.text(0.059, y - 0.009, note, color=MUTE, fontsize=9.4, va="top")


def map_ecological_stress(gdf):
    g = gdf.to_crs("EPSG:32643")
    pal = cmap(PALETTE_B, "stress_map")
    norm = Normalize(g["ecological_stress_index"].quantile(0.02), g["ecological_stress_index"].quantile(0.98))
    fig = plt.figure(figsize=(10.5, 13), dpi=170, facecolor=BG)
    ax = setup_map_ax(g, fig)
    g.plot(ax=ax, column="ecological_stress_index", cmap=pal, norm=norm,
           linewidth=0.25, edgecolor="#817870")
    top = g.nsmallest(8, "ecological_stress_rank")
    top.boundary.plot(ax=ax, color="#261714", linewidth=1.15)
    label_names = g.nsmallest(5, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_labels(ax, g, label_names)
    add_scalebar(ax, xfrac=0.08, yfrac=0.07)
    add_north(ax)

    cax = fig.add_axes([0.28, 0.115, 0.48, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=pal), cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor("#b7ac9f")
    cb.ax.tick_params(labelsize=8, colors=INK, length=0)
    cb.set_label("Ecological stress index (weighted z-score)", fontsize=8.5, color=INK)

    add_title_block(
        fig,
        "Bengaluru Ecological\nStresspeaks",
        [
            "Heat gain, vegetation loss, water decline,",
            "and built-up expansion mapped together",
        ],
        "Ward-level composite index from Landsat LST, NDVI, MNDWI and NDBI.",
    )
    fig.text(0.07, 0.061,
             "Formula: 0.30*z(LST gain since ~1995) + 0.20*z(present anomaly) + "
             "0.20*z(NDVI loss) + 0.15*z(water decline) + 0.15*z(NDBI gain).",
             fontsize=8.4, color=MUTE, va="top")
    fig.text(0.07, 0.039,
             "Source: Landsat C2 L2 via Microsoft Planetary Computer; BBMP ward boundaries. Checkpoint sample.",
             fontsize=8.4, color=MUTE, va="top")
    out = OUT_MAPS / "sample_ecological_stress_index_ward_map.png"
    fig.savefig(out, facecolor=BG)
    fig.savefig(out.with_suffix(".pdf"), facecolor=BG)
    plt.close(fig)
    return out


def bivariate_palette():
    # rows = vegetation loss low to high, columns = heat gain low to high
    return np.array([
        ["#eef1e8", "#d8c89d", "#c27c45"],
        ["#cfd4b2", "#c59a61", "#9d5039"],
        ["#9fb198", "#806246", "#3e1d22"],
    ])


def map_bivariate(gdf):
    g = gdf.to_crs("EPSG:32643").copy()
    heat = pd.qcut(g["lst_gain_1995_raw"], 3, labels=False, duplicates="drop")
    veg = pd.qcut(g["ndvi_loss_1995_raw"], 3, labels=False, duplicates="drop")
    g["bivar"] = veg.astype(int) * 3 + heat.astype(int)
    colors = bivariate_palette().reshape(-1)
    cmap_bi = ListedColormap(colors)

    fig = plt.figure(figsize=(10.5, 13), dpi=170, facecolor=BG)
    ax = setup_map_ax(g, fig)
    g.plot(ax=ax, column="bivar", cmap=cmap_bi, vmin=0, vmax=8,
           linewidth=0.24, edgecolor="#7e766f")
    g.nsmallest(6, "ecological_stress_rank").boundary.plot(ax=ax, color="#241512", linewidth=1.05)
    label_names = g.nsmallest(4, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_labels(ax, g, label_names)
    add_scalebar(ax, xfrac=0.08, yfrac=0.07)
    add_north(ax)

    leg_ax = fig.add_axes([0.36, 0.098, 0.16, 0.105])
    leg_ax.set_axis_off()
    pal = bivariate_palette()
    for r in range(3):
        for c in range(3):
            leg_ax.add_patch(Rectangle((c, r), 1, 1, facecolor=pal[r, c], edgecolor=BG, lw=1))
    leg_ax.set_xlim(0, 3)
    leg_ax.set_ylim(0, 3)
    leg_ax.text(1.5, -0.38, "Heat gain ->", ha="center", va="top", fontsize=8.4, color=INK)
    leg_ax.text(-0.32, 1.5, "NDVI loss ->", ha="right", va="center", rotation=90, fontsize=8.4, color=INK)
    leg_ax.text(0.5, 3.15, "low", ha="center", fontsize=7.2, color=MUTE)
    leg_ax.text(2.5, 3.15, "high", ha="center", fontsize=7.2, color=MUTE)

    add_title_block(
        fig,
        "Bivariate Ecological Stress",
        ["Heat gain x vegetation loss"],
        "Darker wards combine stronger warming since ~1995 with larger NDVI decline.",
    )
    fig.text(0.07, 0.049,
             "Source: Landsat C2 L2 via Microsoft Planetary Computer; tertile classes per BBMP ward. Checkpoint sample.",
             fontsize=8.4, color=MUTE, va="top")
    out = OUT_MAPS / "sample_bivariate_heatgain_vegetationloss.png"
    fig.savefig(out, facecolor=BG)
    fig.savefig(out.with_suffix(".pdf"), facecolor=BG)
    plt.close(fig)
    return out


def map_contour_dem(stress_norm, mask, bounds, gdf):
    pal = cmap(PALETTE_C, "stress_dem")
    data = np.where(mask, gaussian_filter(np.nan_to_num(stress_norm, nan=0), sigma=3.2), np.nan)
    gy, gx = np.gradient(np.nan_to_num(data, nan=np.nanmean(data)))
    shade = 1 - np.clip(np.hypot(gx, gy) * 8, 0, 0.36)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    g = gdf.to_crs("EPSG:32643")
    fig = plt.figure(figsize=(10.5, 13), dpi=170, facecolor=BG)
    ax = setup_map_ax(g, fig)
    ax.imshow(data, extent=extent, origin="upper", cmap=pal, vmin=0, vmax=1, alpha=0.96)
    ax.imshow(shade, extent=extent, origin="upper", cmap="gray", alpha=0.16, vmin=0.60, vmax=1.0)
    levels = np.linspace(0.18, 0.92, 10)
    ax.contour(data, levels=levels, extent=extent, origin="upper",
               colors="#3a2a24", linewidths=0.28, alpha=0.30)
    g.boundary.plot(ax=ax, color="#42332c", linewidth=0.25, alpha=0.32)
    g.nsmallest(6, "ecological_stress_rank").boundary.plot(ax=ax, color="#231412", linewidth=1.0)
    label_names = g.nsmallest(3, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_labels(ax, g, label_names)
    add_scalebar(ax, xfrac=0.08, yfrac=0.07)
    add_north(ax)

    add_title_block(
        fig,
        "Ecological Stress Contours",
        ["DEM-style view of Bengaluru's combined stress surface"],
        "Contours and shaded relief show where urban-ecological pressures rise together.",
    )
    fig.text(0.07, 0.049,
             "Height/contours use the composite ecological stress index, not real elevation. Checkpoint sample.",
             fontsize=8.4, color=MUTE, va="top")
    out = OUT / "sample_ecological_stress_contour_dem.png"
    fig.savefig(out, facecolor=BG)
    fig.savefig(out.with_suffix(".pdf"), facecolor=BG)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build ecological stress sample assets.")
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="Rebuild stress rasters, heightmaps, mesh, and drape JSON without overwriting approved 2D map samples.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MAPS.mkdir(parents=True, exist_ok=True)
    stress, stress_norm, components, profile, bounds, mask = compute_pixel_stress()
    gdf = compute_ward_stress()
    make_heightmaps(stress_norm, mask, profile, bounds)
    obj_path, mesh_meta = write_obj_mesh(stress_norm, mask, bounds)
    drape_path = make_drape_json(gdf, stress_norm, mask, bounds, mesh_meta)
    if args.assets_only:
        print(f"Wrote {obj_path}")
        print(f"Wrote {drape_path}")
        return
    m1 = map_ecological_stress(gdf)
    m2 = map_bivariate(gdf)
    m3 = map_contour_dem(stress_norm, mask, bounds, gdf)
    print(f"Wrote {obj_path}")
    print(f"Wrote {drape_path}")
    print(f"Wrote {m1}")
    print(f"Wrote {m2}")
    print(f"Wrote {m3}")


if __name__ == "__main__":
    main()
