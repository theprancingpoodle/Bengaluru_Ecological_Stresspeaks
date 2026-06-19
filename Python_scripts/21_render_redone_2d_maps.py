"""
Redone 2D checkpoint maps for Bengaluru Ecological Stresspeaks.

This script uses the stress-index outputs already created by
scripts/18_build_ecological_stress_samples.py, but replaces the cartographic
layout from scratch:
- one uniform paper background, including the contour map
- external callout boxes only, with halo anchors and dotted leader lines
- larger typography
- method + source footer on every map
- bivariate legends placed in the right-side whitespace
- all six pairings among heat gain, vegetation loss, water loss, built-up gain
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
WARD_MAPS = ROOT / "outputs" / "ward_maps"
STRESS = ROOT / "outputs" / "stresspeaks"

BG = "#ece9e3"
INK = "#282421"
MUTE = "#6e675f"
RUST = "#9c3b25"
WARD_EDGE = "#837b72"
TOP_EDGE = "#241512"
OUTER_EDGE = "#4f453d"
LABEL_EDGE = "#9a9085"
ANCHOR_FILL = "#00d5ff"
ANCHOR_OUTLINE = "#18120f"
ANCHOR_HALO = "#fffaf0"
LEADER = "#00a9c7"
LEADER_SHADOW = "#2c211c"

STRESS_PALETTE = ["#dfe8df", "#d9ca96", "#bf6f43", "#7c2830", "#2b1718"]
CONTOUR_PALETTE = ["#b9cdc6", "#e6dac2", "#d18a48", "#8c2f36", "#251816"]
BIVAR_PALETTES = {
    "sample_bivariate_heatgain_vegetationloss": np.array([
        ["#eef1e8", "#d8c89d", "#c27c45"],
        ["#cfd4b2", "#c59a61", "#9d5039"],
        ["#9fb198", "#806246", "#3e1d22"],
    ]),
    "sample_bivariate_heatgain_waterloss": np.array([
        ["#e8f0ee", "#d7c696", "#c86f42"],
        ["#bfd2cd", "#c39561", "#984d42"],
        ["#7ca4a4", "#705f4d", "#321f2a"],
    ]),
    "sample_bivariate_heatgain_builtupgain": np.array([
        ["#eef0e7", "#dbc79c", "#c98749"],
        ["#ced1c1", "#ad8b64", "#8b4f3f"],
        ["#a3aaa1", "#6e6258", "#2d2422"],
    ]),
    "sample_bivariate_vegetationloss_waterloss": np.array([
        ["#eef2e9", "#c7d2bd", "#7fa69d"],
        ["#d8c999", "#a7966d", "#5e7872"],
        ["#c78145", "#8d6046", "#302425"],
    ]),
    "sample_bivariate_vegetationloss_builtupgain": np.array([
        ["#f0f2e9", "#cfd6b4", "#a1b391"],
        ["#dccb95", "#aa9564", "#786c4f"],
        ["#c97845", "#96553e", "#35251f"],
    ]),
    "sample_bivariate_waterloss_builtupgain": np.array([
        ["#eef1ec", "#c8d6ce", "#87a8a1"],
        ["#ddc898", "#aa9067", "#78614f"],
        ["#ca8448", "#985b42", "#33242a"],
    ]),
}

METHOD_STRESS = (
    "Method: ward score = 0.30*z(LST gain since ~1995) + 0.20*z(present LST anomaly) "
    "+ 0.20*z(NDVI loss) + 0.15*z(MNDWI/water decline) + 0.15*z(NDBI/built-up gain)."
)
METHOD_BIVAR = (
    "Method: each variable is split into ward-level tertiles; the 3x3 legend shows the "
    "intersection of low, medium, and high ecological-pressure classes."
)
METHOD_CONTOUR = (
    "Method: pixel-level composite stress index uses the same weighted z-score formula; "
    "contours are drawn from a lightly smoothed stress raster. Height/contours are not real elevation."
)
SOURCE = "Source: Landsat C2 L2 via Microsoft Planetary Computer; BBMP ward boundaries. Checkpoint sample."
OUT_PREFIX = "redone_v6_"


VARIABLES = {
    "heat_gain": {
        "column": "lst_gain_1995_raw",
        "short": "heat gain",
        "title": "Heat gain",
        "subtitle": "LST increase since ~1995",
    },
    "vegetation_loss": {
        "column": "ndvi_loss_1995_raw",
        "short": "vegetation loss",
        "title": "Vegetation loss",
        "subtitle": "NDVI decline since ~1995",
    },
    "water_loss": {
        "column": "water_decline_1995_raw",
        "short": "water loss",
        "title": "Water loss",
        "subtitle": "MNDWI/water decline since ~1995",
    },
    "builtup_gain": {
        "column": "ndbi_gain_1995_raw",
        "short": "built-up gain",
        "title": "Built-up gain",
        "subtitle": "NDBI increase since ~1995",
    },
}

BIVAR_PAIRS = [
    ("heat_gain", "vegetation_loss", "sample_bivariate_heatgain_vegetationloss"),
    ("heat_gain", "water_loss", "sample_bivariate_heatgain_waterloss"),
    ("heat_gain", "builtup_gain", "sample_bivariate_heatgain_builtupgain"),
    ("vegetation_loss", "water_loss", "sample_bivariate_vegetationloss_waterloss"),
    ("vegetation_loss", "builtup_gain", "sample_bivariate_vegetationloss_builtupgain"),
    ("water_loss", "builtup_gain", "sample_bivariate_waterloss_builtupgain"),
]


def cmap(hexes: list[str], name: str) -> LinearSegmentedColormap:
    cm = LinearSegmentedColormap.from_list(name, hexes)
    cm.set_bad((0, 0, 0, 0))
    return cm


def read_data() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(TABLES / "ecological_stress_index.geojson")
    for cfg in VARIABLES.values():
        gdf[cfg["column"]] = pd.to_numeric(gdf[cfg["column"]], errors="coerce")
    gdf["ecological_stress_index"] = pd.to_numeric(gdf["ecological_stress_index"], errors="coerce")
    gdf["ecological_stress_rank"] = pd.to_numeric(gdf["ecological_stress_rank"], errors="coerce").astype(int)
    return gdf.to_crs("EPSG:32643")


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG)
    fig.savefig(path.with_suffix(".pdf"), facecolor=BG)
    plt.close(fig)
    return path


def setup_figure(title: str, subtitle: str, note: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.2, 13.4), dpi=180, facecolor=BG)
    fig.text(0.055, 0.955, title, color=RUST, fontsize=27, family="serif", weight="bold", va="top")
    y = 0.905
    for line in textwrap.wrap(subtitle, width=72)[:2]:
        fig.text(0.057, y, line, color=INK, fontsize=15.4, weight="bold", va="top")
        y -= 0.024
    if note:
        wrapped = textwrap.wrap(note, width=92)
        for i, line in enumerate(wrapped[:2]):
            fig.text(0.057, y - 0.006 - i * 0.020, line, color=MUTE, fontsize=11.3, va="top")
    ax = fig.add_axes([0.030, 0.170, 0.845, 0.655], facecolor=BG)
    ax.set_axis_off()
    return fig, ax


def set_map_limits(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    ax.set_xlim(minx - width * 0.055, maxx + width * 0.105)
    ax.set_ylim(miny - height * 0.105, maxy + height * 0.115)
    ax.set_aspect("equal")
    return minx, miny, maxx, maxy, width, height


def add_footer(fig: plt.Figure, method: str) -> None:
    method_lines = textwrap.wrap(method, width=142)
    for i, line in enumerate(method_lines[:2]):
        fig.text(0.057, 0.062 - i * 0.018, line, fontsize=10.6, color=MUTE, va="top")
    fig.text(0.057, 0.027, SOURCE, fontsize=10.6, color=MUTE, va="top")


def add_scalebar(ax: plt.Axes, bounds: tuple[float, float, float, float, float, float], length_km: int = 10) -> None:
    minx, miny, _maxx, _maxy, width, height = bounds
    x = minx - width * 0.09
    y = miny - height * 0.095
    total_m = length_km * 1000
    half_m = total_m / 2
    bar_h = height * 0.007
    tick_h = height * 0.018
    for i, fill in enumerate((INK, BG)):
        ax.add_patch(Rectangle(
            (x + i * half_m, y),
            half_m,
            bar_h,
            facecolor=fill,
            edgecolor=INK,
            linewidth=1.15,
            zorder=20,
            clip_on=False,
        ))
    for label, offset in ((0, 0), (length_km // 2, half_m), (length_km, total_m)):
        xi = x + offset
        ax.plot([xi, xi], [y, y + tick_h], color=INK, lw=1.15, zorder=21, clip_on=False)
        ax.text(xi, y + tick_h + height * 0.006, f"{label}",
                ha="center", va="bottom", fontsize=9.5, color=INK, zorder=21)
    ax.text(x + total_m + width * 0.016, y + tick_h + height * 0.006, "km",
            ha="left", va="bottom", fontsize=9.5, color=INK, zorder=21)


def add_north(ax: plt.Axes, bounds: tuple[float, float, float, float, float, float]) -> None:
    _minx, _miny, maxx, maxy, width, height = bounds
    x = maxx + width * 0.12
    y = maxy - height * 0.06
    ax.add_patch(FancyArrowPatch((x, y - height * 0.11), (x, y),
                                 arrowstyle="-|>", mutation_scale=16, lw=1.8, color=INK, zorder=30))
    ax.text(x, y - height * 0.145, "N", ha="center", va="center", fontsize=11.2, weight="bold", color=INK, zorder=30)


def add_north_figure(fig: plt.Figure, x: float = 0.905, y: float = 0.902) -> None:
    fig.add_artist(FancyArrowPatch((x, y - 0.045), (x, y),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=16, lw=1.8, color=INK, zorder=35))
    fig.text(x, y - 0.065, "N", ha="center", va="center", fontsize=11.2,
             weight="bold", color=INK, zorder=35)


def data_to_fig(fig: plt.Figure, ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    return tuple(fig.transFigure.inverted().transform(ax.transData.transform((x, y))))


def external_callouts(
    fig: plt.Figure,
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    names: list[str],
    top_y: float = 0.618,
    bottom_y: float = 0.382,
    label_x: float = 0.828,
    end_x: float = 0.826,
    elbow_x: float = 0.765,
) -> None:
    rows = []
    for name in names:
        match = gdf[gdf["ward_name"].astype(str) == name]
        if match.empty:
            continue
        row = match.iloc[0]
        pt = row.geometry.representative_point()
        rows.append((name, row, pt))

    rows = sorted(rows, key=lambda r: r[2].y, reverse=True)
    if not rows:
        return
    slots = np.linspace(top_y, bottom_y, len(rows)) if len(rows) > 1 else np.array([(top_y + bottom_y) / 2])
    for slot_y, (name, row, pt) in zip(slots, rows):
        anchor_fig = data_to_fig(fig, ax, pt.x, pt.y)
        # A dark under-stroke keeps the bright leader visible on pale wards.
        for color, lw, alpha, zorder in (
            (LEADER_SHADOW, 2.85, 0.46, 17),
            (LEADER, 1.65, 0.94, 18),
        ):
            fig.add_artist(Line2D(
                [anchor_fig[0], elbow_x, end_x],
                [anchor_fig[1], slot_y, slot_y],
                transform=fig.transFigure,
                color=color,
                lw=lw,
                alpha=alpha,
                linestyle="-",
                solid_capstyle="round",
                zorder=zorder,
                clip_on=False,
            ))
        ax.scatter([pt.x], [pt.y], s=52, facecolor=ANCHOR_HALO, edgecolor=ANCHOR_OUTLINE, linewidth=1.15,
                   zorder=24, clip_on=False)
        ax.scatter([pt.x], [pt.y], s=24, facecolor=ANCHOR_FILL, edgecolor=ANCHOR_OUTLINE, linewidth=0.7,
                   zorder=25, clip_on=False)
        text = f"{name}\n#{int(row['ecological_stress_rank'])} ecological stress"
        fig.text(label_x, slot_y, text, ha="left", va="center", fontsize=9.6, color=INK,
                 linespacing=1.08, zorder=26,
                 bbox=dict(boxstyle="round,pad=0.24", fc=BG, ec=LABEL_EDGE, lw=0.72, alpha=0.98))


def color_luminance(value: float, colormap, norm=None) -> float:
    if pd.isna(value):
        return 0.74
    if norm is not None:
        rgba = colormap(norm(float(value)))
    elif isinstance(colormap, ListedColormap):
        rgba = colormap(int(np.clip(round(float(value)), 0, colormap.N - 1)))
    else:
        rgba = colormap(np.clip(float(value) / 8.0, 0, 1))
    r, g, b = rgba[:3]
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def line_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type in {"LineString", "LinearRing"}:
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            parts.extend(line_parts(part))
        return parts
    return []


def adaptive_ward_boundaries(ax: plt.Axes, gdf: gpd.GeoDataFrame, column: str, colormap, norm=None) -> None:
    g = gdf.reset_index(drop=True)
    luminance = [color_luminance(v, colormap, norm) for v in g[column]]
    spatial_index = g.sindex
    light_edge = "#f8f1e6"
    dark_edge = "#4f453d"
    segments = {light_edge: [], dark_edge: []}
    for i, geom in enumerate(g.geometry):
        for j in spatial_index.intersection(geom.bounds):
            if j <= i:
                continue
            other = g.geometry.iloc[j]
            if not geom.intersects(other):
                continue
            shared = geom.boundary.intersection(other.boundary)
            if shared.is_empty or shared.length <= 0.5:
                continue
            avg_luma = (luminance[i] + luminance[j]) / 2.0
            edge = light_edge if avg_luma < 0.50 else dark_edge
            for line in line_parts(shared):
                xs, ys = line.xy
                segments[edge].append(np.column_stack([xs, ys]))
    for edge, lines in segments.items():
        if not lines:
            continue
        halo = dark_edge if edge == light_edge else light_edge
        ax.add_collection(LineCollection(lines, colors=halo, linewidths=1.10, alpha=0.34,
                                         zorder=8, capstyle="round", joinstyle="round"))
        ax.add_collection(LineCollection(lines, colors=edge, linewidths=0.66, alpha=0.92,
                                         zorder=9, capstyle="round", joinstyle="round"))


def plot_boundaries(ax: plt.Axes, gdf: gpd.GeoDataFrame, column: str, colormap, norm=None) -> None:
    adaptive_ward_boundaries(ax, gdf, column, colormap, norm)
    gdf.dissolve().boundary.plot(ax=ax, color=OUTER_EDGE, linewidth=1.95, alpha=0.98, zorder=10)
    gdf.nsmallest(8, "ecological_stress_rank").boundary.plot(ax=ax, color=TOP_EDGE, linewidth=1.22, alpha=0.94, zorder=11)


def base_ward_plot(ax: plt.Axes, gdf: gpd.GeoDataFrame, column: str, colormap, norm=None) -> tuple:
    bounds = set_map_limits(ax, gdf)
    gdf.plot(ax=ax, column=column, cmap=colormap, norm=norm, vmin=None if norm else 0, vmax=None if norm else 8,
             linewidth=0, edgecolor="none", zorder=4)
    plot_boundaries(ax, gdf, column, colormap, norm)
    add_scalebar(ax, bounds)
    return bounds


def render_stress_map(gdf: gpd.GeoDataFrame) -> Path:
    pal = cmap(STRESS_PALETTE, "stress_index_redone")
    norm = Normalize(gdf["ecological_stress_index"].quantile(0.02), gdf["ecological_stress_index"].quantile(0.98))
    fig, ax = setup_figure(
        "Bengaluru Ecological Stresspeaks",
        "Heat gain, vegetation loss, water decline, and built-up expansion mapped together",
        "Ward-level composite stress index. Taller future 3D peaks use the same ecological stress score.",
    )
    bounds = base_ward_plot(ax, gdf, "ecological_stress_index", pal, norm)
    labels = gdf.nsmallest(5, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_callouts(fig, ax, gdf, labels)
    add_north_figure(fig)

    cax = fig.add_axes([0.28, 0.135, 0.48, 0.019])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=pal), cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor("#a89d91")
    cb.ax.tick_params(labelsize=10.2, colors=INK, length=0)
    cb.set_label("Ecological stress index (weighted z-score)", fontsize=10.2, color=INK)
    add_footer(fig, METHOD_STRESS)
    return save(fig, WARD_MAPS / f"{OUT_PREFIX}sample_ecological_stress_index_ward_map.png")


def tertile(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 3, labels=False).astype(int)


def add_bivariate_legend(fig: plt.Figure, x_label: str, y_label: str, palette: np.ndarray) -> None:
    leg = fig.add_axes([0.720, 0.128, 0.142, 0.142], facecolor=BG)
    leg.set_axis_off()
    for r in range(3):
        for c in range(3):
            leg.add_patch(Rectangle((c, r), 1, 1, facecolor=palette[r, c], edgecolor=BG, lw=1.2))
    leg.set_xlim(-0.58, 3.25)
    leg.set_ylim(-0.55, 3.35)
    leg.text(1.5, -0.36, f"{x_label} ->", ha="center", va="top", fontsize=8.9, color=INK)
    leg.text(-0.36, 1.5, f"{y_label} ->", ha="right", va="center", rotation=90, fontsize=8.9, color=INK)
    leg.text(0.5, 3.14, "low", ha="center", va="bottom", fontsize=7.8, color=MUTE)
    leg.text(2.5, 3.14, "high", ha="center", va="bottom", fontsize=7.8, color=MUTE)
    leg.text(1.5, 3.46, "Bivariate classes", ha="center", va="bottom", fontsize=9.0, weight="bold", color=INK)


def render_bivariate(gdf: gpd.GeoDataFrame, x_key: str, y_key: str, stem: str) -> Path:
    x_cfg, y_cfg = VARIABLES[x_key], VARIABLES[y_key]
    g = gdf.copy()
    x_class = tertile(g[x_cfg["column"]])
    y_class = tertile(g[y_cfg["column"]])
    g["bivar"] = y_class * 3 + x_class
    palette = BIVAR_PALETTES[stem]
    cmap_bi = ListedColormap(palette.reshape(-1))

    fig, ax = setup_figure(
        f"{x_cfg['title']} x {y_cfg['title']}",
        "Bivariate ecological stress pattern by BBMP ward",
        f"Shows where {x_cfg['short']} and {y_cfg['short']} overlap, using low/medium/high ward classes.",
    )
    base_ward_plot(ax, g, "bivar", cmap_bi, None)
    labels = g.nsmallest(4, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_callouts(fig, ax, g, labels)
    add_bivariate_legend(fig, x_cfg["title"], y_cfg["title"], palette)
    add_north_figure(fig)
    add_footer(fig, METHOD_BIVAR)
    return save(fig, WARD_MAPS / f"{OUT_PREFIX}{stem}.png")


def render_contour(gdf: gpd.GeoDataFrame) -> Path:
    path = STRESS / "ecological_stress_index_pixel_norm.tif"
    with rasterio.open(path) as ds:
        stress = ds.read(1).astype("float32")
        bounds_r = ds.bounds
    valid = np.isfinite(stress)
    smooth = gaussian_filter(np.where(valid, stress, 0), sigma=3.0)
    smooth = np.where(valid, smooth, np.nan)
    gy, gx = np.gradient(np.nan_to_num(smooth, nan=np.nanmean(smooth)))
    shade = 1 - np.clip(np.hypot(gx, gy) * 7.5, 0, 0.34)
    shade = np.where(valid, shade, np.nan)

    pal = cmap(CONTOUR_PALETTE, "contour_redone")
    gray = plt.get_cmap("gray").copy()
    gray.set_bad((0, 0, 0, 0))
    data = np.ma.masked_invalid(smooth)
    hill = np.ma.masked_invalid(shade)
    extent = [bounds_r.left, bounds_r.right, bounds_r.bottom, bounds_r.top]

    fig, ax = setup_figure(
        "Ecological Stress Contours",
        "DEM-style view of Bengaluru's combined stress surface",
        "Contours and shaded relief show where multiple urban-ecological pressures rise together.",
    )
    map_bounds = set_map_limits(ax, gdf)
    ax.imshow(data, extent=extent, origin="upper", cmap=pal, vmin=0, vmax=1, alpha=0.96, zorder=2)
    ax.imshow(hill, extent=extent, origin="upper", cmap=gray, alpha=0.14, vmin=0.60, vmax=1.0, zorder=3)
    ax.contour(data, levels=np.linspace(0.18, 0.92, 10), extent=extent, origin="upper",
               colors="#352620", linewidths=0.34, alpha=0.36, zorder=5)
    contour_norm = Normalize(gdf["ecological_stress_index"].quantile(0.02), gdf["ecological_stress_index"].quantile(0.98))
    adaptive_ward_boundaries(ax, gdf, "ecological_stress_index", pal, contour_norm)
    gdf.dissolve().boundary.plot(ax=ax, color=OUTER_EDGE, linewidth=1.95, alpha=0.96, zorder=9)
    gdf.nsmallest(6, "ecological_stress_rank").boundary.plot(ax=ax, color=TOP_EDGE, linewidth=1.18, alpha=0.9, zorder=10)
    labels = gdf.nsmallest(3, "ecological_stress_rank")["ward_name"].astype(str).tolist()
    external_callouts(fig, ax, gdf, labels)
    add_scalebar(ax, map_bounds)
    add_north_figure(fig)

    cax = fig.add_axes([0.735, 0.235, 0.18, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=pal), cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor("#a89d91")
    cb.ax.tick_params(labelsize=9.4, colors=INK, length=0)
    cb.set_label("Composite stress surface", fontsize=9.7, color=INK)
    add_footer(fig, METHOD_CONTOUR)
    return save(fig, STRESS / f"{OUT_PREFIX}sample_ecological_stress_contour_dem.png")


def main() -> None:
    WARD_MAPS.mkdir(parents=True, exist_ok=True)
    STRESS.mkdir(parents=True, exist_ok=True)
    gdf = read_data()
    outputs = [render_stress_map(gdf)]
    for x_key, y_key, stem in BIVAR_PAIRS:
        outputs.append(render_bivariate(gdf, x_key, y_key, stem))
    outputs.append(render_contour(gdf))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
