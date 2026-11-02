"""Put the held-out predictions on a map of New York City.

Inspections are pooled into H3 hexagons (resolution 8, about 0.7 km^2 each) so
the map shows, per few-block area, how often the model expected inspectors to
find rats versus how often they actually did.
"""

import json
from importlib import resources

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from shapely.geometry import mapping, shape

from ratcast import config, style

MIN_INSPECTIONS = 25  # hexes with fewer held-out inspections are too noisy to colour
PRIOR_FINDINGS = 5
LAT0 = 40.7128
ASPECT = 1 / np.cos(np.radians(LAT0))


def hex_summary(preds: pd.DataFrame, min_inspections: int = MIN_INSPECTIONS) -> pd.DataFrame:
    city_p, city_a = preds["p_rats"].mean(), preds["label"].mean()
    g = preds.groupby("h3_8", observed=True)
    out = g.agg(inspections=("label", "size"), found=("label", "sum"),
                actual=("label", "mean"), predicted=("p_rats", "mean"))
    out["nta"] = g["nta"].agg(lambda s: s.value_counts().index[0] if s.notna().any() else "")
    out["borough"] = g["borough"].agg(lambda s: s.value_counts().index[0] if s.notna().any() else "")
    out = out[out["inspections"] >= min_inspections].copy()
    # Each map against its own city-wide average, so the comparison is about
    # *where* rats turn up rather than the overall level (which can shift
    # city-wide in ways no forecast made months earlier could know).
    out["predicted_rel"] = out["predicted"] / city_p
    out["actual_rel"] = out["actual"] / city_a
    # > 1: more rats than expected here, once the city-wide gap is set aside.
    # Rat findings vs. the forecast's count scaled to the city-wide level, with
    # PRIOR_FINDINGS added to both so a thinly inspected hexagon can't look
    # like a big miss on a handful of visits.
    expected_found = out["predicted"] * (city_a / city_p) * out["inspections"]
    out["found_vs_expected"] = (out["found"] + PRIOR_FINDINGS) / (expected_found + PRIOR_FINDINGS)
    return out.reset_index()


def neighbourhood_summary(preds: pd.DataFrame, min_inspections: int = 200) -> pd.DataFrame:
    g = preds.groupby(["borough", "nta"], observed=True)
    out = g.agg(inspections=("label", "size"), actual=("label", "mean"), predicted=("p_rats", "mean"))
    return out[out["inspections"] >= min_inspections].sort_values("predicted", ascending=False).reset_index()


def _boroughs() -> list[dict]:
    gj = json.loads(config.BOROUGHS_GEOJSON.read_text())
    out = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"]).simplify(0.00025, preserve_topology=True)
        out.append({"name": feat["properties"]["boroname"], "geometry": geom})
    return out


def _hex_polys(cells) -> list[np.ndarray]:
    return [np.array([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]


def _draw_city(ax, boroughs, label: bool = True) -> None:
    for b in boroughs:
        geoms = getattr(b["geometry"], "geoms", [b["geometry"]])
        polys = [np.asarray(g.exterior.coords) for g in geoms]
        ax.add_collection(PolyCollection(polys, facecolor="#222221", edgecolor="#383835", linewidth=0.6, zorder=1))
    ax.set_facecolor(style.PAGE)
    ax.set_aspect(ASPECT)
    ax.set_xlim(-74.265, -73.69)
    ax.set_ylim(40.49, 40.92)
    ax.axis("off")
    if label:
        # Placed on open water / empty land so they never sit on a hexagon.
        spots = {"Manhattan": (-74.035, 40.80), "Bronx": (-73.795, 40.905), "Brooklyn": (-73.985, 40.553),
                 "Queens": (-73.745, 40.665), "Staten Island": (-74.225, 40.64)}
        for name, (x, y) in spots.items():
            ax.text(x, y, name, color=style.MUTED, fontsize=10, ha="center", va="center", zorder=5)


def _draw_hexes(ax, polys, values, cmap, norm) -> None:
    ax.add_collection(PolyCollection(polys, array=values, cmap=cmap, norm=norm,
                                     edgecolor=style.PAGE, linewidth=0.25, zorder=2))


def _risk_norm(hexes: pd.DataFrame) -> Normalize:
    top = np.quantile(hexes["actual_rel"], 0.95)
    return Normalize(vmin=0, vmax=np.ceil(top * 2) / 2)  # round up to the next 0.5x


TIMES = FuncFormatter(lambda v, _: f"{v:g}×")


def _colorbar(fig, mappable_norm, cax, label: str, cmap=style.RISK_CMAP, fmt=TIMES, ticks=None,
              fontsize: float = 11) -> None:
    sm = plt.cm.ScalarMappable(norm=mappable_norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.xaxis.set_major_formatter(fmt)
    cb.ax.tick_params(colors=style.INK_2, labelsize=fontsize, length=0)
    cb.ax.set_title(label, color=style.INK_2, fontsize=fontsize, pad=8)


def _side_by_side(hexes, boroughs, path, size, scale: float, headline: str, lines: list[str]) -> None:
    polys = _hex_polys(hexes["h3_8"])
    norm = _risk_norm(hexes)
    fig = plt.figure(figsize=size)
    top = 0.86 - 0.035 * (len(lines) - 1)
    ax1 = fig.add_axes([0.02, 0.12, 0.47, top - 0.14])
    ax2 = fig.add_axes([0.51, 0.12, 0.47, top - 0.14])
    for ax, col, title in ((ax1, "predicted_rel", "What the model expected"), (ax2, "actual_rel", "What inspectors found")):
        _draw_city(ax, boroughs, label=scale >= 1)
        _draw_hexes(ax, polys, hexes[col].to_numpy(), style.RISK_CMAP, norm)
        ax.set_title(title, color=style.INK, fontsize=22 * scale, weight="bold", loc="center", pad=6)
    fig.text(0.03, 0.95, headline, fontsize=30 * scale, weight="bold", color=style.INK)
    for i, line in enumerate(lines):
        fig.text(0.03, 0.905 - i * 0.035, line, fontsize=14 * scale, color=style.MUTED)
    _colorbar(fig, norm, fig.add_axes([0.35, 0.06, 0.30, 0.022]),
              "Rat-find rate compared with that map's city-wide average", fontsize=11 * max(scale, 0.8))
    if scale >= 1:
        fig.text(0.98, 0.02, "Data: NYC DOHMH Rodent Inspections (NYC Open Data)", ha="right",
                 color=style.MUTED, fontsize=10)
    style.save(fig, path, dpi=100, exact_frame=True)


def draw_side_by_side(hexes, boroughs, period: str, levels: tuple[float, float]) -> None:
    expected, found = levels
    _side_by_side(
        hexes, boroughs, config.FIGURE_DIR / "map_predicted_vs_actual.png", (19.2, 10.8), 1.0,
        "Where will NYC inspectors find rats?",
        [f"First-time rodent inspections, {period} · forecast made Jan 1, 2025, knowing nothing after that morning",
         f"Each map against its own city-wide average (1× = average). Overall, the model expected rats on "
         f"{expected:.0%} of first visits; inspectors found them on {found:.0%}."],
    )


def draw_social_card(hexes, boroughs, period: str) -> None:
    """1200x630 link-preview image for the published page."""
    _side_by_side(hexes, boroughs, config.FIGURE_DIR / "social_card.png", (12, 6.3), 0.62,
                  "Where will NYC inspectors find rats?",
                  [f"Forecast made Jan 1, 2025 vs. what inspectors found, {period}"])


def draw_vertical(hexes, boroughs, period: str) -> None:
    polys = _hex_polys(hexes["h3_8"])
    norm = _risk_norm(hexes)
    fig = plt.figure(figsize=(10.8, 19.2))
    ax1 = fig.add_axes([0.02, 0.50, 0.96, 0.36])
    ax2 = fig.add_axes([0.02, 0.10, 0.96, 0.36])
    for ax, col, title in ((ax1, "predicted_rel", "What the model expected"), (ax2, "actual_rel", "What inspectors found")):
        _draw_city(ax, boroughs, label=False)
        _draw_hexes(ax, polys, hexes[col].to_numpy(), style.RISK_CMAP, norm)
        ax.set_title(title, color=style.INK, fontsize=30, weight="bold", loc="center", pad=4)
    fig.text(0.5, 0.945, "Where will NYC inspectors\nfind rats?", fontsize=40, weight="bold",
             color=style.INK, ha="center", va="center", linespacing=1.1)
    fig.text(0.5, 0.895, f"Forecast made Jan 1, 2025 · first visits {period}", fontsize=18, color=style.MUTED, ha="center")
    _colorbar(fig, norm, fig.add_axes([0.2, 0.055, 0.6, 0.012]), "Compared with the city-wide average", fontsize=14)
    style.save(fig, config.FIGURE_DIR / "map_predicted_vs_actual_vertical.png", dpi=100, exact_frame=True)


def miss_summary(hexes: pd.DataFrame) -> dict:
    """The map gives every hexagon equal room, but inspections pile up in a
    few: say how the busiest third (most of all inspections) fared."""
    busy = hexes["inspections"] >= hexes["inspections"].quantile(2 / 3)
    return {
        "busy_share": float(hexes.loc[busy, "inspections"].sum() / hexes["inspections"].sum()),
        "busy_median": float(hexes.loc[busy, "found_vs_expected"].median()),
    }


def draw_difference(hexes, boroughs, period: str, levels: tuple[float, float]) -> None:
    expected, found = levels
    misses = miss_summary(hexes)
    polys = _hex_polys(hexes["h3_8"])
    norm = Normalize(vmin=-1, vmax=1)  # log2 scale: half as many ... twice as many
    fig = plt.figure(figsize=(12, 11.5))
    ax = fig.add_axes([0.02, 0.1, 0.96, 0.78])
    _draw_city(ax, boroughs)
    ratio = np.log2(hexes["found_vs_expected"].clip(0.5, 2.0)).to_numpy()
    _draw_hexes(ax, polys, ratio, style.DIVERGING_CMAP, norm)
    fig.text(0.04, 0.94, "Where the model missed", fontsize=26, weight="bold", color=style.INK)
    fig.text(0.04, 0.905, f"Rats found vs. expected in each hexagon, after setting aside the city-wide gap "
             f"({found:.0%} found vs. {expected:.0%} expected) · {period}", fontsize=12.5, color=style.MUTED)
    fig.text(0.04, 0.88, f"The busiest third of hexagons hold {misses['busy_share']:.0%} of first visits and came in at "
             f"{misses['busy_median']:.2f}× (median). Most red is quieter areas that didn't share in the drop.",
             fontsize=12.5, color=style.MUTED)
    fmt = FuncFormatter(lambda v, _: {-1: "half", 0: "as expected", 1: "double"}.get(int(round(v)), ""))
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    _colorbar(fig, norm, cax, "Rats found compared with what the model expected",
              cmap=style.DIVERGING_CMAP, fmt=fmt, ticks=[-1, 0, 1])
    cax.text(0, -2.6, "Fewer rats than expected", transform=cax.transAxes, ha="left", color=style.INK_2, fontsize=11)
    cax.text(1, -2.6, "More rats than expected", transform=cax.transAxes, ha="right", color=style.INK_2, fontsize=11)
    style.save(fig, config.FIGURE_DIR / "map_where_the_model_missed.png", dpi=100)


def write_interactive(hexes, neighbourhoods, boroughs, metrics: dict | None, period: str,
                      levels: tuple[float, float]) -> None:
    """Self-contained HTML page: no tiles, no server, opens straight from disk."""
    features = []
    for row, poly in zip(hexes.itertuples(index=False), _hex_polys(hexes["h3_8"])):
        features.append({
            "c": [[round(x, 5), round(y, 5)] for x, y in poly],
            "n": int(row.inspections),
            "a": round(float(row.actual), 4),
            "p": round(float(row.predicted), 4),
            "h": str(row.nta),
            "b": str(row.borough),
            "r": round(float(row.found_vs_expected), 4),
        })
    outlines = []
    for b in boroughs:
        geom = mapping(b["geometry"])
        rings = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        outlines.append({"name": b["name"], "rings": [[[round(x, 5), round(y, 5)] for x, y in poly[0]] for poly in rings]})
    payload = {
        "period": period,
        "hexes": features,
        "boroughs": outlines,
        "neighbourhoods": neighbourhoods.head(400).round(4).to_dict(orient="records"),
        "metrics": {k: metrics[k] for k in ("roc_auc", "base_rate", "mean_predicted", "test_inspections",
                                            "top_10pct_lift", "top_10pct_hit_rate", "map")} if metrics else None,
        "city": {"predicted": float(levels[0]), "actual": float(levels[1])},
        "misses": miss_summary(hexes),
        "scale_max": float(_risk_norm(hexes).vmax),
    }
    template = resources.files("ratcast").joinpath("templates/rat_map.html").read_text()
    html = (template.replace("/*__RATCAST_DATA__*/null", json.dumps(payload, separators=(",", ":")))
            .replace("__SITE_URL__", config.SITE_URL))
    out = config.MAP_DIR / "rat_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"  wrote {out.relative_to(config.PROJECT_ROOT)} ({len(html) / 1e6:.1f} MB)")


def make_maps(preds: pd.DataFrame | None = None) -> None:
    preds = pd.read_parquet(config.PREDICTIONS_PARQUET) if preds is None else preds
    preds["date"] = pd.to_datetime(preds["date"])
    period = f"{preds['date'].min():%b %Y} – {preds['date'].max():%b %Y}"
    hexes = hex_summary(preds)
    hoods = neighbourhood_summary(preds)
    boroughs = _boroughs()

    hexes.to_csv(config.OUTPUT_DIR / "hex_predicted_vs_actual.csv", index=False)
    hoods.to_csv(config.OUTPUT_DIR / "neighbourhood_predicted_vs_actual.csv", index=False)
    corr = np.corrcoef(hexes["predicted"], hexes["actual"])[0, 1]
    levels = (preds["p_rats"].mean(), preds["label"].mean())
    print(f"{len(hexes):,} hexagons with >= {MIN_INSPECTIONS} held-out inspections; "
          f"predicted vs actual correlation r = {corr:.2f}; city-wide expected {levels[0]:.1%}, found {levels[1]:.1%}")

    metrics_path = config.OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    style.apply()
    draw_side_by_side(hexes, boroughs, period, levels)
    draw_social_card(hexes, boroughs, period)
    draw_vertical(hexes, boroughs, period)
    draw_difference(hexes, boroughs, period, levels)
    write_interactive(hexes, hoods, boroughs, metrics, period, levels)
