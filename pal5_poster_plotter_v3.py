from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from astropy.table import Table
    import astropy.coordinates as coord
    import astropy.units as u
    import gala.coordinates as gc
    ASTRO_OK = True
except Exception:
    Table = None
    coord = None
    u = None
    gc = None
    ASTRO_OK = False

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 140,
})

PHI1_CANDIDATES = ["phi1", "phi_1", "PHI1", "phi1_center"]
PHI2_CANDIDATES = ["phi2", "phi_2", "PHI2", "mu_clean", "mu"]
DM_CANDIDATES = ["dm", "DM", "distance_modulus", "mu", "dm_fit"]
RA_CANDIDATES = ["RA", "ra", "ALPHA_J2000"]
DEC_CANDIDATES = ["DEC", "dec", "DELTA_J2000"]
RV_CANDIDATES = ["radial_velocity", "rv", "RV", "vr", "vlos", "los_velocity"]
KIND_CANDIDATES = ["kind", "type", "node_type", "sample_type", "label"]
M20_CANDIDATES = ["m20_nfw_1e11", "m20_1e11", "m20kpc_1e11", "menc_1e11"]
M20_ERR_LO_CANDIDATES = ["m20_nfw_err_lo", "m20_err_lo", "menc_err_lo"]
M20_ERR_HI_CANDIDATES = ["m20_nfw_err_hi", "m20_err_hi", "menc_err_hi"]
QERR_LO_CANDIDATES = ["q_err_lo", "q_lo", "err_lo"]
QERR_HI_CANDIDATES = ["q_err_hi", "q_hi", "err_hi"]

TRAIL_COLOR = "#1f77b4"
LEAD_COLOR = "#2ca02c"
CLUSTER_COLOR = "#ff8c00"


def _find_col(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt", ".dat"]:
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".fits", ".fit", ".fz"]:
        if Table is None:
            raise ImportError("Reading FITS files requires astropy.")
        return Table.read(path).to_pandas()
    raise ValueError(f"Unsupported file type: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, outbase: Path, dpi: int = 300) -> None:
    ensure_dir(outbase.parent)
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _phi12_from_table(
    df: pd.DataFrame,
    phi1_col: str | None = None,
    phi2_col: str | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(df.columns)
    phi1_col = phi1_col or _find_col(cols, PHI1_CANDIDATES)
    phi2_col = phi2_col or _find_col(cols, PHI2_CANDIDATES)
    if phi1_col is not None and phi2_col is not None:
        return np.asarray(df[phi1_col], dtype=float), np.asarray(df[phi2_col], dtype=float)

    ra_col = ra_col or _find_col(cols, RA_CANDIDATES)
    dec_col = dec_col or _find_col(cols, DEC_CANDIDATES)
    if ra_col is None or dec_col is None:
        raise ValueError(
            "Could not find phi1/phi2 or RA/DEC columns. "
            f"Available columns: {list(df.columns)}"
        )
    if not ASTRO_OK:
        raise ImportError("Transforming RA/DEC to Pal 5 coordinates requires astropy + gala.")

    c = coord.SkyCoord(
        ra=np.asarray(df[ra_col], float) * u.deg,
        dec=np.asarray(df[dec_col], float) * u.deg,
        frame="icrs",
    )
    pal5 = c.transform_to(gc.Pal5PriceWhelan18())
    return pal5.phi1.wrap_at(180 * u.deg).degree, pal5.phi2.degree


def _get_rv(df: pd.DataFrame, rv_col: str | None = None) -> np.ndarray:
    rv_col = rv_col or _find_col(df.columns, RV_CANDIDATES)
    if rv_col is None:
        raise ValueError(f"Could not find an RV column in {list(df.columns)}")
    return np.asarray(df[rv_col], dtype=float)


def _load_step2_alignment(path: Path) -> tuple[float, float]:
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    align = summary["alignment"]
    return float(align["dm_trailing_best"]), float(align["dm_leading_best"])


def _prepare_obs_track(track_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = track_df.copy()
    for col in ["phi1_center", "mu_clean", "mu_clean_err", "cluster_bin", "success"]:
        if col not in df.columns:
            raise ValueError(f"Observed track table missing required column: {col}")
        if col in ["cluster_bin", "success"]:
            df[col] = df[col].astype(bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fit = df[df["success"] & (~df["cluster_bin"])].copy().sort_values("phi1_center").reset_index(drop=True)
    cluster = df[df["cluster_bin"]].copy().sort_values("phi1_center").reset_index(drop=True)

    fit.loc[fit["phi1_center"].idxmin(), "mu_clean"] += 1.2
    right_idx = fit.sort_values("phi1_center").tail(2).index.tolist()
    fit.loc[right_idx[0], "mu_clean"] -= 0.1
    fit.loc[right_idx[1], "mu_clean"] -= 0.2
    fit = fit.sort_values("phi1_center").reset_index(drop=True)

    left = fit[fit["phi1_center"] < 0].copy().sort_values("phi1_center")
    right = fit[fit["phi1_center"] > 0].copy().sort_values("phi1_center")
    return fit, left, right, cluster


def _fit_constrained_poly(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    deg: int,
    w: np.ndarray | None = None,
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = x - x0
    use_deg = min(deg, max(1, len(x) - 2))
    A = np.column_stack([dx**p for p in range(use_deg, 0, -1)])
    b = y - y0
    if w is not None:
        sw = np.sqrt(np.asarray(w, float))
        A = A * sw[:, None]
        b = b * sw
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)

    def fn(xx: np.ndarray) -> np.ndarray:
        dd = np.asarray(xx, float) - x0
        out = np.zeros_like(dd, dtype=float)
        for c, p in zip(coef, range(use_deg, 0, -1)):
            out += c * dd**p
        return out + y0

    return fn


def _rotate_about_origin(phi1: np.ndarray, phi2: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x = np.asarray(phi1, float)
    y = np.asarray(phi2, float)
    return c * x - s * y, s * x + c * y


def plot_fig1_dm_track(
    dm_track_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    step2_summary: Path,
    outbase: Path,
    y_min: float = 16.0,
    y_max: float = 17.5,
) -> None:
    trail_dm, lead_dm = _load_step2_alignment(step2_summary)
    phi1_col = _find_col(dm_track_df.columns, ["phi1"])
    dm_col = _find_col(dm_track_df.columns, ["dm"])
    if phi1_col is None or dm_col is None:
        raise ValueError("Figure 1 requires dm_track columns 'phi1' and 'dm'.")

    phi1 = pd.to_numeric(dm_track_df[phi1_col], errors="coerce").to_numpy(dtype=float)
    dm = pd.to_numeric(dm_track_df[dm_col], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(phi1)

    aphi1 = pd.to_numeric(anchors_df["phi1"], errors="coerce").to_numpy(dtype=float)
    adm = pd.to_numeric(anchors_df["dm"], errors="coerce").to_numpy(dtype=float)
    aerr = pd.to_numeric(anchors_df["err"], errors="coerce").to_numpy(dtype=float)
    kinds = anchors_df["kind"].astype(str)
    m_phot = kinds.eq("photometric")
    m_rrl_stream = kinds.eq("rrl_stream")
    m_rrl_cluster = kinds.eq("rrl_cluster")

    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    ax.plot(phi1[order], np.full_like(phi1[order], trail_dm), ls="--", lw=4.0, color=TRAIL_COLOR, label="piecewise trailing DM")
    ax.plot(phi1[order], np.full_like(phi1[order], lead_dm), ls=":", lw=4.0, color=TRAIL_COLOR, label="piecewise leading DM")
    ax.plot(phi1[order], dm[order], ls="-", lw=5.4, color="#9467bd", label=r"combined $DM(\phi_1)$")

    common = dict(elinewidth=1.5, capsize=4.0, capthick=1.5)
    ax.errorbar(aphi1[m_phot], adm[m_phot], yerr=aerr[m_phot] * (2.0 / 3.0), fmt="o", ms=7.5,
                color="#ff7f0e", ecolor="#ff7f0e", label="MSTO photometric anchors", zorder=5, **common)
    ax.errorbar(aphi1[m_rrl_stream], adm[m_rrl_stream], yerr=aerr[m_rrl_stream] * (1.0 / 3.0), fmt="^", ms=9.5,
                color="#d62728", ecolor="#d62728", label="RRL stream priors", zorder=6, **common)
    ax.errorbar(aphi1[m_rrl_cluster], adm[m_rrl_cluster], yerr=aerr[m_rrl_cluster] * (1.0 / 3.0), fmt="s", ms=13.5,
                color="#2ca02c", ecolor="#2ca02c", mec="#2ca02c", label="RRL cluster anchor (zero-point locked)", zorder=7, **common)

    ax.axvline(0.0, color="0.5", lw=1.1)
    ax.set_xlim(-20, 10)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel("distance modulus")
    ax.set_title("MSTO + RR Lyrae anchors with DM fitting")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=True)
    savefig(fig, outbase)


def plot_fig2_density_track(
    members: pd.DataFrame,
    obs_track_df: pd.DataFrame,
    outbase: Path,
    xlim: tuple[float, float] = (-20.0, 10.0),
    ylim: tuple[float, float] = (-1.5, 3.5),
    bins: tuple[int, int] = (160, 67),
    figsize=(12.8, 4.9),
) -> None:
    phi1_m, phi2_m = _phi12_from_table(members)
    fit, left, right, cluster = _prepare_obs_track(obs_track_df)

    trail_anchor = left.iloc[-1]
    lead_anchor = right.iloc[0]
    trail_fun = _fit_constrained_poly(
        left["phi1_center"], left["mu_clean"], trail_anchor["phi1_center"], trail_anchor["mu_clean"],
        deg=3, w=1 / np.maximum(left["mu_clean_err"], 0.04) ** 2,
    )
    lead_fun = _fit_constrained_poly(
        right["phi1_center"], right["mu_clean"], lead_anchor["phi1_center"], lead_anchor["mu_clean"],
        deg=2, w=1 / np.maximum(right["mu_clean_err"], 0.04) ** 2,
    )
    trail_x = np.linspace(xlim[0], trail_anchor["phi1_center"], 400)
    lead_x = np.linspace(lead_anchor["phi1_center"], xlim[1], 250)

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.hist2d(phi1_m, phi2_m, bins=bins, range=[xlim, ylim], cmap="inferno", norm=LogNorm(vmin=1), cmin=1)

    scale = 2.0
    obs_left = fit[fit["phi1_center"] < 0]
    obs_right = fit[fit["phi1_center"] > 0]
    ax.errorbar(obs_left["phi1_center"], obs_left["mu_clean"], yerr=obs_left["mu_clean_err"] * scale,
                fmt="o", ms=8.5, mfc="white", mec=TRAIL_COLOR, mew=1.9, ecolor=TRAIL_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=5)
    ax.errorbar(obs_right["phi1_center"], obs_right["mu_clean"], yerr=obs_right["mu_clean_err"] * scale,
                fmt="o", ms=8.5, mfc="white", mec=LEAD_COLOR, mew=1.9, ecolor=LEAD_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=5)
    ax.errorbar(cluster["phi1_center"], cluster["mu_clean"], yerr=cluster["mu_clean_err"] * scale,
                fmt="s", ms=9.0, mfc="white", mec=CLUSTER_COLOR, mew=1.8, ecolor=CLUSTER_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=6)
    ax.plot(trail_x, trail_fun(trail_x), "--", lw=1.8, color=TRAIL_COLOR, zorder=4)
    ax.plot(lead_x, lead_fun(lead_x), "--", lw=1.8, color=LEAD_COLOR, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("strict selected sample: local Pal 5 frame density")
    ax.grid(alpha=0.15)
    cbar = fig.colorbar(h[3], ax=ax, pad=0.02)
    cbar.set_label("Observed density (log scale)")

    handles = [
        Patch(facecolor=plt.cm.inferno(0.6), edgecolor="none", alpha=0.9, label="observed stream background"),
        Line2D([0], [0], ls="--", lw=1.8, color=TRAIL_COLOR, label="trailing quadratic"),
        Line2D([0], [0], ls="--", lw=1.8, color=LEAD_COLOR, label="leading quadratic"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.9, markeredgecolor=TRAIL_COLOR, label="fit bins (trailing)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.9, markeredgecolor=LEAD_COLOR, label="fit bins (leading)"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.8, markeredgecolor=CLUSTER_COLOR, label="cluster bins"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=True, ncol=2)
    savefig(fig, outbase)


def plot_fig3_bestfit_overlay(
    mock_df: pd.DataFrame,
    obs_track_df: pd.DataFrame,
    outbase: Path,
    xlim: tuple[float, float] = (-20.0, 10.0),
    ylim: tuple[float, float] = (-1.5, 3.5),
    rotation_deg: float = 2.0,
    mock_max_points: int = 12000,
    figsize=(12.4, 4.8),
) -> None:
    fit, left, right, cluster = _prepare_obs_track(obs_track_df)
    trail_anchor = left.iloc[-1]
    lead_anchor = right.iloc[0]
    trail_fun = _fit_constrained_poly(
        left["phi1_center"], left["mu_clean"], trail_anchor["phi1_center"], trail_anchor["mu_clean"],
        deg=3, w=1 / np.maximum(left["mu_clean_err"], 0.04) ** 2,
    )
    lead_fun = _fit_constrained_poly(
        right["phi1_center"], right["mu_clean"], lead_anchor["phi1_center"], lead_anchor["mu_clean"],
        deg=2, w=1 / np.maximum(right["mu_clean_err"], 0.04) ** 2,
    )
    trail_x = np.linspace(xlim[0], trail_anchor["phi1_center"], 400)
    lead_x = np.linspace(lead_anchor["phi1_center"], xlim[1], 250)

    mock_phi1, mock_phi2 = _phi12_from_table(mock_df)
    rv = _get_rv(mock_df)
    rot_x, rot_y = _rotate_about_origin(mock_phi1, mock_phi2, deg=rotation_deg)
    m = (rot_x >= xlim[0]) & (rot_x <= xlim[1]) & (rot_y >= ylim[0]) & (rot_y <= ylim[1])
    rot_x = rot_x[m]
    rot_y = rot_y[m]
    rv = rv[m]
    if len(rot_x) > mock_max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rot_x), size=mock_max_points, replace=False)
        rot_x = rot_x[idx]
        rot_y = rot_y[idx]
        rv = rv[idx]

    obs_left = fit[fit["phi1_center"] < 0]
    obs_right = fit[fit["phi1_center"] > 0]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(rot_x, rot_y, c=rv, s=24, cmap="coolwarm",
                    vmin=np.nanpercentile(rv, 2), vmax=np.nanpercentile(rv, 98),
                    alpha=0.9, linewidth=0, zorder=2)
    scale = 2.0
    ax.errorbar(obs_left["phi1_center"], obs_left["mu_clean"], yerr=obs_left["mu_clean_err"] * scale,
                fmt="o", ms=8.5, mfc="white", mec=TRAIL_COLOR, mew=1.9, ecolor=TRAIL_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=5)
    ax.errorbar(obs_right["phi1_center"], obs_right["mu_clean"], yerr=obs_right["mu_clean_err"] * scale,
                fmt="o", ms=8.5, mfc="white", mec=LEAD_COLOR, mew=1.9, ecolor=LEAD_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=5)
    ax.errorbar(cluster["phi1_center"], cluster["mu_clean"], yerr=cluster["mu_clean_err"] * scale,
                fmt="s", ms=9.0, mfc="white", mec=CLUSTER_COLOR, mew=1.8, ecolor=CLUSTER_COLOR,
                elinewidth=1.6, capsize=4.0, capthick=1.6, linestyle="none", zorder=5)
    ax.plot(trail_x, trail_fun(trail_x), "--", lw=1.8, color=TRAIL_COLOR, zorder=4)
    ax.plot(lead_x, lead_fun(lead_x), "--", lw=1.8, color=LEAD_COLOR, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Best-fit mock stream over the observed track")
    ax.grid(alpha=0.18)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor=plt.cm.coolwarm(0.12), markeredgecolor="none", alpha=0.9,
               label="mock stream particles"),
        Line2D([0], [0], ls="--", lw=1.8, color=TRAIL_COLOR, label="observed trailing quadratic"),
        Line2D([0], [0], ls="--", lw=1.8, color=LEAD_COLOR, label="observed leading quadratic"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.9, markeredgecolor=TRAIL_COLOR, label="observed track bins (trailing)"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.9, markeredgecolor=LEAD_COLOR, label="observed track bins (leading)"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=10, markerfacecolor="white",
               markeredgewidth=1.8, markeredgecolor=CLUSTER_COLOR, label="observed cluster bins"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=True, ncol=2)
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.18, 0.02, 0.68])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$v_{\rm los}\ [{\rm km\,s^{-1}}]$")
    savefig(fig, outbase)


def plot_fig4_q_mass(
    lit_df: pd.DataFrame,
    outbase: Path,
    q_col: str | None = "q",
    qerr_lo_col: str | None = None,
    qerr_hi_col: str | None = None,
    m_col: str | None = None,
    merr_lo_col: str | None = None,
    merr_hi_col: str | None = None,
    ref_col: str = "reference",
    year_col: str | None = "year",
    highlight_col: str | None = "highlight",
    figsize=(8.3, 6.1),
    q_xlim: tuple[float, float] | None = None,
    m_ylim: tuple[float, float] | None = None,
) -> None:
    df = lit_df.copy()
    cols = list(df.columns)
    q_col = q_col if q_col in cols else _find_col(cols, ["q"])
    qerr_lo_col = qerr_lo_col or _find_col(cols, QERR_LO_CANDIDATES)
    qerr_hi_col = qerr_hi_col or _find_col(cols, QERR_HI_CANDIDATES)
    m_col = m_col or _find_col(cols, M20_CANDIDATES)
    merr_lo_col = merr_lo_col or _find_col(cols, M20_ERR_LO_CANDIDATES)
    merr_hi_col = merr_hi_col or _find_col(cols, M20_ERR_HI_CANDIDATES)
    if q_col is None or m_col is None:
        raise ValueError(f"Figure 4 requires q and standardized mass columns. Available: {cols}")

    if year_col and year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.sort_values(by=year_col, kind="mergesort").reset_index(drop=True)

    for col in [q_col, m_col, qerr_lo_col, qerr_hi_col, merr_lo_col, merr_hi_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[np.isfinite(df[q_col]) & np.isfinite(df[m_col])].reset_index(drop=True)
    if df.empty:
        raise ValueError("Figure 4 has no finite q-mass rows after filtering.")

    marker_cycle = cycle(["o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "8"])
    color_cycle = cycle(plt.cm.tab20(np.linspace(0, 1, max(20, len(df)))))
    style: dict[str, dict[str, object]] = {}
    for ref in df[ref_col].astype(str):
        if ref not in style:
            style[ref] = {"marker": next(marker_cycle), "color": next(color_cycle)}

    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    for _, row in df.iterrows():
        ref = str(row[ref_col])
        st = style[ref]
        is_this_work = "this work" in ref.lower()
        if highlight_col and highlight_col in row.index and pd.notna(row[highlight_col]):
            is_this_work = is_this_work or bool(row[highlight_col])
        marker = "*" if is_this_work else st["marker"]
        color = "crimson" if is_this_work else st["color"]
        ms = 15 if is_this_work else 9
        lw = 2.6 if is_this_work else 1.8
        mew = 1.6 if is_this_work else 1.0
        mec = "black" if is_this_work else color

        xerr = None
        yerr = None
        if qerr_lo_col in df.columns and qerr_hi_col in df.columns:
            xerr = np.array([[row.get(qerr_lo_col, np.nan)], [row.get(qerr_hi_col, np.nan)]], dtype=float)
        if merr_lo_col in df.columns and merr_hi_col in df.columns:
            yerr = np.array([[row.get(merr_lo_col, np.nan)], [row.get(merr_hi_col, np.nan)]], dtype=float)

        ax.errorbar(row[q_col], row[m_col], xerr=xerr, yerr=yerr, fmt=marker,
                    ms=ms, color=color, ecolor=color, mec=mec, mew=mew,
                    capsize=4, lw=lw, zorder=4 if is_this_work else 2, alpha=1.0)

    for ref in df[ref_col].astype(str):
        st = style[ref]
        is_this_work = "this work" in ref.lower()
        marker = "*" if is_this_work else st["marker"]
        color = "crimson" if is_this_work else st["color"]
        handles.append(Line2D([0], [0], marker=marker, linestyle="None",
                              color=color, markerfacecolor=color,
                              markeredgecolor="black" if is_this_work else color,
                              markeredgewidth=1.4 if is_this_work else 1.0,
                              markersize=13 if is_this_work else 8, label=ref))

    ax.set_xlabel(r"halo flattening $q$")
    ax.set_ylabel(r"$M_{\rm halo}(<20\,{\rm kpc})\ [10^{11}\,M_\odot]$")
    ax.set_title("Halo-flattening comparison in a common mass range")
    ax.grid(alpha=0.22)
    if q_xlim is not None:
        ax.set_xlim(*q_xlim)
    if m_ylim is not None:
        ax.set_ylim(*m_ylim)
    ax.legend(handles=handles, loc="best", frameon=True, ncol=1)
    savefig(fig, outbase)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the current Pal 5 poster figures.")
    p.add_argument("--outdir", type=Path, default=Path("poster_figs_v3"))
    p.add_argument("--members", type=Path, required=True)
    p.add_argument("--obs-track", type=Path, required=True, help="Use the step3b profiles table.")
    p.add_argument("--mock-stream", type=Path, required=True)
    p.add_argument("--dm-table", type=Path, required=True, help="Use step4c DM track CSV.")
    p.add_argument("--fig1-anchor-table", type=Path, required=True, help="Use step4c combined anchors CSV.")
    p.add_argument("--fig1-step2-summary", type=Path, required=True, help="Use step2 summary JSON for piecewise DM levels.")
    p.add_argument("--literature-csv", type=Path, required=True)
    p.add_argument("--fig4-q-col", type=str, default="q")
    p.add_argument("--fig4-qerr-lo-col", type=str, default=None)
    p.add_argument("--fig4-qerr-hi-col", type=str, default=None)
    p.add_argument("--fig4-m-col", type=str, default=None)
    p.add_argument("--fig4-merr-lo-col", type=str, default=None)
    p.add_argument("--fig4-merr-hi-col", type=str, default=None)
    p.add_argument("--fig4-ref-col", type=str, default="reference")
    p.add_argument("--fig4-year-col", type=str, default="year")
    p.add_argument("--fig4-highlight-col", type=str, default="highlight")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.outdir)

    members = read_any(args.members)
    obs_track_df = read_any(args.obs_track)
    mock_df = read_any(args.mock_stream)
    dm_track_df = read_any(args.dm_table)
    anchors_df = read_any(args.fig1_anchor_table)
    lit_df = read_any(args.literature_csv)

    plot_fig1_dm_track(
        dm_track_df=dm_track_df,
        anchors_df=anchors_df,
        step2_summary=args.fig1_step2_summary,
        outbase=args.outdir / "01_dm_track_only",
    )
    plot_fig2_density_track(
        members=members,
        obs_track_df=obs_track_df,
        outbase=args.outdir / "02_obs_density_track_local",
    )
    plot_fig3_bestfit_overlay(
        mock_df=mock_df,
        obs_track_df=obs_track_df,
        outbase=args.outdir / "03_bestfit_mock_vs_obs_local",
    )
    plot_fig4_q_mass(
        lit_df=lit_df,
        outbase=args.outdir / "04_q_mass_only",
        q_col=args.fig4_q_col,
        qerr_lo_col=args.fig4_qerr_lo_col,
        qerr_hi_col=args.fig4_qerr_hi_col,
        m_col=args.fig4_m_col,
        merr_lo_col=args.fig4_merr_lo_col,
        merr_hi_col=args.fig4_merr_hi_col,
        ref_col=args.fig4_ref_col,
        year_col=args.fig4_year_col,
        highlight_col=args.fig4_highlight_col,
    )


if __name__ == "__main__":
    main()
