#!/usr/bin/env python3
"""
Pal 5 step 2: Bonaca-style baseline member selection.

This script is intentionally conservative and follows the Bonaca+2020 logic:
  1) start from the preprocessed clean stellar catalog,
  2) apply the z-locus star/galaxy cut,
  3) apply a hard CMD isochrone cut,
  4) output a strict member catalog for downstream 1D morphology modeling.

Compared with the paper, this script adds one pragmatic nuisance-alignment step:
  - fit small global CMD warp terms on the cluster center CMD:
      dmu  : magnitude / distance-modulus offset
      dc0  : global color offset
      dc1  : linear color tilt as a function of magnitude

These nuisance terms are *not* interpreted physically. They exist only to absorb
small filter-system / zeropoint / residual-calibration mismatches between the
input isochrone file and the observed DECam photometry.

The script does NOT implement the final Bonaca distance-gradient interpolation in
2-degree phi1 bins. Instead it provides a faithful baseline first-pass selection
using a two-arm distance model:
    phi1 <= 0  -> trailing-arm distance
    phi1 >  0  -> leading-arm distance
This is the right place to start before building a smoother DM(phi1) model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RA_MIN, RA_MAX = 210.0, 260.0
DEC_MIN, DEC_MAX = -20.0, 20.0
PHI1_MIN, PHI1_MAX = -25.0, 20.0
PHI2_MIN, PHI2_MAX = -5.0, 10.0

# Cluster center in the Pal 5 aligned frame is (0, 0) by construction.
R_CLUSTER_IN = 0.15   # deg
R_CLUSTER_BG_IN = 0.25
R_CLUSTER_BG_OUT = 0.45

# Bonaca-style strict sample.
STRICT_GMIN = 20.0
STRICT_GMAX = 23.0
STRICT_GR_MIN = -0.35
STRICT_GR_MAX = 1.25

# z-locus from Bonaca+2020.
ZLOCUS_SLOPE = 1.7
ZLOCUS_INTERCEPT = -0.17
ZLOCUS_TOL = 0.10
ZLOCUS_GR_MAX = 1.20

# Initial cluster distance modulus guess.
DM_CLUSTER0 = 16.835
# Bonaca first-pass arm distances: 23 kpc (trailing, phi1<=0) and 19 kpc (leading, phi1>0).
DM_TRAILING_REL = 5.0 * np.log10(23_000.0 / 10.0) - DM_CLUSTER0
DM_LEADING_REL = 5.0 * np.log10(19_000.0 / 10.0) - DM_CLUSTER0

# CMD selection-box width around the ridge in (g-r, g).
CMD_W0 = 0.06          # at g=20
CMD_W_SLOPE = 0.018    # mag in color per mag in g
CMD_W_MIN = 0.05
CMD_W_MAX = 0.14
CMD_W_REF = 20.0

# Fitting ranges for the nuisance CMD warp.
FIT_GMIN = 19.2
FIT_GMAX = 23.2
FIT_DMU_GRID = np.linspace(-0.35, 0.35, 36)
FIT_DC0_GRID = np.linspace(-0.14, 0.14, 29)
FIT_DC1_GRID = np.linspace(-0.05, 0.05, 21)
FIT_SIGMA0 = 0.05
FIT_SIGMA_SLOPE = 0.015
FIT_SIGMA_MIN = 0.04
FIT_SIGMA_MAX = 0.12

# Plots / sampling.
QC_SCATTER_MAX = 120_000
RNG_SEED = 12345


@dataclass
class AlignmentResult:
    dmu: float
    dc0: float
    dc1: float
    score: float
    dm_cluster_best: float
    dm_trailing_best: float
    dm_leading_best: float


# -----------------------------------------------------------------------------
# Isochrone handling
# -----------------------------------------------------------------------------
def read_parsec_like_isochrone(path: str) -> Dict[str, np.ndarray]:
    """Read a PARSEC/MIST-like ASCII isochrone file.

    Expected header contains a comment line beginning with '# Zini ...'.
    Required columns: label, g_f0, r_f0, z_f0.
    """
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Zini"):
                header = line.lstrip("#").strip()
                break
    if header is None:
        raise RuntimeError("Could not find the '# Zini ...' header line in the isochrone file.")

    colnames = header.split()
    name_to_idx = {name: i for i, name in enumerate(colnames)}
    needed = ["label", "g_f0", "r_f0", "z_f0"]
    missing = [name for name in needed if name not in name_to_idx]
    if missing:
        raise KeyError(f"Missing required isochrone columns: {missing}")

    arr = np.loadtxt(path, comments="#")
    label = arr[:, name_to_idx["label"]].astype(int)

    # Keep main sequence + SGB + RGB phases from the user-provided file,
    # but the final strict selection later only uses the relevant magnitude range.
    keep = np.isin(label, [1, 2, 3])
    arr = arr[keep]

    g_abs = arr[:, name_to_idx["g_f0"]].astype(float)
    r_abs = arr[:, name_to_idx["r_f0"]].astype(float)
    z_abs = arr[:, name_to_idx["z_f0"]].astype(float)

    return {
        "g_abs": g_abs,
        "r_abs": r_abs,
        "z_abs": z_abs,
        "gr_abs": g_abs - r_abs,
        "gz_abs": g_abs - z_abs,
        "rz_abs": r_abs - z_abs,
    }


class RidgeModel:
    def __init__(self, g_model: np.ndarray, gr_model: np.ndarray):
        g = np.asarray(g_model, dtype=float)
        c = np.asarray(gr_model, dtype=float)
        ok = np.isfinite(g) & np.isfinite(c)
        g = g[ok]
        c = c[ok]
        if g.size < 5:
            raise ValueError("Too few valid isochrone points after filtering.")

        # Sort by apparent g and collapse repeated / nearly repeated g values.
        order = np.argsort(g)
        g = g[order]
        c = c[order]

        g_unique: List[float] = []
        c_unique: List[float] = []
        i = 0
        while i < len(g):
            j = i + 1
            while j < len(g) and abs(g[j] - g[i]) < 1e-4:
                j += 1
            g_unique.append(float(np.mean(g[i:j])))
            c_unique.append(float(np.mean(c[i:j])))
            i = j

        self.g = np.asarray(g_unique, dtype=float)
        self.c = np.asarray(c_unique, dtype=float)
        self.gmin = float(np.min(self.g))
        self.gmax = float(np.max(self.g))

    def color_at(self, g_obs: np.ndarray) -> np.ndarray:
        g_obs = np.asarray(g_obs, dtype=float)
        out = np.full(g_obs.shape, np.nan, dtype=float)
        ok = (g_obs >= self.gmin) & (g_obs <= self.gmax) & np.isfinite(g_obs)
        if np.any(ok):
            out[ok] = np.interp(g_obs[ok], self.g, self.c)
        return out


def build_gr_ridge(
    iso: Dict[str, np.ndarray],
    dm: float,
    dmu: float,
    dc0: float,
    dc1: float,
    g_ref: float = 21.0,
) -> RidgeModel:
    g_app = iso["g_abs"] + dm + dmu
    gr = iso["gr_abs"] + dc0 + dc1 * (g_app - g_ref)

    # The strict CMD selection is only used on the main-sequence region.
    keep = np.isfinite(g_app) & np.isfinite(gr) & (g_app >= 17.0) & (g_app <= 25.5)
    if np.sum(keep) < 5:
        raise ValueError("Isochrone transform left too few usable points.")

    return RidgeModel(g_app[keep], gr[keep])


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def sigma_fit_g(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    s = FIT_SIGMA0 + FIT_SIGMA_SLOPE * np.clip(g - 20.0, 0.0, None)
    return np.clip(s, FIT_SIGMA_MIN, FIT_SIGMA_MAX)


def cmd_half_width(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    w = CMD_W0 + CMD_W_SLOPE * (g - CMD_W_REF)
    return np.clip(w, CMD_W_MIN, CMD_W_MAX)


def zlocus_model(gr0: np.ndarray) -> np.ndarray:
    return ZLOCUS_SLOPE * gr0 + ZLOCUS_INTERCEPT


def zlocus_residual(gr0: np.ndarray, gz0: np.ndarray) -> np.ndarray:
    return gz0 - zlocus_model(gr0)


def finite_grz(g0: np.ndarray, r0: np.ndarray, z0: np.ndarray) -> np.ndarray:
    return np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)


def sample_indices(mask: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= max_n:
        return idx
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def safe_remove_glob(dirname: str, suffix: str = ".fits") -> None:
    if not os.path.isdir(dirname):
        return
    for name in os.listdir(dirname):
        if name.endswith(suffix):
            os.remove(os.path.join(dirname, name))


def merge_fits_list(file_list: List[str], out_fits: str, batch: int = 20) -> None:
    file_list = sorted(file_list)
    if not file_list:
        raise RuntimeError(f"No files to merge for {out_fits}")

    acc: Optional[Table] = None
    for i in range(0, len(file_list), batch):
        group = file_list[i:i + batch]
        tabs = [Table.read(fn) for fn in group]
        t = vstack(tabs, metadata_conflicts="silent")
        acc = t if acc is None else vstack([acc, t], metadata_conflicts="silent")
        print(f"[merge] {i + len(group)}/{len(file_list)} temporary files -> rows {len(acc):,}")
    assert acc is not None
    acc.write(out_fits, overwrite=True)
    print(f"[write] {out_fits} rows={len(acc):,}")


# -----------------------------------------------------------------------------
# Fitting nuisance CMD alignment on the cluster center
# -----------------------------------------------------------------------------
def fit_cluster_alignment(
    phi1: np.ndarray,
    phi2: np.ndarray,
    g0: np.ndarray,
    r0: np.ndarray,
    z0: np.ndarray,
    iso: Dict[str, np.ndarray],
) -> AlignmentResult:
    rr = np.hypot(phi1, phi2)
    m_core = rr < R_CLUSTER_IN
    m_bg = (rr >= R_CLUSTER_BG_IN) & (rr < R_CLUSTER_BG_OUT)

    gr0 = g0 - r0
    gz0 = g0 - z0
    zres = zlocus_residual(gr0, gz0)
    m_z = (
        np.isfinite(zres)
        & np.isfinite(gr0)
        & np.isfinite(g0)
        & (g0 >= FIT_GMIN)
        & (g0 <= FIT_GMAX)
        & (gr0 >= STRICT_GR_MIN)
        & (gr0 <= ZLOCUS_GR_MAX)
        & (np.abs(zres) <= ZLOCUS_TOL)
    )

    m_core &= m_z
    m_bg &= m_z

    g_core = np.asarray(g0[m_core], dtype=float)
    gr_core = np.asarray(gr0[m_core], dtype=float)
    g_bg = np.asarray(g0[m_bg], dtype=float)
    gr_bg = np.asarray(gr0[m_bg], dtype=float)

    if g_core.size < 200:
        raise RuntimeError(
            "Too few stars in the cluster-center fitting sample after z-locus. "
            f"Got {g_core.size} stars."
        )
    if g_bg.size < 200:
        raise RuntimeError(
            "Too few stars in the local background annulus after z-locus. "
            f"Got {g_bg.size} stars."
        )

    area_core = math.pi * (R_CLUSTER_IN ** 2)
    area_bg = math.pi * (R_CLUSTER_BG_OUT ** 2 - R_CLUSTER_BG_IN ** 2)
    bg_scale = area_core / area_bg

    best: Optional[AlignmentResult] = None
    gsig_core = sigma_fit_g(g_core)
    gsig_bg = sigma_fit_g(g_bg)

    # Three-parameter grid search. This is intentionally simple and transparent.
    # Score = cluster-center ridge weight - area-scaled background ridge weight.
    n_try = len(FIT_DMU_GRID) * len(FIT_DC0_GRID) * len(FIT_DC1_GRID)
    print(f"[fit] grid-searching {n_try:,} nuisance parameter combinations")
    counter = 0
    for dmu in FIT_DMU_GRID:
        for dc0 in FIT_DC0_GRID:
            for dc1 in FIT_DC1_GRID:
                ridge = build_gr_ridge(iso, DM_CLUSTER0, dmu, dc0, dc1)
                c_core_model = ridge.color_at(g_core)
                c_bg_model = ridge.color_at(g_bg)

                d_core = gr_core - c_core_model
                d_bg = gr_bg - c_bg_model

                w_core = np.exp(-0.5 * (d_core / gsig_core) ** 2)
                w_bg = np.exp(-0.5 * (d_bg / gsig_bg) ** 2)
                score = float(np.nansum(w_core) - bg_scale * np.nansum(w_bg))

                if best is None or score > best.score:
                    dm_cluster_best = DM_CLUSTER0 + dmu
                    best = AlignmentResult(
                        dmu=float(dmu),
                        dc0=float(dc0),
                        dc1=float(dc1),
                        score=score,
                        dm_cluster_best=float(dm_cluster_best),
                        dm_trailing_best=float(dm_cluster_best + DM_TRAILING_REL),
                        dm_leading_best=float(dm_cluster_best + DM_LEADING_REL),
                    )
                counter += 1
                if counter % 2000 == 0:
                    print(f"[fit] tested {counter:,}/{n_try:,} ... current best score={best.score:.3f}")

    assert best is not None
    print("[fit] best alignment:", best)
    return best


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_cluster_cmd_alignment(
    out_png: str,
    phi1: np.ndarray,
    phi2: np.ndarray,
    g0: np.ndarray,
    r0: np.ndarray,
    z0: np.ndarray,
    iso: Dict[str, np.ndarray],
    align: AlignmentResult,
) -> None:
    rr = np.hypot(phi1, phi2)
    m_core = rr < R_CLUSTER_IN
    m_bg = (rr >= R_CLUSTER_BG_IN) & (rr < R_CLUSTER_BG_OUT)

    gr0 = g0 - r0
    gz0 = g0 - z0
    zres = zlocus_residual(gr0, gz0)

    m_plot = np.isfinite(g0) & np.isfinite(gr0) & np.isfinite(gz0) & (g0 >= 18.0) & (g0 <= 23.6)
    m_core_plot = m_core & m_plot
    m_core_z = m_core_plot & (gr0 <= ZLOCUS_GR_MAX) & (np.abs(zres) <= ZLOCUS_TOL)
    m_bg_plot = m_bg & m_plot & (gr0 <= ZLOCUS_GR_MAX) & (np.abs(zres) <= ZLOCUS_TOL)

    rng = np.random.default_rng(RNG_SEED)
    idx_core = sample_indices(m_core_plot, 30_000, rng)
    idx_core_z = sample_indices(m_core_z, 30_000, rng)
    idx_bg = sample_indices(m_bg_plot, 30_000, rng)

    ridge0 = build_gr_ridge(iso, DM_CLUSTER0, 0.0, 0.0, 0.0)
    ridge1 = build_gr_ridge(iso, DM_CLUSTER0, align.dmu, align.dc0, align.dc1)

    ggrid = np.linspace(18.0, 23.5, 500)
    c0 = ridge0.color_at(ggrid)
    c1 = ridge1.color_at(ggrid)
    w1 = cmd_half_width(ggrid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    axes[0].scatter(gr0[idx_core], g0[idx_core], s=4, alpha=0.18, rasterized=True)
    axes[0].scatter(gr0[idx_bg], g0[idx_bg], s=4, alpha=0.10, rasterized=True)
    axes[0].plot(c0, ggrid, lw=2.0, label="raw isochrone")
    axes[0].set_title("cluster CMD before nuisance alignment")
    axes[0].set_xlabel(r"$(g-r)_0$")
    axes[0].set_ylabel(r"$g_0$")
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.35, 1.05)
    axes[0].set_ylim(23.6, 18.0)
    axes[0].legend(loc="best")

    axes[1].scatter(gr0[idx_core_z], g0[idx_core_z], s=4, alpha=0.18, rasterized=True, label="cluster core + z-locus")
    axes[1].plot(c1, ggrid, lw=2.0, label="aligned isochrone")
    axes[1].plot(c1 - w1, ggrid, lw=1.0, ls="--")
    axes[1].plot(c1 + w1, ggrid, lw=1.0, ls="--")
    axes[1].set_title("cluster CMD after z-locus + aligned ridge")
    axes[1].set_xlabel(r"$(g-r)_0$")
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_color_color(
    out_png: str,
    gr0: np.ndarray,
    gz0: np.ndarray,
    zres: np.ndarray,
    sel_z: np.ndarray,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    finite = np.isfinite(gr0) & np.isfinite(gz0)
    idx = sample_indices(finite, QC_SCATTER_MAX, rng)
    idx_sel = sample_indices(sel_z & finite, min(50_000, QC_SCATTER_MAX), rng)

    x = np.linspace(-0.3, 1.3, 300)
    y = zlocus_model(x)

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(gr0[idx], gz0[idx], s=2, alpha=0.08, rasterized=True, label="finite grz sample")
    plt.scatter(gr0[idx_sel], gz0[idx_sel], s=2, alpha=0.15, rasterized=True, label="z-locus selected")
    plt.plot(x, y, lw=2.0, label=r"$(g-z)_0 = 1.7(g-r)_0 - 0.17$")
    plt.plot(x, y - ZLOCUS_TOL, lw=1.0, ls="--")
    plt.plot(x, y + ZLOCUS_TOL, lw=1.0, ls="--")
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.4, 2.5)
    plt.xlabel(r"$(g-r)_0$")
    plt.ylabel(r"$(g-z)_0$")
    plt.title("Bonaca-style z-locus cut")
    plt.legend(loc="best", markerscale=3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_density_map(
    out_png: str,
    hist2d: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    dens = hist2d / area
    plt.figure(figsize=(8, 6))
    plt.imshow(
        dens.T,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_selected_cmd(out_png: str, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    plt.figure(figsize=(6.5, 5.5))
    img = np.log10(H.T + 1.0)
    plt.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.set_label(r"$\log_{10}(N+1)$")
    plt.xlabel(r"$(g-r)_0$")
    plt.ylabel(r"$g_0$")
    plt.title("strict selected sample CMD")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main chunked selection pass
# -----------------------------------------------------------------------------
def process_catalog(
    in_fits: str,
    iso: Dict[str, np.ndarray],
    align: AlignmentResult,
    outdir: str,
    chunk_size: int,
) -> Tuple[str, Dict[str, int], Dict[str, np.ndarray]]:
    tmp_dir = os.path.join(outdir, "tmp_step2_chunks")
    os.makedirs(tmp_dir, exist_ok=True)
    safe_remove_glob(tmp_dir, ".fits")

    plots_dir = os.path.join(outdir, "plots_step2")
    os.makedirs(plots_dir, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    hdul = fits.open(in_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[read] {in_fits} rows={n_rows:,}")

    needed = ["RA", "DEC", "PHI1", "PHI2", "G0", "R0", "Z0"]
    missing = [c for c in needed if c not in data.dtype.names]
    if missing:
        raise KeyError(f"Missing required columns in input FITS: {missing}")

    # Histograms for selected sample diagnostics.
    ra_edges = np.arange(RA_MIN, RA_MAX + 0.25, 0.25)
    dec_edges = np.arange(DEC_MIN, DEC_MAX + 0.25, 0.25)
    p1_edges = np.arange(PHI1_MIN, PHI1_MAX + 0.25, 0.25)
    p2_edges = np.arange(PHI2_MIN, PHI2_MAX + 0.25, 0.25)
    cmd_x_edges = np.arange(-0.35, 1.30 + 0.02, 0.02)
    cmd_y_edges = np.arange(18.0, 23.6 + 0.05, 0.05)

    H_radec = np.zeros((len(ra_edges) - 1, len(dec_edges) - 1), dtype=np.int64)
    H_phi = np.zeros((len(p1_edges) - 1, len(p2_edges) - 1), dtype=np.int64)
    H_cmd = np.zeros((len(cmd_x_edges) - 1, len(cmd_y_edges) - 1), dtype=np.int64)

    cutflow = {
        "input": 0,
        "finite_grz": 0,
        "strict_mag": 0,
        "z_locus": 0,
        "iso": 0,
        "strict_selected": 0,
    }

    # Global subsample for the color-color QC plot.
    gr_pool: List[np.ndarray] = []
    gz_pool: List[np.ndarray] = []
    zres_pool: List[np.ndarray] = []
    selz_pool: List[np.ndarray] = []
    pool_budget = QC_SCATTER_MAX

    out_files: List[str] = []
    chunk_id = 0

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]
        cutflow["input"] += len(sub)

        ra = np.asarray(sub["RA"], dtype=float)
        dec = np.asarray(sub["DEC"], dtype=float)
        phi1 = np.asarray(sub["PHI1"], dtype=float)
        phi2 = np.asarray(sub["PHI2"], dtype=float)
        g0 = np.asarray(sub["G0"], dtype=float)
        r0 = np.asarray(sub["R0"], dtype=float)
        z0 = np.asarray(sub["Z0"], dtype=float)

        m_fin = finite_grz(g0, r0, z0)
        cutflow["finite_grz"] += int(np.sum(m_fin))

        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = zlocus_residual(gr0, gz0)

        m_mag = m_fin & (g0 >= STRICT_GMIN) & (g0 < STRICT_GMAX) & (gr0 >= STRICT_GR_MIN) & (gr0 <= STRICT_GR_MAX)
        cutflow["strict_mag"] += int(np.sum(m_mag))

        m_z = m_mag & (gr0 <= ZLOCUS_GR_MAX) & np.isfinite(zres) & (np.abs(zres) <= ZLOCUS_TOL)
        cutflow["z_locus"] += int(np.sum(m_z))

        dm_used = np.where(phi1 > 0.0, align.dm_leading_best, align.dm_trailing_best)
        # Build two ridge models only once per chunk.
        ridge_tr = build_gr_ridge(iso, align.dm_trailing_best, 0.0, align.dc0, align.dc1)
        ridge_ld = build_gr_ridge(iso, align.dm_leading_best, 0.0, align.dc0, align.dc1)
        c_model = np.full_like(g0, np.nan, dtype=float)
        m_tr = phi1 <= 0.0
        m_ld = ~m_tr
        c_model[m_tr] = ridge_tr.color_at(g0[m_tr])
        c_model[m_ld] = ridge_ld.color_at(g0[m_ld])

        dcol = gr0 - c_model
        width = cmd_half_width(g0)
        m_iso = m_z & np.isfinite(dcol) & (np.abs(dcol) <= width)
        cutflow["iso"] += int(np.sum(m_iso))
        cutflow["strict_selected"] += int(np.sum(m_iso))

        # QC subsample for color-color.
        if pool_budget > 0:
            m_pool = m_mag & np.isfinite(gz0)
            idx_pool = np.flatnonzero(m_pool)
            if idx_pool.size > 0:
                take = min(pool_budget, min(50_000, idx_pool.size))
                pick = np.sort(rng.choice(idx_pool, size=take, replace=False))
                gr_pool.append(gr0[pick].astype(np.float32))
                gz_pool.append(gz0[pick].astype(np.float32))
                zres_pool.append(zres[pick].astype(np.float32))
                selz_pool.append(m_z[pick])
                pool_budget -= take

        if np.any(m_iso):
            sel = m_iso
            h, _, _ = np.histogram2d(ra[sel], dec[sel], bins=[ra_edges, dec_edges])
            H_radec += h.astype(np.int64)
            h, _, _ = np.histogram2d(phi1[sel], phi2[sel], bins=[p1_edges, p2_edges])
            H_phi += h.astype(np.int64)
            h, _, _ = np.histogram2d(gr0[sel], g0[sel], bins=[cmd_x_edges, cmd_y_edges])
            H_cmd += h.astype(np.int64)

            t = Table(sub[sel])
            t["GR0"] = gr0[sel].astype(np.float32)
            t["GZ0"] = gz0[sel].astype(np.float32)
            t["ZLOCUS_RESID"] = zres[sel].astype(np.float32)
            t["ISO_DCOL"] = dcol[sel].astype(np.float32)
            t["ISO_HALF_WIDTH"] = width[sel].astype(np.float32)
            t["DM_USED"] = dm_used[sel].astype(np.float32)
            t["SEL_ZLOCUS"] = np.ones(np.sum(sel), dtype=np.int16)
            t["SEL_ISO"] = np.ones(np.sum(sel), dtype=np.int16)
            t["SEL_STRICT"] = np.ones(np.sum(sel), dtype=np.int16)

            out_chunk = os.path.join(tmp_dir, f"step2_members_chunk{chunk_id:04d}.fits")
            t.write(out_chunk, overwrite=True)
            out_files.append(out_chunk)

        print(
            f"[chunk {chunk_id:04d}] rows {start:,}-{stop:,} | "
            f"mag={np.sum(m_mag):,} z={np.sum(m_z):,} iso={np.sum(m_iso):,}"
        )
        chunk_id += 1

    hdul.close()

    if not out_files:
        raise RuntimeError("Selection produced no output rows.")

    out_members = os.path.join(outdir, "pal5_step2_strict_members.fits")
    merge_fits_list(out_files, out_members, batch=20)

    # QC plots.
    if gr_pool:
        gr_pool_arr = np.concatenate(gr_pool)
        gz_pool_arr = np.concatenate(gz_pool)
        zres_pool_arr = np.concatenate(zres_pool)
        selz_pool_arr = np.concatenate(selz_pool)
        plot_color_color(
            os.path.join(plots_dir, "qc_color_color_zlocus.png"),
            gr_pool_arr,
            gz_pool_arr,
            zres_pool_arr,
            selz_pool_arr,
        )

    plot_density_map(
        os.path.join(plots_dir, "qc_selected_density_radec.png"),
        H_radec,
        ra_edges,
        dec_edges,
        "strict selected sample: RA-Dec number density",
        "RA [deg]",
        "Dec [deg]",
    )
    plot_density_map(
        os.path.join(plots_dir, "qc_selected_density_phi12.png"),
        H_phi,
        p1_edges,
        p2_edges,
        "strict selected sample: Pal 5 frame number density",
        r"$\phi_1$ [deg]",
        r"$\phi_2$ [deg]",
    )
    plot_selected_cmd(
        os.path.join(plots_dir, "qc_selected_cmd_gr_g.png"),
        H_cmd,
        cmd_x_edges,
        cmd_y_edges,
    )

    hist_data = {
        "H_radec": H_radec,
        "H_phi": H_phi,
        "H_cmd": H_cmd,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "p1_edges": p1_edges,
        "p2_edges": p2_edges,
        "cmd_x_edges": cmd_x_edges,
        "cmd_y_edges": cmd_y_edges,
    }
    return out_members, cutflow, hist_data


# -----------------------------------------------------------------------------
# First scan: gather small samples for alignment diagnostics
# -----------------------------------------------------------------------------
def gather_cluster_samples(in_fits: str, chunk_size: int) -> Dict[str, np.ndarray]:
    hdul = fits.open(in_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[scan] gathering cluster-center / annulus samples from {n_rows:,} rows")

    keep: Dict[str, List[np.ndarray]] = {
        "phi1": [],
        "phi2": [],
        "g0": [],
        "r0": [],
        "z0": [],
    }

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]
        phi1 = np.asarray(sub["PHI1"], dtype=float)
        phi2 = np.asarray(sub["PHI2"], dtype=float)
        rr = np.hypot(phi1, phi2)
        m_keep = rr < R_CLUSTER_BG_OUT
        if np.any(m_keep):
            keep["phi1"].append(phi1[m_keep].astype(np.float32))
            keep["phi2"].append(phi2[m_keep].astype(np.float32))
            keep["g0"].append(np.asarray(sub["G0"], dtype=float)[m_keep].astype(np.float32))
            keep["r0"].append(np.asarray(sub["R0"], dtype=float)[m_keep].astype(np.float32))
            keep["z0"].append(np.asarray(sub["Z0"], dtype=float)[m_keep].astype(np.float32))
        if (start // chunk_size) % 5 == 0:
            print(f"[scan] {start:,}/{n_rows:,}")

    hdul.close()
    out = {k: np.concatenate(v) if v else np.array([], dtype=np.float32) for k, v in keep.items()}
    print(f"[scan] kept {len(out['g0']):,} stars within r < {R_CLUSTER_BG_OUT:.2f} deg of the cluster center")
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pal 5 step 2 member selection (Bonaca-style baseline)")
    parser.add_argument("--input", default="final_g25_preproc.fits", help="Input preprocessed FITS catalog")
    parser.add_argument("--iso", default="pal5.dat", help="Input isochrone ASCII file")
    parser.add_argument("--outdir", default="step2_outputs", help="Output directory")
    parser.add_argument("--chunk", type=int, default=2_000_000, help="Chunk size for memmap processing")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots_step2")
    os.makedirs(plots_dir, exist_ok=True)

    iso = read_parsec_like_isochrone(args.iso)

    # Small first pass: fit nuisance CMD alignment on the cluster center CMD.
    cluster = gather_cluster_samples(args.input, args.chunk)
    align = fit_cluster_alignment(
        cluster["phi1"], cluster["phi2"], cluster["g0"], cluster["r0"], cluster["z0"], iso
    )

    plot_cluster_cmd_alignment(
        os.path.join(plots_dir, "qc_cluster_cmd_alignment.png"),
        cluster["phi1"], cluster["phi2"], cluster["g0"], cluster["r0"], cluster["z0"], iso, align
    )

    # Main selection pass.
    out_members, cutflow, _ = process_catalog(args.input, iso, align, args.outdir, args.chunk)

    # Save alignment + cutflow reports.
    align_json = os.path.join(args.outdir, "pal5_step2_alignment.json")
    with open(align_json, "w", encoding="utf-8") as f:
        json.dump(asdict(align), f, indent=2)
    print(f"[write] {align_json}")

    cutflow_txt = os.path.join(args.outdir, "pal5_step2_cutflow.txt")
    with open(cutflow_txt, "w", encoding="utf-8") as f:
        f.write("Pal 5 step 2 member-selection cut-flow\n")
        f.write(f"INPUT_FITS = {args.input}\n")
        f.write(f"ISO_FILE   = {args.iso}\n")
        f.write(f"OUTPUT_FITS = {out_members}\n\n")
        total = max(cutflow["input"], 1)
        for key, val in cutflow.items():
            f.write(f"{key:24s}: {val:12d}   ({100.0 * val / total:8.4f}%)\n")
    print(f"[write] {cutflow_txt}")

    summary_json = os.path.join(args.outdir, "pal5_step2_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.input,
                "iso": args.iso,
                "output_members": out_members,
                "alignment": asdict(align),
                "cutflow": cutflow,
                "strict_config": {
                    "STRICT_GMIN": STRICT_GMIN,
                    "STRICT_GMAX": STRICT_GMAX,
                    "ZLOCUS_TOL": ZLOCUS_TOL,
                    "CMD_W0": CMD_W0,
                    "CMD_W_SLOPE": CMD_W_SLOPE,
                    "CMD_W_MIN": CMD_W_MIN,
                    "CMD_W_MAX": CMD_W_MAX,
                },
            },
            f,
            indent=2,
        )
    print(f"[write] {summary_json}")

    print("\nDone.")
    print(f"Strict member catalog: {out_members}")
    print(f"Plots directory       : {plots_dir}")


if __name__ == "__main__":
    main()
