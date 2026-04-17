#!/usr/bin/env python3
"""
Pal 5 step 3b: selection-aware Bonaca-style 1D stream density modeling.

This script is the stable follow-up to the exploratory step 3 run. It keeps the
same intrinsic morphology model used by Bonaca+2020 in Section 3:

    intrinsic stream  = single Gaussian in phi2
    intrinsic background = linearly varying function of phi2

but upgrades the *observation model* by multiplying both components by a local
selection-response template eta(phi2). This is needed because the Pal 5 region
has targeted deeper imaging whose depth stripe is partially aligned with the
stream, so a naive fit can absorb survey-depth structure into the stream width
or density.

What this script does
---------------------
1. Reads the strict member catalog from step 2 (signal sample).
2. Reconstructs a matched control sample from the preprocessed parent catalog
   using the same strict magnitude + z-locus cuts as step 2, but selecting
   isochrone sidebands instead of the signal ridge.
3. Builds, in each overlapping phi1 window, a local eta(phi2) template from the
   control sample and optionally a depth-based eta(phi2) template from
   PSFDEPTH_G / PSFDEPTH_Z.
4. Fits the observed phi2 distribution of the strict members with
      eta(phi2) x [ f * Gaussian(mu, sigma) + (1-f) * linear_background ]
   using a robust multistart MAP fit with transformed parameters.
5. If emcee is available (or requested), samples the posterior near the MAP;
   otherwise falls back to a Laplace / finite-difference approximation.
6. Writes profile tables and QC plots, using only genuinely converged bins in
   the main science plots.

Inputs
------
- step2_outputs/pal5_step2_strict_members.fits
- final_g25_preproc.fits
- step2_outputs/pal5_step2_summary.json
- pal5.dat
- optionally a prior track file from the exploratory step 3 run

Outputs
-------
- pal5_step3b_profiles.fits / .csv
- pal5_step3b_summary.json
- pal5_step3b_control_sidebands.fits
- QC plots in the output directory
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import emcee  # type: ignore
    HAVE_EMCEE = True
except Exception:
    emcee = None
    HAVE_EMCEE = False


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_SIGNAL = "step2_outputs/pal5_step2_strict_members.fits"
DEFAULT_PREPROC = "final_g25_preproc.fits"
DEFAULT_STEP2_SUMMARY = "step2_outputs/pal5_step2_summary.json"
DEFAULT_ISO = "pal5.dat"
DEFAULT_OUTDIR = "step3b_outputs"
DEFAULT_MU_PRIOR = ""

PHI1_MIN = -20.0
PHI1_MAX = 10.0
PHI1_STEP = 0.75
WINDOW_SCALE = 1.5
WINDOW_WIDTH_PHI1 = WINDOW_SCALE * PHI1_STEP
FIT_HALFWIDTH = 1.50
MIN_SIGNAL_STARS = 60
CLUSTER_MASK_HALFWIDTH = 0.75

PASS0_PHI2_MIN = -2.5
PASS0_PHI2_MAX = 2.5

# Control-sample sidebands around the strict isochrone ridge.
SIDEBAND_GAP = 0.03
SIDEBAND_WIDTH = 0.12

# eta template construction.
ETA_BIN = 0.05
ETA_SMOOTH_BINS = 1.5
ETA_MIN = 0.10
ETA_MAX = 10.0
DEPTH_STRENGTH = 0.35

# Parameter bounds / priors.
SIGMA_MIN = 0.03
SIGMA_MAX = 1.20
UF_MIN = -7.0
UF_MAX = 7.0
RAW_TILT_MIN = -6.0
RAW_TILT_MAX = 6.0
MU_PRIOR_SIGMA = 0.35
MU_START_OFFSETS = (-0.12, 0.0, 0.12)
TRACK_PLOT_OUTLIER_ABS = 0.35
TRACK_COHERENCE_ABS = 0.30
SIGMA_COHERENCE_MIN_RATIO = 0.60
SIGMA_COHERENCE_MAX_RATIO = 2.50
MU_CLEAN_ERR_FLOOR = 0.08
SIGMA_CLEAN_ERR_FLOOR = 0.05

# Optional sampling.
DEFAULT_SAMPLER = "auto"  # auto | emcee | map
DEFAULT_NWALKERS = 48
DEFAULT_BURN = 256
DEFAULT_STEPS = 512

# QC plots.
MAP_BIN = 0.10
FULL_PHI1_MIN = -25.0
FULL_PHI1_MAX = 20.0
FULL_PHI2_MIN = -5.0
FULL_PHI2_MAX = 10.0
EXAMPLE_BIN_CENTERS = (-13.0, -7.0, -3.0, 0.0, 3.0, 6.0)
RNG_SEED = 12345


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(np.asarray(p), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def finite_grz(g0: np.ndarray, r0: np.ndarray, z0: np.ndarray) -> np.ndarray:
    return np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)


def zlocus_model(gr0: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * gr0 + intercept


def zlocus_residual(gr0: np.ndarray, gz0: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return gz0 - zlocus_model(gr0, slope, intercept)


def safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, fill: float = 1.0) -> np.ndarray:
    out = np.full_like(x, fill, dtype=float)
    ok = np.isfinite(x)
    if np.sum(ok) == 0:
        return out
    out[ok] = np.interp(x[ok], xp, fp, left=fp[0], right=fp[-1])
    return out


def fill_nan_linear(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.array(y, copy=True)
    m = np.isfinite(out)
    if m.sum() == 0:
        return np.ones_like(out)
    if m.sum() == 1:
        out[~m] = out[m][0]
        return out
    x = np.arange(len(out), dtype=float)
    out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def mad_scale(x: np.ndarray, floor: float = 0.03) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return floor
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, floor)


def histogram_mode(x: np.ndarray, lo: float, hi: float, bin_size: float) -> float:
    if x.size == 0:
        return 0.0
    edges = np.arange(lo, hi + bin_size, bin_size)
    if edges.size < 2:
        return float(np.nanmedian(x))
    hist, edges = np.histogram(x, bins=edges)
    if hist.sum() == 0:
        return float(np.nanmedian(x))
    idx = int(np.argmax(hist))
    return 0.5 * (edges[idx] + edges[idx + 1])


def smooth_track_by_arm(phi1: np.ndarray, mu: np.ndarray) -> np.ndarray:
    out = np.array(mu, dtype=float, copy=True)
    finite = np.isfinite(out)
    if finite.sum() >= 2:
        out[~finite] = np.interp(phi1[~finite], phi1[finite], out[finite])
    else:
        out[~finite] = 0.0
    for arm_mask in (phi1 <= 0.0, phi1 >= 0.0):
        idx = np.where(arm_mask)[0]
        if idx.size < 5:
            continue
        sub = out[idx]
        win = min(9, idx.size if idx.size % 2 == 1 else idx.size - 1)
        if win < 5:
            continue
        out[idx] = savgol_filter(sub, window_length=win, polyorder=2, mode="interp")
    return out


def smooth_profile_by_arm(phi1: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    for arm_mask in (phi1 <= 0.0, phi1 >= 0.0):
        idx = np.where(arm_mask)[0]
        if idx.size == 0:
            continue
        sub = np.array(out[idx], dtype=float, copy=True)
        finite = np.isfinite(sub)
        if finite.sum() == 0:
            continue
        if finite.sum() == 1:
            sub[~finite] = sub[finite][0]
        else:
            sub[~finite] = np.interp(phi1[idx][~finite], phi1[idx][finite], sub[finite])
        win = min(9, idx.size if idx.size % 2 == 1 else idx.size - 1)
        if win >= 5:
            sub = savgol_filter(sub, window_length=win, polyorder=2, mode="interp")
        out[idx] = sub
    return out


# -----------------------------------------------------------------------------
# Step-2 isochrone helpers (copied / adapted from step 2 for reproducibility)
# -----------------------------------------------------------------------------
class RidgeModel:
    def __init__(self, g_model: np.ndarray, gr_model: np.ndarray):
        g = np.asarray(g_model, dtype=float)
        c = np.asarray(gr_model, dtype=float)
        ok = np.isfinite(g) & np.isfinite(c)
        g = g[ok]
        c = c[ok]
        if g.size < 5:
            raise ValueError("Too few valid isochrone points.")

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
        out = np.full_like(g_obs, np.nan, dtype=float)
        ok = np.isfinite(g_obs) & (g_obs >= self.gmin) & (g_obs <= self.gmax)
        if np.any(ok):
            out[ok] = np.interp(g_obs[ok], self.g, self.c)
        return out


def read_parsec_like_isochrone(path: str) -> Dict[str, np.ndarray]:
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
    }


def build_gr_ridge(iso: Dict[str, np.ndarray], dm: float, dc0: float, dc1: float, g_ref: float = 21.0) -> RidgeModel:
    g_app = iso["g_abs"] + dm
    gr = iso["gr_abs"] + dc0 + dc1 * (g_app - g_ref)
    keep = np.isfinite(g_app) & np.isfinite(gr) & (g_app >= 17.0) & (g_app <= 25.5)
    return RidgeModel(g_app[keep], gr[keep])


def cmd_half_width(g: np.ndarray, cfg: Dict[str, float]) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    w = cfg["CMD_W0"] + cfg["CMD_W_SLOPE"] * (g - cfg.get("CMD_W_REF", 20.0))
    return np.clip(w, cfg["CMD_W_MIN"], cfg["CMD_W_MAX"])


# -----------------------------------------------------------------------------
# Model / templates
# -----------------------------------------------------------------------------
@dataclass
class Step2Models:
    summary_path: str
    strict_cfg: Dict[str, float]
    alignment: Dict[str, float]
    ridge_trailing: RidgeModel
    ridge_leading: RidgeModel
    z_slope: float
    z_intercept: float
    z_tol: float
    z_gr_max: float


@dataclass
class TemplateBundle:
    grid: np.ndarray
    eta_control: np.ndarray
    eta_depth: np.ndarray
    eta_total: np.ndarray
    n_control: int
    n_depth: int
    fit_lo: float
    fit_hi: float


@dataclass
class BinResult:
    phi1_center: float
    phi1_lo: float
    phi1_hi: float
    fit_lo: float
    fit_hi: float
    n_signal: int
    n_control: int
    n_depth: int
    cluster_bin: bool
    success: bool
    sampler_used: str
    optimizer_success: bool
    sampler_success: bool
    message: str
    mu_prior_center: float
    mu_prior_sigma: float
    f_stream: float = np.nan
    f_stream_err: float = np.nan
    mu: float = np.nan
    mu_err: float = np.nan
    sigma: float = np.nan
    sigma_err: float = np.nan
    bg_tilt: float = np.nan
    bg_tilt_err: float = np.nan
    n_stream: float = np.nan
    n_stream_err: float = np.nan
    linear_density: float = np.nan
    linear_density_err: float = np.nan
    peak_surface_density: float = np.nan
    peak_surface_density_err: float = np.nan
    map_neglogpost: float = np.nan
    acc_frac: float = np.nan


def load_step2_models(step2_summary_path: str, iso_path: str) -> Step2Models:
    with open(step2_summary_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    strict_cfg = dict(meta["strict_config"])
    strict_cfg.setdefault("STRICT_GMIN", 20.0)
    strict_cfg.setdefault("STRICT_GMAX", 23.0)
    strict_cfg.setdefault("STRICT_GR_MIN", -0.35)
    strict_cfg.setdefault("STRICT_GR_MAX", 1.25)
    strict_cfg.setdefault("CMD_W_REF", 20.0)
    alignment = dict(meta["alignment"])

    iso = read_parsec_like_isochrone(iso_path)
    ridge_trailing = build_gr_ridge(iso, alignment["dm_trailing_best"], alignment["dc0"], alignment["dc1"])
    ridge_leading = build_gr_ridge(iso, alignment["dm_leading_best"], alignment["dc0"], alignment["dc1"])

    return Step2Models(
        summary_path=step2_summary_path,
        strict_cfg=strict_cfg,
        alignment=alignment,
        ridge_trailing=ridge_trailing,
        ridge_leading=ridge_leading,
        z_slope=1.7,
        z_intercept=-0.17,
        z_tol=float(strict_cfg.get("ZLOCUS_TOL", 0.10)),
        z_gr_max=1.20,
    )


def choose_ridge(phi1: np.ndarray, models: Step2Models, g0: np.ndarray) -> np.ndarray:
    out = np.full_like(g0, np.nan, dtype=float)
    m_tr = phi1 <= 0.0
    m_ld = ~m_tr
    if np.any(m_tr):
        out[m_tr] = models.ridge_trailing.color_at(g0[m_tr])
    if np.any(m_ld):
        out[m_ld] = models.ridge_leading.color_at(g0[m_ld])
    return out


def build_control_and_depth_views(
    preproc_fits: str,
    models: Step2Models,
    outdir: Path,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """
    Build two parent views from the preprocessed catalog:
      1) control sidebands: strict_mag + z_locus + isochrone sidebands
      2) z-locus parent: strict_mag + z_locus only, used for depth templates
    """
    print(f"[read parent] {preproc_fits}")
    hdul = fits.open(preproc_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[info] parent rows={n_rows:,}")

    needed = ["PHI1", "PHI2", "G0", "R0", "Z0", "PSFDEPTH_G"]
    missing = [c for c in needed if c not in data.dtype.names]
    if missing:
        raise KeyError(f"Missing required columns in parent FITS: {missing}")

    have_depth_z = "PSFDEPTH_Z" in data.dtype.names

    ctl_phi1_list: List[np.ndarray] = []
    ctl_phi2_list: List[np.ndarray] = []
    zl_phi1_list: List[np.ndarray] = []
    zl_phi2_list: List[np.ndarray] = []
    zl_gd_list: List[np.ndarray] = []
    zl_zd_list: List[np.ndarray] = []

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]

        phi1 = np.asarray(sub["PHI1"], dtype=float)
        phi2 = np.asarray(sub["PHI2"], dtype=float)
        g0 = np.asarray(sub["G0"], dtype=float)
        r0 = np.asarray(sub["R0"], dtype=float)
        z0 = np.asarray(sub["Z0"], dtype=float)
        gd = np.asarray(sub["PSFDEPTH_G"], dtype=float)
        zd = np.asarray(sub["PSFDEPTH_Z"], dtype=float) if have_depth_z else np.full_like(gd, np.nan)

        m_fin = finite_grz(g0, r0, z0)
        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = zlocus_residual(gr0, gz0, models.z_slope, models.z_intercept)

        cfg = models.strict_cfg
        m_mag = (
            m_fin
            & (g0 >= cfg["STRICT_GMIN"])
            & (g0 < cfg["STRICT_GMAX"])
            & (gr0 >= cfg["STRICT_GR_MIN"])
            & (gr0 <= cfg["STRICT_GR_MAX"])
        )
        m_z = m_mag & (gr0 <= models.z_gr_max) & np.isfinite(zres) & (np.abs(zres) <= models.z_tol)

        c_model = choose_ridge(phi1, models, g0)
        dcol = gr0 - c_model
        w = cmd_half_width(g0, cfg)

        blue = (dcol <= -(w + SIDEBAND_GAP)) & (dcol >= -(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        red = (dcol >= +(w + SIDEBAND_GAP)) & (dcol <= +(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        m_ctl = m_z & np.isfinite(dcol) & (blue | red)

        if np.any(m_ctl):
            ctl_phi1_list.append(phi1[m_ctl].astype(np.float32))
            ctl_phi2_list.append(phi2[m_ctl].astype(np.float32))
        if np.any(m_z):
            zl_phi1_list.append(phi1[m_z].astype(np.float32))
            zl_phi2_list.append(phi2[m_z].astype(np.float32))
            zl_gd_list.append(gd[m_z].astype(np.float32))
            zl_zd_list.append(zd[m_z].astype(np.float32))

        if ((start // chunk_size) % 5) == 0 or stop == n_rows:
            print(f"[parent chunk] {start:,}-{stop:,} | z={np.sum(m_z):,} ctl={np.sum(m_ctl):,}")

    if len(ctl_phi1_list) == 0:
        raise RuntimeError("Control sideband sample is empty.")
    if len(zl_phi1_list) == 0:
        raise RuntimeError("z-locus parent sample is empty.")

    ctl_phi1 = np.concatenate(ctl_phi1_list)
    ctl_phi2 = np.concatenate(ctl_phi2_list)
    zl_phi1 = np.concatenate(zl_phi1_list)
    zl_phi2 = np.concatenate(zl_phi2_list)
    zl_gd = np.concatenate(zl_gd_list)
    zl_zd = np.concatenate(zl_zd_list)

    control_tab = Table({
        "PHI1": ctl_phi1,
        "PHI2": ctl_phi2,
    })
    control_path = outdir / "pal5_step3b_control_sidebands.fits"
    control_tab.write(control_path, overwrite=True)
    print(f"[write] {control_path}")

    return {
        "ctl_phi1": ctl_phi1,
        "ctl_phi2": ctl_phi2,
        "zl_phi1": zl_phi1,
        "zl_phi2": zl_phi2,
        "zl_gd": zl_gd,
        "zl_zd": zl_zd,
    }


def build_template_grid(fit_lo: float, fit_hi: float, binw: float = ETA_BIN) -> np.ndarray:
    n = max(64, int(np.ceil((fit_hi - fit_lo) / binw)) + 1)
    return np.linspace(fit_lo, fit_hi, n)


def build_control_eta(phi2_control: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if phi2_control.size < 20:
        return np.ones_like(grid)
    edges = np.concatenate([[grid[0] - 0.5 * (grid[1] - grid[0])], 0.5 * (grid[1:] + grid[:-1]), [grid[-1] + 0.5 * (grid[1] - grid[0])]])
    hist, _ = np.histogram(phi2_control, bins=edges)
    smooth = gaussian_filter1d(hist.astype(float), ETA_SMOOTH_BINS, mode="nearest")
    smooth = fill_nan_linear(smooth)
    mean = np.mean(smooth) if np.mean(smooth) > 0 else 1.0
    eta = smooth / mean
    return np.clip(eta, ETA_MIN, ETA_MAX)


def build_depth_eta(phi2_parent: np.ndarray, gd_parent: np.ndarray, zd_parent: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if phi2_parent.size < 20:
        return np.ones_like(grid)

    edges = np.concatenate([[grid[0] - 0.5 * (grid[1] - grid[0])], 0.5 * (grid[1:] + grid[:-1]), [grid[-1] + 0.5 * (grid[1] - grid[0])]])
    idx = np.digitize(phi2_parent, edges) - 1
    nbin = len(grid)
    med_g = np.full(nbin, np.nan, dtype=float)
    med_z = np.full(nbin, np.nan, dtype=float)

    for j in range(nbin):
        m = idx == j
        if np.any(m):
            gsub = gd_parent[m]
            zsub = zd_parent[m]
            gsub = gsub[np.isfinite(gsub)]
            zsub = zsub[np.isfinite(zsub)]
            if gsub.size > 0:
                med_g[j] = np.median(gsub)
            if zsub.size > 0:
                med_z[j] = np.median(zsub)

    med_g = gaussian_filter1d(fill_nan_linear(med_g), ETA_SMOOTH_BINS, mode="nearest")
    med_z = gaussian_filter1d(fill_nan_linear(med_z), ETA_SMOOTH_BINS, mode="nearest")

    zg = (med_g - np.median(med_g)) / mad_scale(med_g, floor=0.05)
    zz = (med_z - np.median(med_z)) / mad_scale(med_z, floor=0.05)
    s = 0.5 * (zg + zz)
    eta = np.exp(DEPTH_STRENGTH * s)
    eta /= max(np.mean(eta), 1e-6)
    return np.clip(eta, ETA_MIN, ETA_MAX)


def build_eta_template(
    ctl_phi2: np.ndarray,
    zl_phi2: np.ndarray,
    zl_gd: np.ndarray,
    zl_zd: np.ndarray,
    fit_lo: float,
    fit_hi: float,
    eta_mode: str,
) -> TemplateBundle:
    grid = build_template_grid(fit_lo, fit_hi, binw=ETA_BIN)
    eta_control = build_control_eta(ctl_phi2, grid)
    eta_depth = build_depth_eta(zl_phi2, zl_gd, zl_zd, grid)

    if eta_mode == "control":
        eta_total = eta_control.copy()
    elif eta_mode == "depth":
        eta_total = eta_depth.copy()
    elif eta_mode == "control_times_depth":
        eta_total = eta_control * eta_depth
        eta_total /= max(np.mean(eta_total), 1e-6)
        eta_total = np.clip(eta_total, ETA_MIN, ETA_MAX)
    else:
        raise ValueError(f"Unknown eta_mode: {eta_mode}")

    return TemplateBundle(
        grid=grid,
        eta_control=eta_control,
        eta_depth=eta_depth,
        eta_total=eta_total,
        n_control=int(len(ctl_phi2)),
        n_depth=int(len(zl_phi2)),
        fit_lo=float(fit_lo),
        fit_hi=float(fit_hi),
    )


# -----------------------------------------------------------------------------
# Probability model
# -----------------------------------------------------------------------------

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return coef * np.exp(-0.5 * z * z)


def transform_bg_tilt(raw_tilt: float, lo: float, hi: float) -> float:
    half = 0.5 * (hi - lo)
    return 0.95 * np.tanh(raw_tilt) / max(half, 1e-6)


def linear_bg_shape(x: np.ndarray, raw_tilt: float, lo: float, hi: float) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    tilt = transform_bg_tilt(raw_tilt, lo, hi)
    y = 1.0 + tilt * (x - mid)
    return np.clip(y, 1e-6, None)


def log_prior_transformed(theta: np.ndarray, mu_prior_center: float, mu_prior_sigma: float, lo: float, hi: float) -> float:
    u_f, mu, log_sigma, raw_tilt = theta
    if not np.isfinite(u_f) or not (UF_MIN <= u_f <= UF_MAX):
        return -np.inf
    if not np.isfinite(mu) or not (lo <= mu <= hi):
        return -np.inf
    if not np.isfinite(log_sigma) or not (np.log(SIGMA_MIN) <= log_sigma <= np.log(SIGMA_MAX)):
        return -np.inf
    if not np.isfinite(raw_tilt) or not (RAW_TILT_MIN <= raw_tilt <= RAW_TILT_MAX):
        return -np.inf

    f = float(sigmoid(u_f))
    # Uniform prior in f => transformed prior includes the Jacobian df/du.
    lp = math.log(max(f * (1.0 - f), 1e-300))
    lp += -0.5 * ((mu - mu_prior_center) / mu_prior_sigma) ** 2
    # Uniform prior in log_sigma within bounds => constant.
    # Weakly flat prior in raw_tilt within bounds => constant.
    return lp


def component_pdfs_observed(
    phi2_eval: np.ndarray,
    theta: np.ndarray,
    template: TemplateBundle,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u_f, mu, log_sigma, raw_tilt = theta
    f = float(sigmoid(u_f))
    sigma = float(np.exp(log_sigma))

    eta_i = safe_interp(phi2_eval, template.grid, template.eta_total, fill=1.0)
    eta_g = template.eta_total
    grid = template.grid

    s_i = eta_i * gaussian_pdf(phi2_eval, mu, sigma)
    s_g = eta_g * gaussian_pdf(grid, mu, sigma)
    b_i = eta_i * linear_bg_shape(phi2_eval, raw_tilt, template.fit_lo, template.fit_hi)
    b_g = eta_g * linear_bg_shape(grid, raw_tilt, template.fit_lo, template.fit_hi)

    I_s = float(np.trapz(s_g, grid))
    I_b = float(np.trapz(b_g, grid))
    if I_s <= 0 or I_b <= 0 or not np.isfinite(I_s) or not np.isfinite(I_b):
        return np.full_like(phi2_eval, np.nan), np.full_like(phi2_eval, np.nan), f, sigma, I_s, I_b

    p_s = s_i / I_s
    p_b = b_i / I_b
    return p_s, p_b, f, sigma, I_s, I_b


def neg_log_posterior(theta: np.ndarray, phi2: np.ndarray, template: TemplateBundle, mu_prior_center: float, mu_prior_sigma: float) -> float:
    lp = log_prior_transformed(theta, mu_prior_center, mu_prior_sigma, template.fit_lo, template.fit_hi)
    if not np.isfinite(lp):
        return np.inf
    p_s, p_b, f, sigma, _, _ = component_pdfs_observed(phi2, theta, template)
    if not np.all(np.isfinite(p_s)) or not np.all(np.isfinite(p_b)):
        return np.inf
    p = f * p_s + (1.0 - f) * p_b
    if np.any(p <= 0.0) or not np.all(np.isfinite(p)):
        return np.inf
    ll = np.sum(np.log(np.clip(p, 1e-300, None)))
    return float(-(ll + lp))


def log_posterior(theta: np.ndarray, phi2: np.ndarray, template: TemplateBundle, mu_prior_center: float, mu_prior_sigma: float) -> float:
    nlp = neg_log_posterior(theta, phi2, template, mu_prior_center, mu_prior_sigma)
    return -nlp if np.isfinite(nlp) else -np.inf


def numerical_hessian(func, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    H = np.zeros((n, n), dtype=float)
    f0 = func(x0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = eps[i]
        f_ip = func(x0 + ei)
        f_im = func(x0 - ei)
        H[i, i] = (f_ip - 2.0 * f0 + f_im) / (eps[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = eps[j]
            f_pp = func(x0 + ei + ej)
            f_pm = func(x0 + ei - ej)
            f_mp = func(x0 - ei + ej)
            f_mm = func(x0 - ei - ej)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps[i] * eps[j])
            H[i, j] = H_ij
            H[j, i] = H_ij
    return H


def start_points(mu_center: float, fit_lo: float, fit_hi: float) -> List[np.ndarray]:
    pts: List[np.ndarray] = []
    mu_candidates = []
    for off in MU_START_OFFSETS:
        mu0 = np.clip(mu_center + off, fit_lo + 0.02, fit_hi - 0.02)
        mu_candidates.append(mu0)
    for f0 in (0.05, 0.12, 0.20, 0.35):
        for s0 in (0.07, 0.12, 0.20, 0.32):
            for t0 in (-1.5, 0.0, 1.5):
                for mu0 in mu_candidates:
                    pts.append(np.array([logit(f0), mu0, np.log(s0), t0], dtype=float))
    return pts


def fit_single_bin_map(phi2: np.ndarray, template: TemplateBundle, mu_prior_center: float, mu_prior_sigma: float) -> Tuple[Optional[np.ndarray], Optional[float], bool, str]:
    if len(phi2) < MIN_SIGNAL_STARS:
        return None, None, False, "too_few_signal_stars"

    best_x = None
    best_fun = np.inf
    best_msg = "optimizer_failed"
    best_success = False
    bounds = [(UF_MIN, UF_MAX), (template.fit_lo, template.fit_hi), (np.log(SIGMA_MIN), np.log(SIGMA_MAX)), (RAW_TILT_MIN, RAW_TILT_MAX)]
    starts = start_points(mu_prior_center, template.fit_lo, template.fit_hi)

    for x0 in starts:
        try:
            res = minimize(
                neg_log_posterior,
                x0=x0,
                args=(phi2, template, mu_prior_center, mu_prior_sigma),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10},
            )
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_x = np.asarray(res.x, dtype=float)
            best_fun = float(res.fun)
            best_msg = str(getattr(res, "message", ""))
            best_success = bool(res.success)

    return best_x, best_fun, best_success, best_msg


def posterior_from_map_only(theta_map: np.ndarray, phi2: np.ndarray, template: TemplateBundle, mu_prior_center: float, mu_prior_sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float]:
    func = lambda th: neg_log_posterior(th, phi2, template, mu_prior_center, mu_prior_sigma)
    eps = np.array([0.05, 0.01, 0.05, 0.05], dtype=float)
    try:
        H = numerical_hessian(func, theta_map, eps)
        H = 0.5 * (H + H.T)
        cov = np.linalg.pinv(H)
        if not np.all(np.isfinite(cov)):
            raise np.linalg.LinAlgError("non-finite covariance")
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return theta_map, err, cov, True, "laplace_ok", np.nan
    except Exception as e:
        err = np.full_like(theta_map, np.nan, dtype=float)
        cov = np.full((len(theta_map), len(theta_map)), np.nan, dtype=float)
        return theta_map, err, cov, False, f"laplace_failed:{e}", np.nan


def posterior_from_emcee(theta_map: np.ndarray, phi2: np.ndarray, template: TemplateBundle, mu_prior_center: float, mu_prior_sigma: float, nwalkers: int, burn: int, steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float]:
    if not HAVE_EMCEE:
        return posterior_from_map_only(theta_map, phi2, template, mu_prior_center, mu_prior_sigma)

    ndim = len(theta_map)
    scales = np.array([0.15, 0.03, 0.12, 0.20], dtype=float)
    p0 = []
    logp0 = []
    for _ in range(max(5000, nwalkers * 50)):
        trial = theta_map + rng.normal(scale=scales, size=ndim)
        trial[0] = np.clip(trial[0], UF_MIN, UF_MAX)
        trial[1] = np.clip(trial[1], template.fit_lo, template.fit_hi)
        trial[2] = np.clip(trial[2], np.log(SIGMA_MIN), np.log(SIGMA_MAX))
        trial[3] = np.clip(trial[3], RAW_TILT_MIN, RAW_TILT_MAX)
        lp = log_posterior(trial, phi2, template, mu_prior_center, mu_prior_sigma)
        if np.isfinite(lp):
            p0.append(trial)
            logp0.append(lp)
        if len(p0) >= nwalkers:
            break
    if len(p0) < nwalkers:
        return posterior_from_map_only(theta_map, phi2, template, mu_prior_center, mu_prior_sigma)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=(phi2, template, mu_prior_center, mu_prior_sigma),
    )
    try:
        pos, _, _ = sampler.run_mcmc(np.asarray(p0), burn, progress=False)
        sampler.reset()
        sampler.run_mcmc(pos, steps, progress=False)
    except Exception as e:
        return posterior_from_map_only(theta_map, phi2, template, mu_prior_center, mu_prior_sigma)

    chain = sampler.get_chain(flat=True)
    if chain.size == 0 or not np.all(np.isfinite(chain)):
        return posterior_from_map_only(theta_map, phi2, template, mu_prior_center, mu_prior_sigma)

    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    theta_med = q50
    theta_err = 0.5 * (q84 - q16)
    cov = np.cov(chain.T)
    acc = float(np.mean(sampler.acceptance_fraction))
    sampler_ok = np.isfinite(acc) and (0.05 <= acc <= 0.80)
    return theta_med, theta_err, cov, bool(sampler_ok), "emcee_ok" if sampler_ok else "emcee_bad_acceptance", acc


def unpack_theta(theta: np.ndarray, theta_err: np.ndarray, phi2: np.ndarray, template: TemplateBundle) -> Dict[str, float]:
    u_f, mu, log_sigma, raw_tilt = [float(x) for x in theta]
    ef, emu, elogsig, erawt = [float(x) if np.isfinite(x) else np.nan for x in theta_err]

    f_stream = float(sigmoid(u_f))
    if np.isfinite(ef):
        f_stream_err = abs(f_stream * (1.0 - f_stream) * ef)
    else:
        f_stream_err = np.nan

    sigma = float(np.exp(log_sigma))
    sigma_err = abs(sigma * elogsig) if np.isfinite(elogsig) else np.nan
    bg_tilt = transform_bg_tilt(raw_tilt, template.fit_lo, template.fit_hi)
    bg_tilt_err = np.nan
    if np.isfinite(erawt):
        half = 0.5 * (template.fit_hi - template.fit_lo)
        deriv = 0.95 * (1.0 - np.tanh(raw_tilt) ** 2) / max(half, 1e-6)
        bg_tilt_err = abs(deriv) * erawt

    p_s, p_b, _, _, _, _ = component_pdfs_observed(phi2, theta, template)
    p = f_stream * p_s + (1.0 - f_stream) * p_b
    if not np.all(np.isfinite(p)):
        n_stream = np.nan
        n_stream_err = np.nan
    else:
        n_stream = f_stream * len(phi2)
        n_stream_err = f_stream_err * len(phi2) if np.isfinite(f_stream_err) else np.nan

    linear_density = n_stream / WINDOW_WIDTH_PHI1 if np.isfinite(n_stream) else np.nan
    linear_density_err = n_stream_err / WINDOW_WIDTH_PHI1 if np.isfinite(n_stream_err) else np.nan

    peak = np.nan
    peak_err = np.nan
    if np.isfinite(n_stream) and np.isfinite(sigma) and sigma > 0:
        peak = n_stream / (WINDOW_WIDTH_PHI1 * np.sqrt(2.0 * np.pi) * sigma)
        if np.isfinite(n_stream_err) and np.isfinite(sigma_err):
            rel2 = 0.0
            if n_stream > 0:
                rel2 += (n_stream_err / n_stream) ** 2
            if sigma > 0:
                rel2 += (sigma_err / sigma) ** 2
            peak_err = abs(peak) * np.sqrt(rel2)

    return {
        "f_stream": f_stream,
        "f_stream_err": f_stream_err,
        "mu": mu,
        "mu_err": emu,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "bg_tilt": bg_tilt,
        "bg_tilt_err": bg_tilt_err,
        "n_stream": n_stream,
        "n_stream_err": n_stream_err,
        "linear_density": linear_density,
        "linear_density_err": linear_density_err,
        "peak_surface_density": peak,
        "peak_surface_density_err": peak_err,
    }


# -----------------------------------------------------------------------------
# Main fitting loop
# -----------------------------------------------------------------------------

def load_mu_prior(centers: np.ndarray, path: str, signal_phi1: np.ndarray, signal_phi2: np.ndarray) -> np.ndarray:
    if path and os.path.exists(path):
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] >= 3:
            xp = arr[:, 0]
            fp = arr[:, 2]
        else:
            xp = arr[:, 0]
            fp = arr[:, 1]
        mu_prior = np.interp(centers, xp, fp)
        return mu_prior.astype(float)

    # Fallback: broad histogram-mode ridge.
    half_w = 0.5 * WINDOW_WIDTH_PHI1
    mu0 = np.full_like(centers, np.nan, dtype=float)
    for i, c in enumerate(centers):
        m = (signal_phi1 >= c - half_w) & (signal_phi1 < c + half_w) & (signal_phi2 >= PASS0_PHI2_MIN) & (signal_phi2 <= PASS0_PHI2_MAX)
        mu0[i] = histogram_mode(signal_phi2[m], PASS0_PHI2_MIN, PASS0_PHI2_MAX, ETA_BIN)
    return smooth_track_by_arm(centers, mu0)


def fit_all_bins(
    signal: Table,
    parent_views: Dict[str, np.ndarray],
    centers: np.ndarray,
    mu_prior: np.ndarray,
    eta_mode: str,
    sampler_mode: str,
    nwalkers: int,
    burn: int,
    steps: int,
    rng: np.random.Generator,
) -> Tuple[List[BinResult], Dict[float, TemplateBundle]]:
    sig_phi1 = np.asarray(signal["PHI1"], dtype=float)
    sig_phi2 = np.asarray(signal["PHI2"], dtype=float)

    ctl_phi1 = parent_views["ctl_phi1"]
    ctl_phi2 = parent_views["ctl_phi2"]
    zl_phi1 = parent_views["zl_phi1"]
    zl_phi2 = parent_views["zl_phi2"]
    zl_gd = parent_views["zl_gd"]
    zl_zd = parent_views["zl_zd"]

    half_w = 0.5 * WINDOW_WIDTH_PHI1
    results: List[BinResult] = []
    template_cache: Dict[float, TemplateBundle] = {}

    for i, c in enumerate(centers):
        phi1_lo = c - half_w
        phi1_hi = c + half_w
        is_cluster_bin = abs(c) < CLUSTER_MASK_HALFWIDTH

        mu0 = float(mu_prior[i])
        fit_lo = mu0 - FIT_HALFWIDTH
        fit_hi = mu0 + FIT_HALFWIDTH

        m_sig = (sig_phi1 >= phi1_lo) & (sig_phi1 < phi1_hi) & (sig_phi2 >= fit_lo) & (sig_phi2 <= fit_hi)
        y = np.asarray(sig_phi2[m_sig], dtype=float)

        m_ctl = (ctl_phi1 >= phi1_lo) & (ctl_phi1 < phi1_hi) & (ctl_phi2 >= fit_lo) & (ctl_phi2 <= fit_hi)
        y_ctl = np.asarray(ctl_phi2[m_ctl], dtype=float)

        m_zl = (zl_phi1 >= phi1_lo) & (zl_phi1 < phi1_hi) & (zl_phi2 >= fit_lo) & (zl_phi2 <= fit_hi)
        y_zl = np.asarray(zl_phi2[m_zl], dtype=float)
        gd_zl = np.asarray(zl_gd[m_zl], dtype=float)
        zd_zl = np.asarray(zl_zd[m_zl], dtype=float)

        template = build_eta_template(y_ctl, y_zl, gd_zl, zd_zl, fit_lo, fit_hi, eta_mode)
        template_cache[c] = template

        row = BinResult(
            phi1_center=float(c),
            phi1_lo=float(phi1_lo),
            phi1_hi=float(phi1_hi),
            fit_lo=float(fit_lo),
            fit_hi=float(fit_hi),
            n_signal=int(len(y)),
            n_control=int(template.n_control),
            n_depth=int(template.n_depth),
            cluster_bin=bool(is_cluster_bin),
            success=False,
            sampler_used="map" if sampler_mode == "map" else ("emcee" if HAVE_EMCEE else "map"),
            optimizer_success=False,
            sampler_success=False,
            message="not_fitted",
            mu_prior_center=float(mu0),
            mu_prior_sigma=float(MU_PRIOR_SIGMA),
        )

        if len(y) < MIN_SIGNAL_STARS:
            row.message = "too_few_signal_stars"
            results.append(row)
            continue

        theta_map, nlp_map, opt_ok, opt_msg = fit_single_bin_map(y, template, mu0, MU_PRIOR_SIGMA)
        row.optimizer_success = bool(opt_ok)
        row.map_neglogpost = float(nlp_map) if nlp_map is not None else np.nan
        if theta_map is None:
            row.message = opt_msg
            results.append(row)
            continue

        use_emcee = (sampler_mode == "emcee") or (sampler_mode == "auto" and HAVE_EMCEE)
        if use_emcee:
            theta_post, theta_err, cov, sampler_ok, smsg, acc = posterior_from_emcee(theta_map, y, template, mu0, MU_PRIOR_SIGMA, nwalkers, burn, steps, rng)
        else:
            theta_post, theta_err, cov, sampler_ok, smsg, acc = posterior_from_map_only(theta_map, y, template, mu0, MU_PRIOR_SIGMA)

        stats = unpack_theta(theta_post, theta_err, y, template)
        for k, v in stats.items():
            setattr(row, k, v)
        row.acc_frac = float(acc) if np.isfinite(acc) else np.nan
        row.sampler_success = bool(sampler_ok)

        # Final success criteria: actual convergence and not pinned to hard bounds.
        bounds_ok = (
            np.isfinite(row.mu)
            and np.isfinite(row.sigma)
            and (fit_lo <= row.mu <= fit_hi)
            and (SIGMA_MIN < row.sigma < SIGMA_MAX)
            and np.isfinite(row.f_stream)
            and (0.0 < row.f_stream < 1.0)
        )
        err_ok = (
            np.isfinite(row.mu_err)
            and np.isfinite(row.sigma_err)
            and (row.mu_err < 0.5)
            and (row.sigma_err < 0.6)
        )
        row.success = bool(opt_ok and sampler_ok and bounds_ok and err_ok)
        row.message = smsg if row.success else f"{opt_msg};{smsg}"
        results.append(row)

        if (i % 10) == 0 or (i == len(centers) - 1):
            print(
                f"[fit] {i+1:02d}/{len(centers)} phi1={c:+5.2f} "
                f"Nsig={len(y):4d} Nctl={len(y_ctl):5d} success={row.success}"
            )

    return results, template_cache


# -----------------------------------------------------------------------------
# Tables / summaries / plotting
# -----------------------------------------------------------------------------

def results_to_table(results: List[BinResult]) -> Table:
    cols: Dict[str, List] = {}
    for key in BinResult.__dataclass_fields__.keys():
        cols[key] = [getattr(r, key) for r in results]
    return Table(cols)


def polyfit_arm(phi1: np.ndarray, mu: np.ndarray, mu_err: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    m = mask & np.isfinite(phi1) & np.isfinite(mu)
    if np.sum(m) < 4:
        return None
    x = np.asarray(phi1[m], dtype=float)
    y = np.asarray(mu[m], dtype=float)
    err = np.asarray(mu_err[m], dtype=float)
    use = np.ones_like(x, dtype=bool)

    for _ in range(3):
        if np.sum(use) < 4:
            break
        if np.isfinite(err[use]).sum() >= 3:
            w = 1.0 / np.clip(err[use], 0.03, None)
        else:
            w = None
        try:
            coeff = np.polyfit(x[use], y[use], deg=2, w=w)
        except Exception:
            return None

        resid = y - np.polyval(coeff, x)
        clip = max(0.35, 4.0 * mad_scale(resid[use], floor=0.05))
        new_use = np.abs(resid) <= clip
        # If clipping would leave too few bins, keep the previous fit.
        if np.sum(new_use) < 4 or np.all(new_use == use):
            return coeff
        use = new_use

    if np.sum(use) < 4:
        return None
    if np.isfinite(err[use]).sum() >= 3:
        w = 1.0 / np.clip(err[use], 0.03, None)
    else:
        w = None
    try:
        return np.polyfit(x[use], y[use], deg=2, w=w)
    except Exception:
        return None


def append_track_polynomials(tab: Table) -> Dict[str, Optional[np.ndarray]]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    mu = np.asarray(tab["mu"], dtype=float)
    mu_err = np.asarray(tab["mu_err"], dtype=float)
    success = np.asarray(tab["fit_success"] if "fit_success" in tab.colnames else tab["success"], dtype=bool)
    cluster = np.asarray(tab["cluster_bin"], dtype=bool)

    trail_mask = success & ~cluster & (phi1 <= -0.75)
    lead_mask = success & ~cluster & (phi1 >= 0.75)
    coeff_tr = polyfit_arm(phi1, mu, mu_err, trail_mask)
    coeff_ld = polyfit_arm(phi1, mu, mu_err, lead_mask)

    poly = np.full(len(tab), np.nan, dtype=float)
    if coeff_tr is not None:
        m = phi1 <= 0.0
        poly[m] = np.polyval(coeff_tr, phi1[m])
    if coeff_ld is not None:
        m = phi1 >= 0.0
        poly[m] = np.polyval(coeff_ld, phi1[m])
    resid = mu - poly
    resid[~np.isfinite(resid)] = np.nan
    tab["track_poly"] = poly
    tab["track_resid"] = resid
    return {"trailing": coeff_tr, "leading": coeff_ld}


def append_clean_track_columns(tab: Table) -> Dict[str, int]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    mu = np.asarray(tab["mu"], dtype=float)
    mu_err = np.asarray(tab["mu_err"], dtype=float)
    sigma = np.asarray(tab["sigma"], dtype=float)
    sigma_err = np.asarray(tab["sigma_err"], dtype=float)
    fit_success = np.asarray(tab["success"], dtype=bool)
    cluster = np.asarray(tab["cluster_bin"], dtype=bool)
    track_poly = np.asarray(tab["track_poly"], dtype=float)
    track_resid = np.asarray(tab["track_resid"], dtype=float)

    sigma_smooth = smooth_profile_by_arm(phi1, np.where(fit_success & ~cluster, sigma, np.nan))
    sigma_log = np.full(len(tab), np.nan, dtype=float)
    sigma_log_err = np.full(len(tab), np.nan, dtype=float)
    m_sig = fit_success & ~cluster & np.isfinite(sigma) & (sigma > 0)
    sigma_log[m_sig] = np.log(sigma[m_sig])
    sigma_log_err[m_sig] = np.where(
        np.isfinite(sigma_err[m_sig]) & (sigma_err[m_sig] > 0),
        sigma_err[m_sig] / sigma[m_sig],
        0.20,
    )
    coeff_sig_tr = polyfit_arm(phi1, sigma_log, sigma_log_err, m_sig & (phi1 <= -0.75))
    coeff_sig_ld = polyfit_arm(phi1, sigma_log, sigma_log_err, m_sig & (phi1 >= 0.75))
    sigma_poly = np.full(len(tab), np.nan, dtype=float)
    if coeff_sig_tr is not None:
        m = phi1 <= 0.0
        sigma_poly[m] = np.exp(np.polyval(coeff_sig_tr, phi1[m]))
    if coeff_sig_ld is not None:
        m = phi1 >= 0.0
        sigma_poly[m] = np.exp(np.polyval(coeff_sig_ld, phi1[m]))
    m_poly = np.isfinite(sigma_poly)
    sigma_smooth[m_poly] = sigma_poly[m_poly]
    sigma_resid = sigma - sigma_smooth

    track_coherent = np.ones(len(tab), dtype=bool)
    m_track = fit_success & ~cluster & np.isfinite(track_poly) & np.isfinite(track_resid)
    track_coherent[m_track] = np.abs(track_resid[m_track]) <= TRACK_COHERENCE_ABS

    width_coherent = np.ones(len(tab), dtype=bool)
    m_width = fit_success & ~cluster & np.isfinite(sigma) & np.isfinite(sigma_smooth) & (sigma_smooth > 0)
    ratio = np.full(len(tab), np.nan, dtype=float)
    ratio[m_width] = sigma[m_width] / sigma_smooth[m_width]
    width_coherent[m_width] = (
        (ratio[m_width] >= SIGMA_COHERENCE_MIN_RATIO)
        & (ratio[m_width] <= SIGMA_COHERENCE_MAX_RATIO)
    )

    final_success = fit_success.copy()
    final_success[~cluster] = fit_success[~cluster] & track_coherent[~cluster] & width_coherent[~cluster]

    mu_clean = np.array(mu, dtype=float, copy=True)
    m_replace_mu = (~cluster) & np.isfinite(track_poly) & ~final_success
    mu_clean[m_replace_mu] = track_poly[m_replace_mu]

    mu_clean_err = np.array(mu_err, dtype=float, copy=True)
    bad_mu_err = ~np.isfinite(mu_clean_err)
    mu_clean_err[bad_mu_err] = MU_CLEAN_ERR_FLOOR
    m_incoh_mu = (~cluster) & np.isfinite(mu_clean)
    mu_clean_err[m_incoh_mu] = np.maximum(
        mu_clean_err[m_incoh_mu],
        np.maximum(np.abs(track_resid[m_incoh_mu]), MU_CLEAN_ERR_FLOOR),
    )

    sigma_clean = np.array(sigma, dtype=float, copy=True)
    m_replace_sigma = (~cluster) & np.isfinite(sigma_smooth) & (~final_success | ~width_coherent)
    sigma_clean[m_replace_sigma] = sigma_smooth[m_replace_sigma]

    sigma_clean_err = np.array(sigma_err, dtype=float, copy=True)
    bad_sig_err = ~np.isfinite(sigma_clean_err)
    sigma_clean_err[bad_sig_err] = SIGMA_CLEAN_ERR_FLOOR
    m_incoh_sig = (~cluster) & np.isfinite(sigma_clean)
    sigma_clean_err[m_incoh_sig] = np.maximum(
        sigma_clean_err[m_incoh_sig],
        np.maximum(np.abs(sigma_resid[m_incoh_sig]), SIGMA_CLEAN_ERR_FLOOR),
    )

    source = np.full(len(tab), "raw_fit", dtype="U24")
    source[cluster] = "cluster"
    source[(~cluster) & fit_success & ~track_coherent] = "smoothed_track"
    source[(~cluster) & fit_success & track_coherent & ~width_coherent] = "smoothed_width"
    source[(~cluster) & fit_success & ~final_success & ~width_coherent & ~track_coherent] = "smoothed_track_width"
    source[(~cluster) & ~fit_success & np.isfinite(track_poly)] = "smoothed_from_failed"
    source[(~cluster) & ~fit_success & ~np.isfinite(track_poly)] = "failed"

    clean_use = (~cluster) & np.isfinite(mu_clean) & np.isfinite(mu_clean_err)

    tab["fit_success"] = fit_success
    tab["track_coherent"] = track_coherent
    tab["width_coherent"] = width_coherent
    tab["sigma_smooth"] = sigma_smooth
    tab["sigma_poly"] = sigma_poly
    tab["sigma_resid"] = sigma_resid
    tab["sigma_ratio_to_smooth"] = ratio
    tab["mu_clean"] = mu_clean
    tab["mu_clean_err"] = mu_clean_err
    tab["sigma_clean"] = sigma_clean
    tab["sigma_clean_err"] = sigma_clean_err
    tab["clean_source"] = np.asarray(source, dtype="U24")
    tab["clean_use"] = clean_use
    tab["success"] = final_success

    return {
        "n_fit_success": int(np.sum(fit_success)),
        "n_final_success": int(np.sum(final_success)),
        "n_track_incoherent": int(np.sum(fit_success & ~cluster & ~track_coherent)),
        "n_width_incoherent": int(np.sum(fit_success & ~cluster & ~width_coherent)),
        "n_clean_use": int(np.sum(clean_use)),
    }


def write_mockfit_track_table(tab: Table, out_csv: Path, out_fits: Path) -> int:
    clean_use = np.asarray(tab["clean_use"], dtype=bool)
    sub = tab[clean_use]

    out = Table()
    out["phi1"] = np.asarray(sub["phi1_center"], dtype=float)
    out["phi2"] = np.asarray(sub["mu_clean"], dtype=float)
    out["phi2_err"] = np.asarray(sub["mu_clean_err"], dtype=float)
    out["width"] = np.asarray(sub["sigma_clean"], dtype=float)
    out["width_err"] = np.asarray(sub["sigma_clean_err"], dtype=float)
    out["success"] = np.ones(len(sub), dtype=bool)
    out["source"] = np.asarray(sub["clean_source"], dtype="U24")
    out["phi2_raw"] = np.asarray(sub["mu"], dtype=float)
    out["width_raw"] = np.asarray(sub["sigma"], dtype=float)
    out["track_poly"] = np.asarray(sub["track_poly"], dtype=float)
    out["sigma_smooth"] = np.asarray(sub["sigma_smooth"], dtype=float)
    out.write(out_fits, overwrite=True)
    out.write(out_csv, format="ascii.csv", overwrite=True)
    return len(out)


def summarize_results(tab: Table, poly_coeffs: Dict[str, Optional[np.ndarray]], meta: Dict[str, object]) -> Dict[str, object]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    success = np.asarray(tab["success"], dtype=bool)
    fit_success = np.asarray(tab["fit_success"], dtype=bool) if "fit_success" in tab.colnames else success
    cluster = np.asarray(tab["cluster_bin"], dtype=bool)
    sigma = np.asarray(tab["sigma_clean"] if "sigma_clean" in tab.colnames else tab["sigma"], dtype=float)
    lin = np.asarray(tab["linear_density"], dtype=float)

    m_lead = success & ~cluster & (phi1 > 0.0) & np.isfinite(sigma)
    m_trail = success & ~cluster & (phi1 < 0.0) & np.isfinite(sigma)
    m_dens = success & ~cluster & np.isfinite(lin)

    out = dict(meta)
    out.update({
        "n_bins": int(len(tab)),
        "n_fit_success": int(np.sum(fit_success)),
        "n_success": int(np.sum(success)),
        "n_success_excluding_cluster": int(np.sum(success & ~cluster)),
        "track_poly_trailing": poly_coeffs["trailing"].tolist() if poly_coeffs["trailing"] is not None else None,
        "track_poly_leading": poly_coeffs["leading"].tolist() if poly_coeffs["leading"] is not None else None,
        "max_width_leading": float(np.nanmax(sigma[m_lead])) if np.any(m_lead) else None,
        "max_width_trailing": float(np.nanmax(sigma[m_trail])) if np.any(m_trail) else None,
        "integrated_stream_stars_excluding_cluster": float(np.nansum(lin[m_dens] * WINDOW_WIDTH_PHI1)) if np.any(m_dens) else None,
    })
    if "track_coherent" in tab.colnames:
        out["n_track_incoherent"] = int(np.sum(fit_success & ~cluster & ~np.asarray(tab["track_coherent"], dtype=bool)))
    if "width_coherent" in tab.colnames:
        out["n_width_incoherent"] = int(np.sum(fit_success & ~cluster & ~np.asarray(tab["width_coherent"], dtype=bool)))
    if "clean_use" in tab.colnames:
        out["n_mockfit_track_nodes"] = int(np.sum(np.asarray(tab["clean_use"], dtype=bool)))
    return out


def save_density_map_with_track(tab_members: Table, tab_fit: Table, out_png: Path, full_frame: bool = True) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)

    if full_frame:
        xlo, xhi = FULL_PHI1_MIN, FULL_PHI1_MAX
        ylo, yhi = FULL_PHI2_MIN, FULL_PHI2_MAX
        title = "strict selected sample: Pal 5 frame number density"
    else:
        xlo, xhi = PHI1_MIN, PHI1_MAX
        ylo, yhi = PASS0_PHI2_MIN, PASS0_PHI2_MAX
        title = "strict selected sample: local Pal 5 frame density"

    xedges = np.arange(xlo, xhi + MAP_BIN, MAP_BIN)
    yedges = np.arange(ylo, yhi + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(phi1, phi2, bins=[xedges, yedges])
    D = H.T / (MAP_BIN * MAP_BIN)

    plt.figure(figsize=(11, 6.5))
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest")
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")

    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    mu = np.asarray(tab_fit["mu_clean"] if "mu_clean" in tab_fit.colnames else tab_fit["mu"], dtype=float)
    sig = np.asarray(tab_fit["sigma_clean"] if "sigma_clean" in tab_fit.colnames else tab_fit["sigma"], dtype=float)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool) if "cluster_bin" in tab_fit.colnames else np.zeros(len(tab_fit), dtype=bool)
    track_poly = np.asarray(tab_fit["track_poly"], dtype=float) if "track_poly" in tab_fit.colnames else np.full(len(tab_fit), np.nan)
    # Keep the density-map overlay as close as possible to the raw local-fit track,
    # and only fall back to the arm-wise quadratic when a bin is an obvious outlier.
    # This preserves the cluster-region shape while still suppressing pathological
    # local-fit failures like the right-arm dip near phi1 ~ 7.75.
    resid = np.asarray(tab_fit["track_resid"], dtype=float) if "track_resid" in tab_fit.colnames else (mu - track_poly)
    bad_raw = (~cluster) & np.isfinite(track_poly) & np.isfinite(resid) & (np.abs(resid) > TRACK_PLOT_OUTLIER_ABS)
    mu_plot = np.where(bad_raw, track_poly, mu)
    ok = success & np.isfinite(mu_plot) & np.isfinite(sig)
    if np.any(ok):
        plt.plot(x[ok], mu_plot[ok], lw=2.0, color="C0")
        plt.plot(x[ok], mu_plot[ok] + sig[ok], "--", lw=1.2, color="C1")
        plt.plot(x[ok], mu_plot[ok] - sig[ok], "--", lw=1.2, color="C2")

    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_control_density_map(parent_views: Dict[str, np.ndarray], out_png: Path) -> None:
    x = np.asarray(parent_views["ctl_phi1"], dtype=float)
    y = np.asarray(parent_views["ctl_phi2"], dtype=float)
    xedges = np.arange(FULL_PHI1_MIN, FULL_PHI1_MAX + MAP_BIN, MAP_BIN)
    yedges = np.arange(FULL_PHI2_MIN, FULL_PHI2_MAX + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges])
    D = H.T / (MAP_BIN * MAP_BIN)
    plt.figure(figsize=(11, 6.5))
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest")
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step 3b control sample (strict mag + z-locus + isochrone sidebands)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_radec_map(tab_members: Table, out_png: Path) -> None:
    ra = np.asarray(tab_members["RA"], dtype=float)
    dec = np.asarray(tab_members["DEC"], dtype=float)
    xedges = np.arange(np.floor(ra.min()), np.ceil(ra.max()) + MAP_BIN, MAP_BIN)
    yedges = np.arange(np.floor(dec.min()), np.ceil(dec.max()) + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(ra, dec, bins=[xedges, yedges])
    D = H.T / (MAP_BIN * MAP_BIN)
    plt.figure(figsize=(10.5, 6.5))
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest")
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("strict selected sample: RA-Dec number density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_track_plot(tab_fit: Table, poly_coeffs: Dict[str, Optional[np.ndarray]], out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    mu = np.asarray(tab_fit["mu_clean"] if "mu_clean" in tab_fit.colnames else tab_fit["mu"], dtype=float)
    mu_err = np.asarray(tab_fit["mu_clean_err"] if "mu_clean_err" in tab_fit.colnames else tab_fit["mu_err"], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool)

    plt.figure(figsize=(10, 4.8))
    ok = success & np.isfinite(mu)
    plt.errorbar(x[ok], mu[ok], yerr=mu_err[ok], fmt="o", ms=3.5, lw=1.0, capsize=2, label="fit")
    if np.any(cluster & np.isfinite(mu)):
        m = cluster & np.isfinite(mu)
        plt.errorbar(x[m], mu[m], yerr=mu_err[m], fmt="s", ms=4.0, lw=1.0, capsize=2, color="0.5", label="cluster bins")
    for arm, coeff in poly_coeffs.items():
        if coeff is None:
            continue
        xx = np.linspace(PHI1_MIN, 0.0, 200) if arm == "trailing" else np.linspace(0.0, PHI1_MAX, 200)
        yy = np.polyval(coeff, xx)
        plt.plot(xx, yy, "--", lw=1.5, label=f"{arm} quadratic")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"track $\mu(\phi_1)$ [deg]")
    plt.title("Selection-aware Bonaca-style 1D model: stream track")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_track_resid_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    r = np.asarray(tab_fit["track_resid"], dtype=float)
    rerr = np.asarray(tab_fit["mu_clean_err"] if "mu_clean_err" in tab_fit.colnames else tab_fit["mu_err"], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool)
    m = success & ~cluster & np.isfinite(r)
    plt.figure(figsize=(10, 4.2))
    plt.axhline(0.0, color="0.5", lw=1.0)
    plt.errorbar(x[m], r[m], yerr=rerr[m], fmt="o", ms=3.5, lw=1.0, capsize=2)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("track residual [deg]")
    plt.title("Selection-aware Bonaca-style 1D model: track residual")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_profile_plot(tab_fit: Table, ycol: str, yerrcol: str, ylabel: str, title: str, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    use_ycol = ycol
    use_yerrcol = yerrcol
    if ycol == "sigma" and "sigma_clean" in tab_fit.colnames:
        use_ycol = "sigma_clean"
    if yerrcol == "sigma_err" and "sigma_clean_err" in tab_fit.colnames:
        use_yerrcol = "sigma_clean_err"
    y = np.asarray(tab_fit[use_ycol], dtype=float)
    yerr = np.asarray(tab_fit[use_yerrcol], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool)

    plt.figure(figsize=(10, 4.2))
    m = success & ~cluster & np.isfinite(y)
    plt.errorbar(x[m], y[m], yerr=yerr[m], fmt="o", ms=3.5, lw=1.0, capsize=2)
    if np.any(cluster & np.isfinite(y)):
        mc = cluster & np.isfinite(y)
        plt.errorbar(x[mc], y[mc], yerr=yerr[mc], fmt="s", ms=4.0, lw=1.0, capsize=2, color="0.5")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_example_fits(tab_members: Table, tab_fit: Table, template_cache: Dict[float, TemplateBundle], out_png: Path) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)
    half_w = 0.5 * WINDOW_WIDTH_PHI1

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    fit_centers = np.asarray(tab_fit["phi1_center"], dtype=float)

    for ax, target in zip(axes, EXAMPLE_BIN_CENTERS):
        idx = int(np.argmin(np.abs(fit_centers - target)))
        row = tab_fit[idx]
        c = float(row["phi1_center"])
        fit_lo = float(row["fit_lo"])
        fit_hi = float(row["fit_hi"])
        m = (phi1 >= c - half_w) & (phi1 < c + half_w) & (phi2 >= fit_lo) & (phi2 <= fit_hi)
        y = np.asarray(phi2[m], dtype=float)
        bins = np.arange(fit_lo, fit_hi + ETA_BIN, ETA_BIN)
        hist, edges = np.histogram(y, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax.step(mids, hist, where="mid", lw=1.2, label="data")

        tmpl = template_cache.get(c)
        if tmpl is not None:
            eta = safe_interp(mids, tmpl.grid, tmpl.eta_total, fill=1.0)
            eta_scaled = eta / np.nanmax(eta) * max(np.max(hist), 1)
            ax.plot(mids, eta_scaled, color="0.6", ls=":", lw=1.5, label="eta (scaled)")

        if bool(row["success"]):
            theta = np.array([
                logit(float(row["f_stream"])),
                float(row["mu"]),
                np.log(float(row["sigma"])),
                np.arctanh(np.clip(float(row["bg_tilt"]) * (0.5 * (fit_hi - fit_lo)) / 0.95, -0.999, 0.999)),
            ])
            p_s, p_b, f_stream, _, _, _ = component_pdfs_observed(mids, theta, tmpl)
            model_counts = len(y) * (f_stream * p_s + (1.0 - f_stream) * p_b) * (edges[1] - edges[0])
            ax.plot(mids, model_counts, lw=2.0, label="fit")
            ax.axvline(float(row["mu"]), lw=1.0)

        ax.set_title(rf"$\phi_1 \approx {c:+.2f}^\circ$")
        ax.set_xlabel(r"$\phi_2$ [deg]")
        if ax in axes[::3]:
            ax.set_ylabel("counts / bin")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(r"Selection-aware local fits in overlapping $\phi_1$ bins")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_eta_examples(template_cache: Dict[float, TemplateBundle], centers: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    for ax, target in zip(axes, EXAMPLE_BIN_CENTERS):
        c = float(centers[int(np.argmin(np.abs(centers - target)))])
        tmpl = template_cache[c]
        ax.plot(tmpl.grid, tmpl.eta_control, label="control eta")
        ax.plot(tmpl.grid, tmpl.eta_depth, label="depth eta")
        ax.plot(tmpl.grid, tmpl.eta_total, lw=2.0, label="total eta")
        ax.set_title(rf"$\phi_1 \approx {c:+.2f}^\circ$")
        ax.set_xlabel(r"$\phi_2$ [deg]")
        if ax in axes[::3]:
            ax.set_ylabel(r"$\eta(\phi_2)$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Selection templates used in step 3b")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def choose_default_mu_prior_file(user_value: str) -> str:
    if user_value:
        return user_value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 step 3b: selection-aware Bonaca-style 1D model")
    p.add_argument("--signal", default=DEFAULT_SIGNAL, help="Step 2 strict member FITS catalog")
    p.add_argument("--preproc", default=DEFAULT_PREPROC, help="Preprocessed parent FITS catalog")
    p.add_argument("--step2-summary", default=DEFAULT_STEP2_SUMMARY, help="Step 2 summary JSON")
    p.add_argument("--iso", default=DEFAULT_ISO, help="Isochrone file used in step 2")
    p.add_argument("--mu-prior-file", default=DEFAULT_MU_PRIOR, help="Optional pass1/smoothed prior track text file")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument("--eta-mode", choices=["control", "depth", "control_times_depth"], default="control", help="Selection template mode")
    p.add_argument("--sampler", choices=["auto", "emcee", "map"], default=DEFAULT_SAMPLER, help="Posterior summarization mode")
    p.add_argument("--nwalkers", type=int, default=DEFAULT_NWALKERS)
    p.add_argument("--burn", type=int, default=DEFAULT_BURN)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--chunk-size", type=int, default=2_000_000, help="Chunk size for scanning the parent catalog")
    p.add_argument("--phi1-min", type=float, default=PHI1_MIN)
    p.add_argument("--phi1-max", type=float, default=PHI1_MAX)
    p.add_argument("--phi1-step", type=float, default=PHI1_STEP)
    p.add_argument("--fit-halfwidth", type=float, default=FIT_HALFWIDTH)
    p.add_argument("--min-signal", type=int, default=MIN_SIGNAL_STARS)
    return p.parse_args()


def main() -> None:
    global PHI1_MIN, PHI1_MAX, PHI1_STEP, WINDOW_WIDTH_PHI1, FIT_HALFWIDTH, MIN_SIGNAL_STARS

    args = parse_args()
    PHI1_MIN = args.phi1_min
    PHI1_MAX = args.phi1_max
    PHI1_STEP = args.phi1_step
    WINDOW_WIDTH_PHI1 = WINDOW_SCALE * PHI1_STEP
    FIT_HALFWIDTH = args.fit_halfwidth
    MIN_SIGNAL_STARS = args.min_signal

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    rng = np.random.default_rng(RNG_SEED)

    mu_prior_file = choose_default_mu_prior_file(args.mu_prior_file)
    if mu_prior_file:
        print(f"[info] using mu prior file: {mu_prior_file}")
    else:
        print("[info] no external mu prior file found; will build a fallback prior from the signal map")

    print(f"[read signal] {args.signal}")
    signal = Table.read(args.signal)
    for col in ("PHI1", "PHI2", "RA", "DEC"):
        if col not in signal.colnames:
            raise KeyError(f"Missing required column in signal FITS: {col}")
    m_fin = np.isfinite(np.asarray(signal["PHI1"], dtype=float)) & np.isfinite(np.asarray(signal["PHI2"], dtype=float))
    signal = signal[m_fin]
    print(f"[info] signal rows={len(signal):,}")

    models = load_step2_models(args.step2_summary, args.iso)
    parent_views = build_control_and_depth_views(args.preproc, models, outdir, chunk_size=args.chunk_size)

    centers = np.arange(PHI1_MIN, PHI1_MAX + 0.5 * PHI1_STEP, PHI1_STEP)
    mu_prior = load_mu_prior(
        centers,
        mu_prior_file,
        np.asarray(signal["PHI1"], dtype=float),
        np.asarray(signal["PHI2"], dtype=float),
    )
    np.savetxt(outdir / "pal5_step3b_mu_prior.txt", np.c_[centers, mu_prior], header="phi1_center  mu_prior")

    results, template_cache = fit_all_bins(
        signal=signal,
        parent_views=parent_views,
        centers=centers,
        mu_prior=mu_prior,
        eta_mode=args.eta_mode,
        sampler_mode=args.sampler,
        nwalkers=args.nwalkers,
        burn=args.burn,
        steps=args.steps,
        rng=rng,
    )

    tab = results_to_table(results)
    poly_coeffs = append_track_polynomials(tab)
    clean_stats = append_clean_track_columns(tab)
    fits_path = outdir / "pal5_step3b_profiles.fits"
    csv_path = outdir / "pal5_step3b_profiles.csv"
    tab.write(fits_path, overwrite=True)
    tab.write(csv_path, format="ascii.csv", overwrite=True)
    print(f"[write] {fits_path}")
    print(f"[write] {csv_path}")

    mockfit_csv_path = outdir / "pal5_step3b_mockfit_track.csv"
    mockfit_fits_path = outdir / "pal5_step3b_mockfit_track.fits"
    n_mockfit = write_mockfit_track_table(tab, mockfit_csv_path, mockfit_fits_path)
    print(f"[write] {mockfit_fits_path}")
    print(f"[write] {mockfit_csv_path}")

    summary = summarize_results(
        tab,
        poly_coeffs,
        {
            "signal": args.signal,
            "preproc": args.preproc,
            "step2_summary": args.step2_summary,
            "iso": args.iso,
            "mu_prior_file": mu_prior_file,
            "eta_mode": args.eta_mode,
            "sampler": args.sampler,
            "have_emcee": HAVE_EMCEE,
            "n_input_signal": int(len(signal)),
            "n_control_total": int(len(parent_views["ctl_phi1"])),
            "n_zparent_total": int(len(parent_views["zl_phi1"])),
            "phi1_min": PHI1_MIN,
            "phi1_max": PHI1_MAX,
            "phi1_step": PHI1_STEP,
            "window_scale": WINDOW_SCALE,
            "window_width_phi1": WINDOW_WIDTH_PHI1,
            "fit_halfwidth": FIT_HALFWIDTH,
            "min_signal_stars": MIN_SIGNAL_STARS,
            "track_coherence_abs": TRACK_COHERENCE_ABS,
            "sigma_coherence_min_ratio": SIGMA_COHERENCE_MIN_RATIO,
            "sigma_coherence_max_ratio": SIGMA_COHERENCE_MAX_RATIO,
            "n_mockfit_track_nodes_written": int(n_mockfit),
            **clean_stats,
        },
    )
    with open(outdir / "pal5_step3b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {outdir / 'pal5_step3b_summary.json'}")

    # QC plots
    save_density_map_with_track(signal, tab, outdir / "qc_step3b_density_phi12.png", full_frame=True)
    save_density_map_with_track(signal, tab, outdir / "qc_step3b_density_phi12_local.png", full_frame=False)
    save_control_density_map(parent_views, outdir / "qc_step3b_control_density_phi12.png")
    save_radec_map(signal, outdir / "qc_step3b_density_radec.png")
    save_track_plot(tab, poly_coeffs, outdir / "qc_step3b_track.png")
    save_track_resid_plot(tab, outdir / "qc_step3b_track_resid.png")
    save_profile_plot(tab, "sigma", "sigma_err", r"width $\sigma(\phi_1)$ [deg]", "Selection-aware Bonaca-style 1D model: stream width", outdir / "qc_step3b_width.png")
    save_profile_plot(tab, "linear_density", "linear_density_err", "stream stars / deg", "Selection-aware Bonaca-style 1D model: linear density profile", outdir / "qc_step3b_linear_density.png")
    save_profile_plot(tab, "f_stream", "f_stream_err", "stream fraction in fit window", "Selection-aware Bonaca-style 1D model: fitted stream fraction", outdir / "qc_step3b_stream_fraction.png")
    save_example_fits(signal, tab, template_cache, outdir / "qc_step3b_example_local_fits.png")
    save_eta_examples(template_cache, centers, outdir / "qc_step3b_eta_examples.png")
    print(f"[done] outputs written to {outdir}")


if __name__ == "__main__":
    main()
