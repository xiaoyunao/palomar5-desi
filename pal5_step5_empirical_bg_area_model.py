#!/usr/bin/env python3
"""
Pal 5 step 5: empirical-background + effective-area upgraded 1D morphology model.

This is the next step after the step-3b / step-3c formal baseline and the
step-4 / step-4b distance-gradient experiments.

Scientific intent
-----------------
- Keep the step-2 strict member sample as the *formal baseline input*.
- Keep the stream morphology model simple: a single Gaussian in phi2.
- Upgrade the background model from "linear background modulated by eta" to an
  *empirical local background shape* built from control sidebands.
- Upgrade the observation model by including an effective-area / coverage
  template derived from the full preprocessed parent catalog.

In other words, within each overlapping phi1 window we fit:

    p_obs(phi2)  f * [A(phi2) * G(phi2 | mu, sigma)]
           + (1-f) * [A(phi2) * B_emp(phi2) * L(phi2)]

where
  A(phi2)    : effective-area / coverage template from the preprocessed catalog
  B_emp(phi2): empirical background shape from control sidebands
  L(phi2)    : weak linear residual freedom around the empirical background

This script is intentionally conservative:
- default sampler is MAP
- emcee is optional and used as a posterior sanity check
- outputs are shaped to be comparable to step 3b / step 3c

Inputs
------
- step2_outputs/pal5_step2_strict_members.fits
- final_g25_preproc.fits
- step2_outputs/pal5_step2_summary.json
- pal5.dat
- optionally a mu-prior track from step 3b control+MAP

Outputs
-------
- pal5_step5_profiles.fits / .csv
- pal5_step5_summary.json
- pal5_step5_control_sidebands.fits
- QC figures, including log-density maps and template diagnostics
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import emcee  # type: ignore
    HAVE_EMCEE = True
except Exception:
    emcee = None
    HAVE_EMCEE = False


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------
DEFAULT_SIGNAL = "step2_outputs/pal5_step2_strict_members.fits"
DEFAULT_PREPROC = "final_g25_preproc.fits"
DEFAULT_STEP2_SUMMARY = "step2_outputs/pal5_step2_summary.json"
DEFAULT_ISO = "pal5.dat"
DEFAULT_MU_PRIOR = "step3b_outputs_control/pal5_step3b_mu_prior.txt"
DEFAULT_OUTDIR = "step5_outputs"
DEFAULT_SUPPORT = "pal5_step3b_selection_aware_1d_model.py"

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

TEMPLATE_BIN = 0.05
MAP_BIN = 0.10
FULL_PHI1_MIN = -25.0
FULL_PHI1_MAX = 20.0
FULL_PHI2_MIN = -5.0
FULL_PHI2_MAX = 10.0
EXAMPLE_BIN_CENTERS = (-13.0, -7.0, -3.0, 0.0, 3.0, 6.0)

SIDEBAND_GAP = 0.03
SIDEBAND_WIDTH = 0.12

# Effective-area / empirical-background templates
AREA_PHI1_BIN = 0.10
AREA_SMOOTH_BINS = 1.0
AREA_MIN = 0.15
AREA_MAX = 5.0
BG_SMOOTH_BINS = 1.5
BG_MIN = 1e-3
BG_MAX = 100.0

# Parameterization / priors
SIGMA_MIN = 0.03
SIGMA_MAX = 1.20
UF_MIN = -7.0
UF_MAX = 7.0
RAW_TILT_MIN = -6.0
RAW_TILT_MAX = 6.0
MU_PRIOR_SIGMA = 0.35
MU_START_OFFSETS = (-0.12, 0.0, 0.12)

DEFAULT_SAMPLER = "map"  # map | emcee | auto
DEFAULT_NWALKERS = 48
DEFAULT_BURN = 256
DEFAULT_STEPS = 512
RNG_SEED = 24680


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


def choose_default_mu_prior_file(user_value: str) -> str:
    if user_value and os.path.exists(user_value):
        return user_value
    candidates = [
        "step3b_outputs_control/pal5_step3b_mu_prior.txt",
        "step3_outputs_hw15/pal5_step3_pass1_prior_track.txt",
        "step3_outputs/pal5_step3_pass1_prior_track.txt",
        "pal5_step3_pass1_prior_track.txt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


# -----------------------------------------------------------------------------
# Dynamic import of step-3b helpers for step-2 selection reproducibility
# -----------------------------------------------------------------------------
def load_support_module(path: str):
    spec = importlib.util.spec_from_file_location("pal5_step3b_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import support module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class TemplateBundle:
    grid: np.ndarray
    area_eta: np.ndarray
    bg_emp: np.ndarray
    n_control: int
    n_coverage: int
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
    n_coverage: int
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


# -----------------------------------------------------------------------------
# Parent views: control sidebands + coverage sample
# -----------------------------------------------------------------------------
def build_parent_views_and_coverage(
    preproc_fits: str,
    models,
    outdir: Path,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """
    Build three views from the preprocessed catalog:
      1) control sidebands: strict_mag + z_locus + isochrone sidebands
      2) z-parent sample: strict_mag + z_locus only (diagnostic / plotting)
      3) coverage sample: all finite preprocessed sources in the modeled phi1/phi2 box,
         used only to estimate effective-area occupancy
    """
    print(f"[read parent] {preproc_fits}")
    hdul = fits.open(preproc_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[info] parent rows={n_rows:,}")

    needed = ["PHI1", "PHI2", "G0", "R0", "Z0"]
    missing = [c for c in needed if c not in data.dtype.names]
    if missing:
        raise KeyError(f"Missing required columns in parent FITS: {missing}")

    ctl_phi1_list: List[np.ndarray] = []
    ctl_phi2_list: List[np.ndarray] = []
    zl_phi1_list: List[np.ndarray] = []
    zl_phi2_list: List[np.ndarray] = []
    cov_phi1_list: List[np.ndarray] = []
    cov_phi2_list: List[np.ndarray] = []

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]

        phi1 = np.asarray(sub["PHI1"], dtype=float)
        phi2 = np.asarray(sub["PHI2"], dtype=float)
        g0 = np.asarray(sub["G0"], dtype=float)
        r0 = np.asarray(sub["R0"], dtype=float)
        z0 = np.asarray(sub["Z0"], dtype=float)

        m_fin = finite_grz(g0, r0, z0)
        m_cov = (
            m_fin
            & (phi1 >= PHI1_MIN - 0.5)
            & (phi1 <= PHI1_MAX + 0.5)
            & (phi2 >= FULL_PHI2_MIN)
            & (phi2 <= FULL_PHI2_MAX)
        )
        if np.any(m_cov):
            cov_phi1_list.append(phi1[m_cov].astype(np.float32))
            cov_phi2_list.append(phi2[m_cov].astype(np.float32))

        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = gz0 - (models.z_slope * gr0 + models.z_intercept)

        cfg = models.strict_cfg
        m_mag = (
            m_fin
            & (g0 >= cfg["STRICT_GMIN"])
            & (g0 < cfg["STRICT_GMAX"])
            & (gr0 >= cfg["STRICT_GR_MIN"])
            & (gr0 <= cfg["STRICT_GR_MAX"])
        )
        m_z = m_mag & (gr0 <= models.z_gr_max) & np.isfinite(zres) & (np.abs(zres) <= models.z_tol)

        c_model = support.choose_ridge(phi1, models, g0)
        dcol = gr0 - c_model
        w = support.cmd_half_width(g0, cfg)

        blue = (dcol <= -(w + SIDEBAND_GAP)) & (dcol >= -(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        red = (dcol >= +(w + SIDEBAND_GAP)) & (dcol <= +(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        m_ctl = m_z & np.isfinite(dcol) & (blue | red)

        if np.any(m_ctl):
            ctl_phi1_list.append(phi1[m_ctl].astype(np.float32))
            ctl_phi2_list.append(phi2[m_ctl].astype(np.float32))
        if np.any(m_z):
            zl_phi1_list.append(phi1[m_z].astype(np.float32))
            zl_phi2_list.append(phi2[m_z].astype(np.float32))

        if ((start // chunk_size) % 5) == 0 or stop == n_rows:
            print(f"[parent chunk] {start:,}-{stop:,} | cov={np.sum(m_cov):,} z={np.sum(m_z):,} ctl={np.sum(m_ctl):,}")

    if len(ctl_phi1_list) == 0:
        raise RuntimeError("Control sideband sample is empty.")
    if len(zl_phi1_list) == 0:
        raise RuntimeError("z-locus parent sample is empty.")
    if len(cov_phi1_list) == 0:
        raise RuntimeError("Coverage sample is empty.")

    ctl_phi1 = np.concatenate(ctl_phi1_list)
    ctl_phi2 = np.concatenate(ctl_phi2_list)
    zl_phi1 = np.concatenate(zl_phi1_list)
    zl_phi2 = np.concatenate(zl_phi2_list)
    cov_phi1 = np.concatenate(cov_phi1_list)
    cov_phi2 = np.concatenate(cov_phi2_list)

    control_tab = Table({"PHI1": ctl_phi1, "PHI2": ctl_phi2})
    control_path = outdir / "pal5_step5_control_sidebands.fits"
    control_tab.write(control_path, overwrite=True)
    print(f"[write] {control_path}")

    return {
        "ctl_phi1": ctl_phi1,
        "ctl_phi2": ctl_phi2,
        "zl_phi1": zl_phi1,
        "zl_phi2": zl_phi2,
        "cov_phi1": cov_phi1,
        "cov_phi2": cov_phi2,
    }


# -----------------------------------------------------------------------------
# Template construction
# -----------------------------------------------------------------------------
def build_template_grid(fit_lo: float, fit_hi: float, binw: float = TEMPLATE_BIN) -> np.ndarray:
    n = max(64, int(np.ceil((fit_hi - fit_lo) / binw)) + 1)
    return np.linspace(fit_lo, fit_hi, n)


def build_area_eta(phi1_cov: np.ndarray, phi2_cov: np.ndarray, phi1_lo: float, phi1_hi: float, grid: np.ndarray) -> Tuple[np.ndarray, int]:
    if phi2_cov.size < 20:
        return np.ones_like(grid), 0
    phi1_edges = np.arange(phi1_lo, phi1_hi + AREA_PHI1_BIN, AREA_PHI1_BIN)
    if phi1_edges.size < 2:
        phi1_edges = np.array([phi1_lo, phi1_hi], dtype=float)
    dphi2 = float(grid[1] - grid[0])
    phi2_edges = np.concatenate([[grid[0] - 0.5 * dphi2], 0.5 * (grid[1:] + grid[:-1]), [grid[-1] + 0.5 * dphi2]])
    H, _, _ = np.histogram2d(phi1_cov, phi2_cov, bins=[phi1_edges, phi2_edges])
    occ = (H > 0).astype(float)
    frac = occ.mean(axis=0)
    frac = gaussian_filter1d(frac, AREA_SMOOTH_BINS, mode="nearest")
    frac = fill_nan_linear(frac)
    good = frac > 0
    if np.any(good):
        frac /= np.mean(frac[good])
    else:
        frac[:] = 1.0
    frac = np.clip(frac, AREA_MIN, AREA_MAX)
    return frac, int(np.sum(H))


def build_empirical_bg(phi2_control: np.ndarray, area_eta: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if phi2_control.size < 20:
        return np.ones_like(grid)
    dphi2 = float(grid[1] - grid[0])
    edges = np.concatenate([[grid[0] - 0.5 * dphi2], 0.5 * (grid[1:] + grid[:-1]), [grid[-1] + 0.5 * dphi2]])
    hist, _ = np.histogram(phi2_control, bins=edges)
    dens = hist.astype(float) / np.clip(area_eta, 1e-3, None)
    dens = gaussian_filter1d(dens, BG_SMOOTH_BINS, mode="nearest")
    dens = fill_nan_linear(dens)
    if np.mean(dens) <= 0:
        dens[:] = 1.0
    else:
        dens /= np.mean(dens)
    return np.clip(dens, BG_MIN, BG_MAX)


def build_empirical_template(
    ctl_phi2: np.ndarray,
    cov_phi1: np.ndarray,
    cov_phi2: np.ndarray,
    phi1_lo: float,
    phi1_hi: float,
    fit_lo: float,
    fit_hi: float,
) -> TemplateBundle:
    grid = build_template_grid(fit_lo, fit_hi, binw=TEMPLATE_BIN)
    area_eta, n_cov = build_area_eta(cov_phi1, cov_phi2, phi1_lo, phi1_hi, grid)
    bg_emp = build_empirical_bg(ctl_phi2, area_eta, grid)
    return TemplateBundle(
        grid=grid,
        area_eta=area_eta,
        bg_emp=bg_emp,
        n_control=int(len(ctl_phi2)),
        n_coverage=int(n_cov),
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


def empirical_bg_shape(x: np.ndarray, raw_tilt: float, template: TemplateBundle) -> np.ndarray:
    bg = safe_interp(x, template.grid, template.bg_emp, fill=1.0)
    return np.clip(bg * linear_bg_shape(x, raw_tilt, template.fit_lo, template.fit_hi), 1e-6, None)


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
    lp = 0.0
    lp += -0.5 * ((mu - mu_prior_center) / mu_prior_sigma) ** 2
    return lp


def component_pdfs_observed(phi2_eval: np.ndarray, theta: np.ndarray, template: TemplateBundle) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u_f, mu, log_sigma, raw_tilt = theta
    f = float(sigmoid(u_f))
    sigma = float(np.exp(log_sigma))

    area_i = safe_interp(phi2_eval, template.grid, template.area_eta, fill=1.0)
    area_g = template.area_eta
    grid = template.grid

    s_i = area_i * gaussian_pdf(phi2_eval, mu, sigma)
    s_g = area_g * gaussian_pdf(grid, mu, sigma)
    b_i = area_i * empirical_bg_shape(phi2_eval, raw_tilt, template)
    b_g = area_g * empirical_bg_shape(grid, raw_tilt, template)

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
    for _ in range(max(5000, nwalkers * 50)):
        trial = theta_map + rng.normal(scale=scales, size=ndim)
        trial[0] = np.clip(trial[0], UF_MIN, UF_MAX)
        trial[1] = np.clip(trial[1], template.fit_lo, template.fit_hi)
        trial[2] = np.clip(trial[2], np.log(SIGMA_MIN), np.log(SIGMA_MAX))
        trial[3] = np.clip(trial[3], RAW_TILT_MIN, RAW_TILT_MAX)
        lp = log_posterior(trial, phi2, template, mu_prior_center, mu_prior_sigma)
        if np.isfinite(lp):
            p0.append(trial)
        if len(p0) >= nwalkers:
            break
    if len(p0) < nwalkers:
        return posterior_from_map_only(theta_map, phi2, template, mu_prior_center, mu_prior_sigma)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(phi2, template, mu_prior_center, mu_prior_sigma))
    try:
        pos, _, _ = sampler.run_mcmc(np.asarray(p0), burn, progress=False)
        sampler.reset()
        sampler.run_mcmc(pos, steps, progress=False)
    except Exception:
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
    f_stream_err = abs(f_stream * (1.0 - f_stream) * ef) if np.isfinite(ef) else np.nan
    sigma = float(np.exp(log_sigma))
    sigma_err = abs(sigma * elogsig) if np.isfinite(elogsig) else np.nan

    bg_tilt = transform_bg_tilt(raw_tilt, template.fit_lo, template.fit_hi)
    bg_tilt_err = np.nan
    if np.isfinite(erawt):
        half = 0.5 * (template.fit_hi - template.fit_lo)
        deriv = 0.95 * (1.0 - np.tanh(raw_tilt) ** 2) / max(half, 1e-6)
        bg_tilt_err = abs(deriv) * erawt

    p_s, p_b, _, _, _, _ = component_pdfs_observed(phi2, theta, template)
    if not np.all(np.isfinite(p_s)) or not np.all(np.isfinite(p_b)):
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


def summarize_one_bin(phi2: np.ndarray, template: TemplateBundle, row: BinResult, sampler_mode: str, nwalkers: int, burn: int, steps: int, rng: np.random.Generator) -> BinResult:
    theta_map, nlp_map, opt_ok, opt_msg = fit_single_bin_map(phi2, template, row.mu_prior_center, row.mu_prior_sigma)
    row.optimizer_success = bool(opt_ok)
    row.map_neglogpost = float(nlp_map) if nlp_map is not None else np.nan
    if theta_map is None:
        row.success = False
        row.message = opt_msg
        return row

    if sampler_mode == "map" or (sampler_mode == "auto" and not HAVE_EMCEE):
        theta, theta_err, cov, samp_ok, msg, acc = posterior_from_map_only(theta_map, phi2, template, row.mu_prior_center, row.mu_prior_sigma)
        row.sampler_used = "map"
    else:
        theta, theta_err, cov, samp_ok, msg, acc = posterior_from_emcee(theta_map, phi2, template, row.mu_prior_center, row.mu_prior_sigma, nwalkers, burn, steps, rng)
        row.sampler_used = "emcee" if HAVE_EMCEE else "map"

    row.sampler_success = bool(samp_ok)
    row.acc_frac = float(acc) if np.isfinite(acc) else np.nan
    row.message = msg if samp_ok else f"{opt_msg}; {msg}"

    vals = unpack_theta(theta, theta_err, phi2, template)
    for k, v in vals.items():
        setattr(row, k, v)

    row.success = bool(np.isfinite(row.mu) and np.isfinite(row.sigma) and (row.sigma >= SIGMA_MIN) and (row.sigma <= SIGMA_MAX))
    return row


# -----------------------------------------------------------------------------
# Fit loop and result handling
# -----------------------------------------------------------------------------
def load_signal_catalog(signal_fits: str) -> Table:
    tab = Table.read(signal_fits)
    needed = ["PHI1", "PHI2", "RA", "DEC"]
    missing = [c for c in needed if c not in tab.colnames]
    if missing:
        raise KeyError(f"Missing columns in signal FITS: {missing}")
    return tab


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
        return np.interp(centers, xp, fp).astype(float)

    half_w = 0.5 * WINDOW_WIDTH_PHI1
    mu0 = np.full_like(centers, np.nan, dtype=float)
    for i, c in enumerate(centers):
        m = (signal_phi1 >= c - half_w) & (signal_phi1 < c + half_w) & (signal_phi2 >= PASS0_PHI2_MIN) & (signal_phi2 <= PASS0_PHI2_MAX)
        mu0[i] = histogram_mode(signal_phi2[m], PASS0_PHI2_MIN, PASS0_PHI2_MAX, TEMPLATE_BIN)
    return smooth_track_by_arm(centers, mu0)


def fit_all_bins(signal: Table, parent_views: Dict[str, np.ndarray], centers: np.ndarray, mu_prior: np.ndarray, sampler_mode: str, nwalkers: int, burn: int, steps: int, rng: np.random.Generator) -> Tuple[List[BinResult], Dict[float, TemplateBundle]]:
    sig_phi1 = np.asarray(signal["PHI1"], dtype=float)
    sig_phi2 = np.asarray(signal["PHI2"], dtype=float)

    ctl_phi1 = parent_views["ctl_phi1"]
    ctl_phi2 = parent_views["ctl_phi2"]
    cov_phi1 = parent_views["cov_phi1"]
    cov_phi2 = parent_views["cov_phi2"]

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

        m_cov = (cov_phi1 >= phi1_lo) & (cov_phi1 < phi1_hi) & (cov_phi2 >= fit_lo) & (cov_phi2 <= fit_hi)
        y_cov_phi1 = np.asarray(cov_phi1[m_cov], dtype=float)
        y_cov_phi2 = np.asarray(cov_phi2[m_cov], dtype=float)

        template = build_empirical_template(y_ctl, y_cov_phi1, y_cov_phi2, phi1_lo, phi1_hi, fit_lo, fit_hi)
        template_cache[c] = template

        row = BinResult(
            phi1_center=float(c),
            phi1_lo=float(phi1_lo),
            phi1_hi=float(phi1_hi),
            fit_lo=float(fit_lo),
            fit_hi=float(fit_hi),
            n_signal=int(len(y)),
            n_control=int(template.n_control),
            n_coverage=int(template.n_coverage),
            cluster_bin=bool(is_cluster_bin),
            success=False,
            sampler_used="map" if sampler_mode == "map" else ("emcee" if HAVE_EMCEE else "map"),
            optimizer_success=False,
            sampler_success=False,
            message="not_fitted",
            mu_prior_center=float(mu0),
            mu_prior_sigma=float(MU_PRIOR_SIGMA),
        )

        row = summarize_one_bin(y, template, row, sampler_mode, nwalkers, burn, steps, rng)
        results.append(row)
        print(f"[bin {i+1:02d}/{len(centers)}] phi1={c:+5.2f} | Nsig={len(y):4d} Nctl={len(y_ctl):4d} Ncov={len(y_cov_phi2):5d} | success={row.success} | msg={row.message}")

    return results, template_cache


def results_to_table(results: Sequence[BinResult]) -> Table:
    cols: Dict[str, List[object]] = {}
    for row in results:
        for k, v in asdict(row).items():
            cols.setdefault(k, []).append(v)
    return Table(cols)


def polyfit_arm(phi1: np.ndarray, mu: np.ndarray, mu_err: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    m = mask & np.isfinite(phi1) & np.isfinite(mu)
    if np.sum(m) < 4:
        return None
    x = phi1[m]
    y = mu[m]
    err = np.asarray(mu_err[m], dtype=float)
    if np.isfinite(err).sum() >= 3:
        w = 1.0 / np.clip(err, 0.03, None)
    else:
        w = None
    try:
        return np.polyfit(x, y, deg=2, w=w)
    except Exception:
        return None


def append_track_polynomials(tab: Table) -> Dict[str, Optional[np.ndarray]]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    mu = np.asarray(tab["mu"], dtype=float)
    mu_err = np.asarray(tab["mu_err"], dtype=float)
    success = np.asarray(tab["success"], dtype=bool)
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


def summarize_results(tab: Table, poly_coeffs: Dict[str, Optional[np.ndarray]], meta: Dict[str, object]) -> Dict[str, object]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    success = np.asarray(tab["success"], dtype=bool)
    cluster = np.asarray(tab["cluster_bin"], dtype=bool)
    sigma = np.asarray(tab["sigma"], dtype=float)
    lin = np.asarray(tab["linear_density"], dtype=float)

    m_lead = success & ~cluster & (phi1 > 0.0) & np.isfinite(sigma)
    m_trail = success & ~cluster & (phi1 < 0.0) & np.isfinite(sigma)
    m_dens = success & ~cluster & np.isfinite(lin)

    out = dict(meta)
    out.update({
        "n_bins": int(len(tab)),
        "n_success": int(np.sum(success)),
        "n_success_excluding_cluster": int(np.sum(success & ~cluster)),
        "track_poly_trailing": poly_coeffs["trailing"].tolist() if poly_coeffs["trailing"] is not None else None,
        "track_poly_leading": poly_coeffs["leading"].tolist() if poly_coeffs["leading"] is not None else None,
        "max_width_leading": float(np.nanmax(sigma[m_lead])) if np.any(m_lead) else None,
        "max_width_trailing": float(np.nanmax(sigma[m_trail])) if np.any(m_trail) else None,
        "integrated_stream_stars_excluding_cluster": float(np.nansum(lin[m_dens] * WINDOW_WIDTH_PHI1)) if np.any(m_dens) else None,
    })
    return out


# -----------------------------------------------------------------------------
# QC plotting
# -----------------------------------------------------------------------------
def _lognorm_for_density(D: np.ndarray) -> Optional[LogNorm]:
    pos = D[np.isfinite(D) & (D > 0)]
    if pos.size == 0:
        return None
    vmin = max(float(np.nanpercentile(pos, 5)), 1e-3)
    vmax = float(np.nanpercentile(pos, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(pos))
    if vmax <= vmin:
        return None
    return LogNorm(vmin=vmin, vmax=vmax)


def save_density_map_with_track(tab_members: Table, tab_fit: Table, out_png: Path, full_frame: bool = True) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)

    if full_frame:
        xlo, xhi = FULL_PHI1_MIN, FULL_PHI1_MAX
        ylo, yhi = FULL_PHI2_MIN, FULL_PHI2_MAX
        title = "step 5 empirical-bg model: Pal 5 frame number density (log scale)"
    else:
        xlo, xhi = PHI1_MIN, PHI1_MAX
        ylo, yhi = PASS0_PHI2_MIN, PASS0_PHI2_MAX
        title = "step 5 empirical-bg model: local Pal 5 frame density (log scale)"

    xedges = np.arange(xlo, xhi + MAP_BIN, MAP_BIN)
    yedges = np.arange(ylo, yhi + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(phi1, phi2, bins=[xedges, yedges])
    D = H.T / (MAP_BIN * MAP_BIN)

    plt.figure(figsize=(11, 6.5))
    norm = _lognorm_for_density(D)
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest", norm=norm)
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")

    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    mu = np.asarray(tab_fit["mu"], dtype=float)
    sig = np.asarray(tab_fit["sigma"], dtype=float)
    ok = success & np.isfinite(mu) & np.isfinite(sig)
    if np.any(ok):
        plt.plot(x[ok], mu[ok], lw=2.0, color="C0")
        plt.plot(x[ok], mu[ok] + sig[ok], "--", lw=1.2, color="C1")
        plt.plot(x[ok], mu[ok] - sig[ok], "--", lw=1.2, color="C2")

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
    norm = _lognorm_for_density(D)
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest", norm=norm)
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step 5 control sample (strict mag + z-locus + isochrone sidebands, log scale)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_coverage_density_map(parent_views: Dict[str, np.ndarray], out_png: Path) -> None:
    x = np.asarray(parent_views["cov_phi1"], dtype=float)
    y = np.asarray(parent_views["cov_phi2"], dtype=float)
    xedges = np.arange(FULL_PHI1_MIN, FULL_PHI1_MAX + MAP_BIN, MAP_BIN)
    yedges = np.arange(FULL_PHI2_MIN, FULL_PHI2_MAX + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges])
    D = H.T / (MAP_BIN * MAP_BIN)
    plt.figure(figsize=(11, 6.5))
    norm = _lognorm_for_density(D)
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest", norm=norm)
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step 5 coverage proxy from full preprocessed parent catalog (log scale)")
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
    norm = _lognorm_for_density(D)
    plt.imshow(D, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest", norm=norm)
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("step 5 strict selected sample: RA-Dec number density (log scale)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_track_plot(tab_fit: Table, poly_coeffs: Dict[str, Optional[np.ndarray]], out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    mu = np.asarray(tab_fit["mu"], dtype=float)
    mu_err = np.asarray(tab_fit["mu_err"], dtype=float)
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
    plt.title("step 5 empirical-background model: stream track")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_track_resid_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    r = np.asarray(tab_fit["track_resid"], dtype=float)
    rerr = np.asarray(tab_fit["mu_err"], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool)
    m = success & ~cluster & np.isfinite(r)
    plt.figure(figsize=(10, 4.2))
    plt.axhline(0.0, color="0.5", lw=1.0)
    plt.errorbar(x[m], r[m], yerr=rerr[m], fmt="o", ms=3.5, lw=1.0, capsize=2)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("track residual [deg]")
    plt.title("step 5 empirical-background model: track residual")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_profile_plot(tab_fit: Table, ycol: str, yerrcol: str, ylabel: str, title: str, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    y = np.asarray(tab_fit[ycol], dtype=float)
    yerr = np.asarray(tab_fit[yerrcol], dtype=float)
    success = np.asarray(tab_fit["success"], dtype=bool)
    cluster = np.asarray(tab_fit["cluster_bin"], dtype=bool)
    ok = success & np.isfinite(y)
    plt.figure(figsize=(10, 4.2))
    plt.errorbar(x[ok], y[ok], yerr=yerr[ok], fmt="o", ms=3.5, lw=1.0, capsize=2, label="usable bins")
    if np.any(cluster & np.isfinite(y)):
        m = cluster & np.isfinite(y)
        plt.errorbar(x[m], y[m], yerr=yerr[m], fmt="s", ms=4.0, lw=1.0, capsize=2, color="0.5", label="cluster bins")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_example_fits(tab_members: Table, tab_fit: Table, template_cache: Dict[float, TemplateBundle], out_png: Path) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)
    half_w = 0.5 * WINDOW_WIDTH_PHI1

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharey=True)
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
        bins = np.arange(fit_lo, fit_hi + TEMPLATE_BIN, TEMPLATE_BIN)
        hist, edges = np.histogram(y, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax.step(mids, hist, where="mid", lw=1.2, label="data")

        tmpl = template_cache.get(c)
        if tmpl is not None:
            area_scaled = safe_interp(mids, tmpl.grid, tmpl.area_eta, fill=1.0)
            area_scaled = area_scaled / np.nanmax(area_scaled) * max(np.max(hist), 1)
            bg_scaled = safe_interp(mids, tmpl.grid, tmpl.bg_emp, fill=1.0)
            bg_scaled = bg_scaled / np.nanmax(bg_scaled) * max(np.max(hist), 1)
            ax.plot(mids, area_scaled, color="0.5", ls=":", lw=1.3, label="area eta (scaled)")
            ax.plot(mids, bg_scaled, color="C3", ls="--", lw=1.2, label="empirical bg (scaled)")

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
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(r"step 5 local fits: empirical background + effective area")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_template_examples(template_cache: Dict[float, TemplateBundle], centers: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharey=True)
    axes = axes.ravel()
    for ax, target in zip(axes, EXAMPLE_BIN_CENTERS):
        c = float(centers[int(np.argmin(np.abs(centers - target)))])
        tmpl = template_cache[c]
        ax.plot(tmpl.grid, tmpl.area_eta, label="area eta")
        ax.plot(tmpl.grid, tmpl.bg_emp, lw=2.0, label="empirical bg")
        ax.set_title(rf"$\phi_1 \approx {c:+.2f}^\circ$")
        ax.set_xlabel(r"$\phi_2$ [deg]")
        if ax in axes[::3]:
            ax.set_ylabel("template amplitude")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("step 5 templates: effective area and empirical background")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 step 5: empirical-background + effective-area 1D model")
    p.add_argument("--signal", default=DEFAULT_SIGNAL, help="Input strict member FITS catalog")
    p.add_argument("--preproc", default=DEFAULT_PREPROC, help="Preprocessed parent FITS catalog")
    p.add_argument("--step2-summary", default=DEFAULT_STEP2_SUMMARY, help="Step 2 summary JSON")
    p.add_argument("--iso", default=DEFAULT_ISO, help="Isochrone file used in step 2")
    p.add_argument("--mu-prior-file", default=DEFAULT_MU_PRIOR, help="Optional mu-prior track text file")
    p.add_argument("--support-script", default=DEFAULT_SUPPORT, help="Path to pal5_step3b_selection_aware_1d_model.py")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
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
    global PHI1_MIN, PHI1_MAX, PHI1_STEP, WINDOW_WIDTH_PHI1, FIT_HALFWIDTH, MIN_SIGNAL_STARS, support

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

    support = load_support_module(args.support_script)
    if args.sampler == "auto":
        sampler_mode = "emcee" if HAVE_EMCEE else "map"
    else:
        sampler_mode = args.sampler

    print(f"[info] support script = {args.support_script}")
    print(f"[info] sampler mode   = {sampler_mode}")
    print(f"[info] HAVE_EMCEE     = {HAVE_EMCEE}")

    signal = load_signal_catalog(args.signal)
    models = support.load_step2_models(args.step2_summary, args.iso)
    parent_views = build_parent_views_and_coverage(args.preproc, models, outdir, chunk_size=args.chunk_size)

    centers = np.arange(PHI1_MIN, PHI1_MAX + 0.5 * PHI1_STEP, PHI1_STEP)
    mu_prior_file = choose_default_mu_prior_file(args.mu_prior_file)
    if mu_prior_file:
        print(f"[info] using mu prior file: {mu_prior_file}")
    else:
        print("[warn] no mu prior file found; falling back to histogram-mode ridge")

    mu_prior = load_mu_prior(
        centers,
        mu_prior_file,
        np.asarray(signal["PHI1"], dtype=float),
        np.asarray(signal["PHI2"], dtype=float),
    )
    np.savetxt(outdir / "pal5_step5_mu_prior.txt", np.c_[centers, mu_prior], header="phi1_center  mu_prior")

    results, template_cache = fit_all_bins(
        signal=signal,
        parent_views=parent_views,
        centers=centers,
        mu_prior=mu_prior,
        sampler_mode=sampler_mode,
        nwalkers=args.nwalkers,
        burn=args.burn,
        steps=args.steps,
        rng=rng,
    )

    tab = results_to_table(results)
    poly_coeffs = append_track_polynomials(tab)
    fits_path = outdir / "pal5_step5_profiles.fits"
    csv_path = outdir / "pal5_step5_profiles.csv"
    tab.write(fits_path, overwrite=True)
    tab.write(csv_path, format="ascii.csv", overwrite=True)
    print(f"[write] {fits_path}")
    print(f"[write] {csv_path}")

    summary = summarize_results(
        tab,
        poly_coeffs,
        {
            "signal": args.signal,
            "preproc": args.preproc,
            "step2_summary": args.step2_summary,
            "iso": args.iso,
            "mu_prior_file": mu_prior_file,
            "support_script": args.support_script,
            "sampler": sampler_mode,
            "have_emcee": HAVE_EMCEE,
            "n_input_signal": int(len(signal)),
            "n_control_total": int(len(parent_views["ctl_phi1"])),
            "n_zparent_total": int(len(parent_views["zl_phi1"])),
            "n_coverage_total": int(len(parent_views["cov_phi1"])),
            "phi1_min": PHI1_MIN,
            "phi1_max": PHI1_MAX,
            "phi1_step": PHI1_STEP,
            "window_scale": WINDOW_SCALE,
            "window_width_phi1": WINDOW_WIDTH_PHI1,
            "fit_halfwidth": FIT_HALFWIDTH,
            "min_signal_stars": MIN_SIGNAL_STARS,
            "area_phi1_bin": AREA_PHI1_BIN,
        },
    )
    with open(outdir / "pal5_step5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {outdir / 'pal5_step5_summary.json'}")

    # QC plots
    save_density_map_with_track(signal, tab, outdir / "qc_step5_density_phi12_log.png", full_frame=True)
    save_density_map_with_track(signal, tab, outdir / "qc_step5_density_phi12_local_log.png", full_frame=False)
    save_control_density_map(parent_views, outdir / "qc_step5_control_density_phi12_log.png")
    save_coverage_density_map(parent_views, outdir / "qc_step5_coverage_phi12_log.png")
    save_radec_map(signal, outdir / "qc_step5_density_radec_log.png")
    save_track_plot(tab, poly_coeffs, outdir / "qc_step5_track.png")
    save_track_resid_plot(tab, outdir / "qc_step5_track_resid.png")
    save_profile_plot(tab, "sigma", "sigma_err", r"width $\sigma(\phi_1)$ [deg]", "step 5 empirical-background model: stream width", outdir / "qc_step5_width.png")
    save_profile_plot(tab, "linear_density", "linear_density_err", "stream stars / deg", "step 5 empirical-background model: linear density profile", outdir / "qc_step5_linear_density.png")
    save_profile_plot(tab, "f_stream", "f_stream_err", "stream fraction in fit window", "step 5 empirical-background model: fitted stream fraction", outdir / "qc_step5_stream_fraction.png")
    save_example_fits(signal, tab, template_cache, outdir / "qc_step5_example_local_fits.png")
    save_template_examples(template_cache, centers, outdir / "qc_step5_template_examples.png")
    print(f"[done] outputs written to {outdir}")


if __name__ == "__main__":
    main()
