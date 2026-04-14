#!/usr/bin/env python3
"""
Pal 5 step 5a: empirical-background model with off-stream anchoring.

Why this exists
---------------
The first step-5 attempt mixed two ingredients in a way that made the stream
and background hard to disentangle:
  1) a parent-catalog count map used as a multiplicative "coverage" template,
  2) an empirical background template that was still too free to look like the
     stream itself.

This step-5a variant is intentionally simpler and more identifiable:
  - keep the intrinsic stream model = single Gaussian in phi2,
  - learn the empirical background shape from control-sideband stars,
  - *anchor* the background normalization using off-stream regions only,
  - do NOT multiply the likelihood by the parent-count stripe as an amplitude
    template (we only keep it as a diagnostic map / optional binary mask),
  - keep log-scale QC figures.

The goal is to test whether the leading-fan discrepancy is better explained by
an empirical off-stream background treatment than by additional distance-model
complexity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------
DEFAULT_SIGNAL = "step2_outputs/pal5_step2_strict_members.fits"
DEFAULT_PREPROC = "final_g25_preproc.fits"
DEFAULT_STEP2_SUMMARY = "step2_outputs/pal5_step2_summary.json"
DEFAULT_ISO = "pal5.dat"
DEFAULT_MU_PRIOR = "step3b_outputs_control/pal5_step3b_mu_prior.txt"
DEFAULT_SUPPORT = "pal5_step3b_selection_aware_1d_model.py"
DEFAULT_OUTDIR = "step5a_outputs_control_map"

PHI1_MIN = -20.0
PHI1_MAX = 10.0
PHI1_STEP = 0.75
WINDOW_SCALE = 1.5
WINDOW_WIDTH_PHI1 = PHI1_STEP * WINDOW_SCALE
FIT_HALFWIDTH = 1.50
MIN_SIGNAL_STARS = 60
CLUSTER_MASK_HALFWIDTH = 0.75

FULL_PHI1_MIN = -25.0
FULL_PHI1_MAX = 20.0
FULL_PHI2_MIN = -5.0
FULL_PHI2_MAX = 10.0
LOCAL_PHI2_MIN = -2.5
LOCAL_PHI2_MAX = 2.5

BIN_FIT = 0.05
BIN_MAP_PHI1 = 0.25
BIN_MAP_PHI2 = 0.05
BIN_PARENT_PROXY = 0.10

# strict sideband control construction
SIDEBAND_GAP = 0.03
SIDEBAND_WIDTH = 0.12
ZLOCUS_SLOPE = 1.7
ZLOCUS_INTERCEPT = -0.17
ZLOCUS_GR_MAX = 1.2

# off-stream anchored background template
BG_EXCLUDE_HALF = 0.55
OFF_INNER = 0.65
OFF_OUTER = 1.50
BG_SMOOTH_BINS = 1.5
BG_MIN_COUNTS = 1e-3

# stream model priors / bounds
MU_PRIOR_SIGMA = 0.30
SIGMA_MIN = 0.03
SIGMA_MAX = 0.70
MU_START_OFFSETS = (-0.15, 0.0, 0.15)
NSTREAM_MIN = 1e-6
NSTREAM_MAX = 5e4

EXAMPLE_BIN_CENTERS = (-13.0, -7.0, -3.0, 0.0, 3.0, 6.0)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


def finite_grz(g0: np.ndarray, r0: np.ndarray, z0: np.ndarray) -> np.ndarray:
    return np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)


def safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


def histogram2d_density(x: np.ndarray, y: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    area = np.diff(xedges)[:, None] * np.diff(yedges)[None, :]
    return h / area


def fill_nan_linear(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float)
    m = np.isfinite(y)
    out = y.copy()
    if m.sum() == 0:
        return np.zeros_like(y)
    if m.sum() == 1:
        out[~m] = out[m][0]
        return out
    out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def mad_scale(x: np.ndarray, floor: float = 0.02) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return floor
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, floor)


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
# Dynamic import of step-3b helpers
# -----------------------------------------------------------------------------
def load_support_module(path: str):
    spec = importlib.util.spec_from_file_location("pal5_step3b_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import support module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Critical for Python 3.11 + dataclass interaction
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class LocalTemplate:
    centers: np.ndarray
    bg_counts: np.ndarray
    n_signal_off: int
    n_control_off: int
    scale_bg: float
    mask_off: np.ndarray
    mask_fit: np.ndarray


@dataclass
class BinResult:
    phi1_center: float
    phi1_lo: float
    phi1_hi: float
    fit_lo: float
    fit_hi: float
    n_signal: int
    n_control: int
    cluster_bin: bool
    success: bool
    optimizer_success: bool
    message: str
    mu_prior_center: float
    mu_prior_sigma: float
    f_stream: float = np.nan
    f_stream_err: float = np.nan
    mu: float = np.nan
    mu_err: float = np.nan
    sigma: float = np.nan
    sigma_err: float = np.nan
    bg_tilt: float = 0.0
    bg_tilt_err: float = np.nan
    n_stream: float = np.nan
    n_stream_err: float = np.nan
    linear_density: float = np.nan
    linear_density_err: float = np.nan
    peak_surface_density: float = np.nan
    peak_surface_density_err: float = np.nan
    map_negloglike: float = np.nan
    acc_frac: float = np.nan
    track_poly: float = np.nan
    track_resid: float = np.nan


# -----------------------------------------------------------------------------
# IO / inputs
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 step 5a: off-stream anchored empirical background model")
    p.add_argument("--signal", default=DEFAULT_SIGNAL)
    p.add_argument("--preproc", default=DEFAULT_PREPROC)
    p.add_argument("--step2-summary", default=DEFAULT_STEP2_SUMMARY)
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--mu-prior-file", default=DEFAULT_MU_PRIOR)
    p.add_argument("--support-script", default=DEFAULT_SUPPORT)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--phi1-min", type=float, default=PHI1_MIN)
    p.add_argument("--phi1-max", type=float, default=PHI1_MAX)
    p.add_argument("--phi1-step", type=float, default=PHI1_STEP)
    p.add_argument("--window-scale", type=float, default=WINDOW_SCALE)
    p.add_argument("--fit-halfwidth", type=float, default=FIT_HALFWIDTH)
    p.add_argument("--min-signal-stars", type=int, default=MIN_SIGNAL_STARS)
    p.add_argument("--cluster-mask-halfwidth", type=float, default=CLUSTER_MASK_HALFWIDTH)
    p.add_argument("--bg-exclude-half", type=float, default=BG_EXCLUDE_HALF)
    p.add_argument("--off-inner", type=float, default=OFF_INNER)
    p.add_argument("--off-outer", type=float, default=OFF_OUTER)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--seed", type=int, default=24680)
    return p.parse_args()


def load_mu_prior(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Bad mu-prior file format: {path}")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def load_signal_catalog(path: str) -> Dict[str, np.ndarray]:
    t = Table.read(path)
    cols = ["PHI1", "PHI2"]
    missing = [c for c in cols if c not in t.colnames]
    if missing:
        raise KeyError(f"Missing signal columns: {missing}")
    return {
        "phi1": np.asarray(t["PHI1"], dtype=float),
        "phi2": np.asarray(t["PHI2"], dtype=float),
    }


# -----------------------------------------------------------------------------
# Build control sample and parent-count proxy
# -----------------------------------------------------------------------------
def build_control_sidebands(preproc_fits: str, support, models, chunk_size: int) -> Dict[str, np.ndarray]:
    hdul = fits.open(preproc_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)

    ctl_phi1, ctl_phi2 = [], []
    parent_phi1, parent_phi2 = [], []

    cfg = models.strict_cfg
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]

        phi1 = np.asarray(sub["PHI1"], dtype=float)
        phi2 = np.asarray(sub["PHI2"], dtype=float)
        g0 = np.asarray(sub["G0"], dtype=float)
        r0 = np.asarray(sub["R0"], dtype=float)
        z0 = np.asarray(sub["Z0"], dtype=float)

        m_fin = support.finite_grz(g0, r0, z0)
        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = support.zlocus_residual(gr0, gz0, 1.7, -0.17)

        m_mag = (
            m_fin
            & (g0 >= cfg["STRICT_GMIN"])
            & (g0 < cfg["STRICT_GMAX"])
            & (gr0 >= cfg["STRICT_GR_MIN"])
            & (gr0 <= cfg["STRICT_GR_MAX"])
        )
        m_z = m_mag & (gr0 <= ZLOCUS_GR_MAX) & np.isfinite(zres) & (np.abs(zres) <= float(cfg.get("ZLOCUS_TOL", 0.1)))

        if np.any(m_z):
            parent_phi1.append(phi1[m_z].astype(np.float32))
            parent_phi2.append(phi2[m_z].astype(np.float32))

        c_model = support.choose_ridge(phi1, models, g0)
        dcol = gr0 - c_model
        w = support.cmd_half_width(g0, cfg)

        blue = (dcol <= -(w + SIDEBAND_GAP)) & (dcol >= -(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        red = (dcol >= +(w + SIDEBAND_GAP)) & (dcol <= +(w + SIDEBAND_GAP + SIDEBAND_WIDTH))
        m_ctl = m_z & np.isfinite(dcol) & (blue | red)
        if np.any(m_ctl):
            ctl_phi1.append(phi1[m_ctl].astype(np.float32))
            ctl_phi2.append(phi2[m_ctl].astype(np.float32))

    if not ctl_phi1:
        raise RuntimeError("Control sideband sample is empty.")
    if not parent_phi1:
        raise RuntimeError("Parent strict-mag+z-locus sample is empty.")

    return {
        "ctl_phi1": np.concatenate(ctl_phi1),
        "ctl_phi2": np.concatenate(ctl_phi2),
        "parent_phi1": np.concatenate(parent_phi1),
        "parent_phi2": np.concatenate(parent_phi2),
    }


# -----------------------------------------------------------------------------
# Local background template
# -----------------------------------------------------------------------------
def build_local_template(
    phi2_signal: np.ndarray,
    phi2_control: np.ndarray,
    mu_prior: float,
    fit_lo: float,
    fit_hi: float,
    binw: float,
    bg_exclude_half: float,
    off_inner: float,
    off_outer: float,
) -> Tuple[np.ndarray, np.ndarray, LocalTemplate]:
    edges = np.arange(fit_lo, fit_hi + binw, binw, dtype=float)
    if edges[-1] < fit_hi:
        edges = np.append(edges, fit_hi)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts_signal, _ = np.histogram(phi2_signal, bins=edges)
    counts_control, _ = np.histogram(phi2_control, bins=edges)

    dx = np.abs(centers - mu_prior)
    mask_fit = dx <= (fit_hi - fit_lo) / 2.0 + 1e-6
    mask_off = (dx >= off_inner) & (dx <= off_outer)
    mask_bg_train = dx >= bg_exclude_half

    # Build control-shape template using only off-stream bins, then interpolate
    bg_train = counts_control.astype(float)
    bg_train[~mask_bg_train] = np.nan
    bg_fill = fill_nan_linear(bg_train)
    bg_smooth = gaussian_filter1d(bg_fill, BG_SMOOTH_BINS, mode="nearest")
    bg_smooth = np.clip(bg_smooth, BG_MIN_COUNTS, None)

    n_signal_off = int(np.sum(counts_signal[mask_off]))
    n_control_off = int(np.sum(counts_control[mask_off]))

    # Normalize background using off-stream regions only.
    bg_off_sum = float(np.sum(bg_smooth[mask_off]))
    if n_control_off <= 0 or bg_off_sum <= 0:
        scale_bg = 0.0
    else:
        scale_bg = n_signal_off / bg_off_sum
    bg_counts = scale_bg * bg_smooth

    template = LocalTemplate(
        centers=centers,
        bg_counts=bg_counts,
        n_signal_off=n_signal_off,
        n_control_off=n_control_off,
        scale_bg=scale_bg,
        mask_off=mask_off,
        mask_fit=mask_fit,
    )
    return centers, counts_signal.astype(float), template


# -----------------------------------------------------------------------------
# MAP fit with fixed background template
# -----------------------------------------------------------------------------
def neg_loglike(theta: np.ndarray, centers: np.ndarray, counts: np.ndarray, template: LocalTemplate, mu_prior: float, mu_prior_sigma: float) -> float:
    log_nstream, mu, log_sigma = theta
    nstream = float(np.exp(log_nstream))
    sigma = float(np.exp(log_sigma))

    if not (SIGMA_MIN <= sigma <= SIGMA_MAX):
        return np.inf
    if not (template.centers[0] <= mu <= template.centers[-1]):
        return np.inf
    if not (NSTREAM_MIN <= nstream <= NSTREAM_MAX):
        return np.inf

    binw = float(np.median(np.diff(centers))) if len(centers) > 1 else BIN_FIT
    lam_stream = nstream * gaussian_pdf(centers, mu, sigma) * binw
    lam = template.bg_counts + lam_stream
    lam = np.clip(lam, 1e-8, None)

    # Poisson NLL
    nll = np.sum(lam - counts * np.log(lam))
    # weak Gaussian prior on mu around the current ridge prior
    nll += 0.5 * ((mu - mu_prior) / mu_prior_sigma) ** 2
    return float(nll)


def fit_one_bin(centers: np.ndarray, counts: np.ndarray, template: LocalTemplate, mu_prior: float, mu_prior_sigma: float, sigma_max: float) -> Tuple[bool, np.ndarray, float, str]:
    global SIGMA_MAX
    old_sigma_max = SIGMA_MAX
    SIGMA_MAX = sigma_max
    try:
        best = None
        best_val = np.inf
        best_msg = ""
        total = float(np.sum(counts))
        bg_total = float(np.sum(template.bg_counts))
        excess = max(total - bg_total, 1.0)
        sigma0 = 0.15

        bounds = [
            (np.log(NSTREAM_MIN), np.log(NSTREAM_MAX)),
            (centers[0], centers[-1]),
            (np.log(SIGMA_MIN), np.log(SIGMA_MAX)),
        ]
        for dmu in MU_START_OFFSETS:
            theta0 = np.array([np.log(excess), mu_prior + dmu, np.log(sigma0)], dtype=float)
            res = minimize(
                neg_loglike,
                theta0,
                args=(centers, counts, template, mu_prior, mu_prior_sigma),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best = np.array(res.x, dtype=float)
                best_msg = str(res.message)
                best_success = bool(res.success)

        if best is None:
            return False, np.full(3, np.nan), np.inf, "no finite MAP fit"

        sigma = float(np.exp(best[2]))
        success = bool(best_success) and np.isfinite(best_val) and (sigma < 0.98 * SIGMA_MAX)
        return success, best, best_val, best_msg
    finally:
        SIGMA_MAX = old_sigma_max


# -----------------------------------------------------------------------------
# Main modeling loop
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    support_path = args.support_script
    if not os.path.exists(support_path):
        raise FileNotFoundError(f"Support script not found: {support_path}")
    support = load_support_module(support_path)
    models = support.load_step2_models(args.step2_summary, args.iso)

    mu_prior_file = choose_default_mu_prior_file(args.mu_prior_file)
    if not mu_prior_file:
        raise FileNotFoundError("Could not locate a mu-prior track file.")
    mu_prior_x, mu_prior_y = load_mu_prior(mu_prior_file)

    signal = load_signal_catalog(args.signal)
    phi1_sig = signal["phi1"]
    phi2_sig = signal["phi2"]

    aux = build_control_sidebands(args.preproc, support, models, args.chunk_size)
    ctl_phi1 = aux["ctl_phi1"]
    ctl_phi2 = aux["ctl_phi2"]
    parent_phi1 = aux["parent_phi1"]
    parent_phi2 = aux["parent_phi2"]

    # Save control sidebands for reproducibility
    Table({"PHI1": ctl_phi1, "PHI2": ctl_phi2}).write(outdir / "pal5_step5a_control_sidebands.fits", overwrite=True)

    phi1_centers = np.arange(args.phi1_min, args.phi1_max + 0.5 * args.phi1_step, args.phi1_step)
    half_window = 0.5 * args.phi1_step * args.window_scale

    results: List[BinResult] = []

    for phi1_c in phi1_centers:
        phi1_lo = phi1_c - half_window
        phi1_hi = phi1_c + half_window
        cluster_bin = abs(phi1_c) <= args.cluster_mask_halfwidth

        m_sig = (phi1_sig >= phi1_lo) & (phi1_sig < phi1_hi)
        m_ctl = (ctl_phi1 >= phi1_lo) & (ctl_phi1 < phi1_hi)
        n_sig = int(np.sum(m_sig))
        n_ctl = int(np.sum(m_ctl))
        mu_prior = float(np.interp(phi1_c, mu_prior_x, mu_prior_y, left=mu_prior_y[0], right=mu_prior_y[-1]))
        fit_lo = max(LOCAL_PHI2_MIN, mu_prior - args.fit_halfwidth)
        fit_hi = min(LOCAL_PHI2_MAX, mu_prior + args.fit_halfwidth)

        rec = BinResult(
            phi1_center=float(phi1_c),
            phi1_lo=float(phi1_lo),
            phi1_hi=float(phi1_hi),
            fit_lo=float(fit_lo),
            fit_hi=float(fit_hi),
            n_signal=n_sig,
            n_control=n_ctl,
            cluster_bin=cluster_bin,
            success=False,
            optimizer_success=False,
            message="",
            mu_prior_center=mu_prior,
            mu_prior_sigma=MU_PRIOR_SIGMA,
        )

        if n_sig < args.min_signal_stars or n_ctl < max(30, args.min_signal_stars // 2):
            rec.message = "insufficient signal/control stars"
            results.append(rec)
            continue

        centers, counts, template = build_local_template(
            phi2_sig[m_sig], ctl_phi2[m_ctl], mu_prior, fit_lo, fit_hi, BIN_FIT,
            args.bg_exclude_half, args.off_inner, args.off_outer,
        )

        success, theta_map, nll, msg = fit_one_bin(centers, counts, template, mu_prior, MU_PRIOR_SIGMA, args.sigma_max)
        rec.optimizer_success = success
        rec.success = success
        rec.message = msg
        rec.map_negloglike = nll

        if success:
            log_nstream, mu, log_sigma = theta_map
            nstream = float(np.exp(log_nstream))
            sigma = float(np.exp(log_sigma))
            bg_total = float(np.sum(template.bg_counts))
            f_stream = nstream / max(nstream + bg_total, 1e-6)
            # heuristic uncertainties for MAP exploratory run
            mu_err = max(0.015, sigma / np.sqrt(max(nstream, 1.0)))
            sigma_err = max(0.008, sigma / np.sqrt(max(2.0 * nstream, 1.0)))
            nstream_err = math.sqrt(max(nstream, 1.0))
            linear_density = nstream / args.phi1_step
            linear_density_err = nstream_err / args.phi1_step
            peak = nstream / (np.sqrt(2.0 * np.pi) * sigma * (args.phi1_step * args.window_scale))
            peak_err = peak * np.hypot(nstream_err / max(nstream, 1e-6), sigma_err / max(sigma, 1e-6))

            rec.f_stream = f_stream
            rec.f_stream_err = min(0.2, math.sqrt(max(f_stream * (1 - f_stream), 0.0) / max(n_sig, 1)))
            rec.mu = mu
            rec.mu_err = mu_err
            rec.sigma = sigma
            rec.sigma_err = sigma_err
            rec.n_stream = nstream
            rec.n_stream_err = nstream_err
            rec.linear_density = linear_density
            rec.linear_density_err = linear_density_err
            rec.peak_surface_density = peak
            rec.peak_surface_density_err = peak_err
            rec.acc_frac = np.nan
        results.append(rec)

    # track polynomials and residuals
    tab = Table(rows=[asdict(r) for r in results])
    df = tab.to_pandas()
    phi1 = df["phi1_center"].to_numpy(dtype=float)
    success = df["success"].to_numpy(dtype=bool)
    cluster = df["cluster_bin"].to_numpy(dtype=bool)
    usable = success & (~cluster) & np.isfinite(df["mu"].to_numpy(dtype=float))

    tr_mask = usable & (phi1 < 0)
    ld_mask = usable & (phi1 > 0)
    track_poly_tr = None
    track_poly_ld = None
    if np.sum(tr_mask) >= 3:
        track_poly_tr = np.polyfit(phi1[tr_mask], df.loc[tr_mask, "mu"].to_numpy(dtype=float), 2).tolist()
        pred = np.polyval(track_poly_tr, phi1[tr_mask])
        df.loc[tr_mask, "track_poly"] = pred
        df.loc[tr_mask, "track_resid"] = df.loc[tr_mask, "mu"].to_numpy(dtype=float) - pred
    if np.sum(ld_mask) >= 3:
        track_poly_ld = np.polyfit(phi1[ld_mask], df.loc[ld_mask, "mu"].to_numpy(dtype=float), 2).tolist()
        pred = np.polyval(track_poly_ld, phi1[ld_mask])
        df.loc[ld_mask, "track_poly"] = pred
        df.loc[ld_mask, "track_resid"] = df.loc[ld_mask, "mu"].to_numpy(dtype=float) - pred

    tab = Table.from_pandas(df)
    profiles_fits = outdir / "pal5_step5a_profiles.fits"
    profiles_csv = outdir / "pal5_step5a_profiles.csv"
    tab.write(profiles_fits, overwrite=True)
    tab.write(profiles_csv, overwrite=True)

    # Summary
    usable_df = df[usable]
    sum_linear = float(np.sum(usable_df["linear_density"].to_numpy(dtype=float) * args.phi1_step)) if len(usable_df) else float("nan")
    max_width_leading = float(np.nanmax(df.loc[(usable & (phi1 >= 5.0) & (phi1 <= 8.0)), "sigma"].to_numpy(dtype=float))) if np.any(usable & (phi1 >= 5.0) & (phi1 <= 8.0)) else float("nan")
    max_width_trailing = float(np.nanmax(df.loc[(usable & (phi1 >= -15.0) & (phi1 <= -5.0)), "sigma"].to_numpy(dtype=float))) if np.any(usable & (phi1 >= -15.0) & (phi1 <= -5.0)) else float("nan")

    summary = {
        "signal": args.signal,
        "preproc": args.preproc,
        "step2_summary": args.step2_summary,
        "iso": args.iso,
        "mu_prior_file": mu_prior_file,
        "support_script": args.support_script,
        "sampler": "map",
        "eta_mode": "offstream_empirical_fixedbg",
        "n_input_signal": int(len(phi1_sig)),
        "n_control_total": int(len(ctl_phi1)),
        "n_parent_total": int(len(parent_phi1)),
        "phi1_min": args.phi1_min,
        "phi1_max": args.phi1_max,
        "phi1_step": args.phi1_step,
        "window_scale": args.window_scale,
        "window_width_phi1": args.phi1_step * args.window_scale,
        "fit_halfwidth": args.fit_halfwidth,
        "min_signal_stars": args.min_signal_stars,
        "n_bins": int(len(df)),
        "n_success": int(np.sum(success)),
        "n_success_excluding_cluster": int(np.sum(usable)),
        "track_poly_trailing": track_poly_tr,
        "track_poly_leading": track_poly_ld,
        "max_width_leading": max_width_leading,
        "max_width_trailing": max_width_trailing,
        "integrated_stream_stars_excluding_cluster": sum_linear,
        "notes": [
            "This step5a run uses an empirical background template learned from control sidebands.",
            "Background normalization is anchored using off-stream bins only.",
            "The parent-count map is retained only as a diagnostic proxy, not as a multiplicative area-amplitude term.",
            "MAP uncertainties are heuristic, not posterior-sampled."
        ],
    }
    summary_path = outdir / "pal5_step5a_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # QC plots
    # ------------------------------------------------------------------
    # 1) control sample log map
    p1_edges = np.arange(FULL_PHI1_MIN, FULL_PHI1_MAX + BIN_MAP_PHI1, BIN_MAP_PHI1)
    p2_edges = np.arange(FULL_PHI2_MIN, FULL_PHI2_MAX + BIN_MAP_PHI2, BIN_MAP_PHI2)
    D_ctl = histogram2d_density(ctl_phi1, ctl_phi2, p1_edges, p2_edges)
    D_parent = histogram2d_density(parent_phi1, parent_phi2, p1_edges, p2_edges)
    D_sig = histogram2d_density(phi1_sig, phi2_sig, p1_edges, p2_edges)

    def plot_log_map(D, title, outpng, xlim=None, ylim=None):
        Dp = np.array(D.T, copy=True)
        pos = Dp[Dp > 0]
        vmin = np.percentile(pos, 5) if pos.size else 1.0
        vmax = np.percentile(pos, 99.5) if pos.size else 10.0
        plt.figure(figsize=(9, 6))
        plt.imshow(
            Dp,
            origin="lower",
            aspect="auto",
            extent=[p1_edges[0], p1_edges[-1], p2_edges[0], p2_edges[-1]],
            norm=LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, max(vmin, 1e-2) * 1.01)),
            cmap="viridis",
        )
        plt.colorbar(label="counts / deg$^2$")
        plt.xlabel(r"$\phi_1$ [deg]")
        plt.ylabel(r"$\phi_2$ [deg]")
        plt.title(title)
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(outpng, dpi=180)
        plt.close()

    plot_log_map(D_ctl, "step 5a control sample (strict mag + z-locus + isochrone sidebands, log scale)", outdir / "qc_step5a_control_density_phi12_log.png", xlim=(-25, 20), ylim=(-5, 10))
    plot_log_map(D_parent, "step 5a parent-count proxy from strict-mag + z-locus parent (NOT pure coverage, log scale)", outdir / "qc_step5a_parent_proxy_phi12_log.png", xlim=(-20, 10), ylim=(-5, 10))
    plot_log_map(D_sig, "step 5a empirical-bg model: Pal 5 frame number density (log scale)", outdir / "qc_step5a_density_phi12_log.png", xlim=(-25, 20), ylim=(-5, 10))
    plot_log_map(D_sig, "step 5a empirical-bg model: local Pal 5 frame density (log scale)", outdir / "qc_step5a_density_phi12_local_log.png", xlim=(-20, 10), ylim=(-2.5, 2.5))

    # RA/Dec map from signal if available
    t_sig = Table.read(args.signal)
    if "RA" in t_sig.colnames and "DEC" in t_sig.colnames:
        ra = np.asarray(t_sig["RA"], dtype=float)
        dec = np.asarray(t_sig["DEC"], dtype=float)
        ra_edges = np.arange(np.floor(np.nanmin(ra) / 0.25) * 0.25, np.ceil(np.nanmax(ra) / 0.25) * 0.25 + 0.25, 0.25)
        dec_edges = np.arange(np.floor(np.nanmin(dec) / 0.25) * 0.25, np.ceil(np.nanmax(dec) / 0.25) * 0.25 + 0.25, 0.25)
        D_radec = histogram2d_density(ra, dec, ra_edges, dec_edges)
        Dp = np.array(D_radec.T, copy=True)
        pos = Dp[Dp > 0]
        vmin = np.percentile(pos, 5) if pos.size else 1.0
        vmax = np.percentile(pos, 99.5) if pos.size else 10.0
        plt.figure(figsize=(9, 7))
        plt.imshow(
            Dp, origin="lower", aspect="auto",
            extent=[ra_edges[0], ra_edges[-1], dec_edges[0], dec_edges[-1]],
            norm=LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, max(vmin, 1e-2) * 1.01)), cmap="viridis",
        )
        plt.xlabel("RA [deg]")
        plt.ylabel("Dec [deg]")
        plt.title("step 5a strict selected sample: RA-Dec number density (log scale)")
        plt.colorbar(label="counts / deg$^2$")
        plt.tight_layout()
        plt.savefig(outdir / "qc_step5a_density_radec_log.png", dpi=180)
        plt.close()

    # 2) template examples and local fits
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    fig2, axs2 = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    for ax, ax2, p1c in zip(axs.ravel(), axs2.ravel(), EXAMPLE_BIN_CENTERS):
        idx = np.argmin(np.abs(df["phi1_center"].to_numpy(dtype=float) - p1c))
        row = df.iloc[idx]
        phi1_c = float(row["phi1_center"])
        phi1_lo = float(row["phi1_lo"])
        phi1_hi = float(row["phi1_hi"])
        mu_prior = float(row["mu_prior_center"])
        fit_lo = float(row["fit_lo"])
        fit_hi = float(row["fit_hi"])

        m_sig = (phi1_sig >= phi1_lo) & (phi1_sig < phi1_hi)
        m_ctl = (ctl_phi1 >= phi1_lo) & (ctl_phi1 < phi1_hi)
        centers, counts, template = build_local_template(
            phi2_sig[m_sig], ctl_phi2[m_ctl], mu_prior, fit_lo, fit_hi, BIN_FIT,
            args.bg_exclude_half, args.off_inner, args.off_outer,
        )
        ax.plot(centers, template.bg_counts / max(np.mean(template.bg_counts), 1e-6), color="C1", lw=2, label="empirical bg")
        ax.axvline(mu_prior, color="C0", lw=1.5)
        ax.set_title(fr"$\phi_1 \approx {phi1_c:+.2f}^\circ$")
        ax.set_xlabel(r"$\phi_2$ [deg]")
        ax.set_ylabel("template amplitude")

        ax2.step(centers, counts, where="mid", color="C0", lw=1.5, label="data")
        ax2.plot(centers, template.bg_counts, color="C3", ls="--", lw=1.5, label="empirical bg (fixed)")
        if bool(row["success"]):
            nstream = float(row["n_stream"])
            mu = float(row["mu"])
            sigma = float(row["sigma"])
            lam = template.bg_counts + nstream * gaussian_pdf(centers, mu, sigma) * BIN_FIT
            ax2.plot(centers, lam, color="C1", lw=2, label="fit")
            ax2.axvline(mu, color="C0", lw=1.2)
        else:
            ax2.axvline(mu_prior, color="0.5", lw=1.2)
        ax2.set_title(fr"$\phi_1 \approx {phi1_c:+.2f}^\circ$")
        ax2.set_xlabel(r"$\phi_2$ [deg]")
        ax2.set_ylabel("counts / bin")
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("step 5a templates: off-stream anchored empirical background")
    fig.tight_layout()
    fig.savefig(outdir / "qc_step5a_template_examples.png", dpi=180)
    plt.close(fig)

    handles, labels = axs2.ravel()[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right")
    fig2.suptitle("step 5a local fits: fixed off-stream empirical background + Gaussian stream")
    fig2.tight_layout()
    fig2.savefig(outdir / "qc_step5a_example_local_fits.png", dpi=180)
    plt.close(fig2)

    # 3) profiles
    usable_mask = df["success"].to_numpy(dtype=bool) & (~df["cluster_bin"].to_numpy(dtype=bool))
    cluster_mask = df["cluster_bin"].to_numpy(dtype=bool)
    phi1 = df["phi1_center"].to_numpy(dtype=float)

    def prof_plot(ycol, yerr, ylabel, outpng):
        plt.figure(figsize=(10, 4.5))
        plt.errorbar(phi1[usable_mask], df.loc[usable_mask, ycol], yerr=df.loc[usable_mask, yerr], fmt="o", ms=5, capsize=3, label="usable bins")
        if np.any(cluster_mask):
            plt.errorbar(phi1[cluster_mask], df.loc[cluster_mask, ycol], yerr=df.loc[cluster_mask, yerr], fmt="s", ms=5, capsize=3, color="0.5", label="cluster bins")
        plt.xlabel(r"$\phi_1$ [deg]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpng, dpi=180)
        plt.close()

    prof_plot("linear_density", "linear_density_err", "stream stars / deg", outdir / "qc_step5a_linear_density.png")
    prof_plot("f_stream", "f_stream_err", "stream fraction in fit window", outdir / "qc_step5a_stream_fraction.png")
    prof_plot("mu", "mu_err", r"track $\mu(\phi_1)$ [deg]", outdir / "qc_step5a_track.png")
    prof_plot("sigma", "sigma_err", r"width $\sigma(\phi_1)$ [deg]", outdir / "qc_step5a_width.png")

    plt.figure(figsize=(10, 4.5))
    plt.errorbar(phi1[usable_mask], df.loc[usable_mask, "track_resid"], yerr=df.loc[usable_mask, "mu_err"], fmt="o", ms=5, capsize=3)
    plt.axhline(0.0, color="0.5")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("track residual [deg]")
    plt.tight_layout()
    plt.savefig(outdir / "qc_step5a_track_resid.png", dpi=180)
    plt.close()

    # 4) local strict-sample map with track overlay
    D_local = histogram2d_density(phi1_sig, phi2_sig, np.arange(args.phi1_min, args.phi1_max + BIN_MAP_PHI1, BIN_MAP_PHI1), np.arange(-2.5, 2.5 + BIN_MAP_PHI2, BIN_MAP_PHI2))
    xedges = np.arange(args.phi1_min, args.phi1_max + BIN_MAP_PHI1, BIN_MAP_PHI1)
    yedges = np.arange(-2.5, 2.5 + BIN_MAP_PHI2, BIN_MAP_PHI2)
    Dp = np.array(D_local.T, copy=True)
    pos = Dp[Dp > 0]
    vmin = np.percentile(pos, 5) if pos.size else 1.0
    vmax = np.percentile(pos, 99.5) if pos.size else 10.0
    plt.figure(figsize=(10, 5))
    plt.imshow(
        Dp, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, max(vmin, 1e-2) * 1.01)), cmap="viridis",
    )
    if np.any(usable_mask):
        plt.errorbar(df.loc[usable_mask, "phi1_center"], df.loc[usable_mask, "mu"], yerr=df.loc[usable_mask, "sigma"], fmt="-", color="C0", lw=2, label="baseline track")
        plt.plot(df.loc[usable_mask, "phi1_center"], df.loc[usable_mask, "mu"] + df.loc[usable_mask, "sigma"], ls="--", color="C1", lw=1.5)
        plt.plot(df.loc[usable_mask, "phi1_center"], df.loc[usable_mask, "mu"] - df.loc[usable_mask, "sigma"], ls="--", color="C1", lw=1.5, label=r"$\mu\pm\sigma$")
    if np.any(cluster_mask):
        plt.scatter(df.loc[cluster_mask, "phi1_center"], df.loc[cluster_mask, "mu_prior_center"], s=30, marker="s", color="0.5", label="cluster bins")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step 5a: strict-sample local Pal 5 density + baseline track")
    plt.legend(loc="upper right")
    plt.colorbar(label="counts / deg$^2$")
    plt.tight_layout()
    plt.savefig(outdir / "qc_step5a_local_map_with_track.png", dpi=180)
    plt.close()

    print(f"[done] wrote {profiles_csv}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
