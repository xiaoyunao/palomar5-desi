#!/usr/bin/env python3
"""
Pal 5 step 3: Bonaca-style 1D stream density modeling.

This script takes the strict member-selected sample from step 2 and fits,
in overlapping phi1 bins, the phi2 distribution with a single Gaussian stream
component plus a locally linear background.

The goal is to reproduce the baseline measurements of:
- stream track mu(phi1)
- stream width sigma(phi1)
- stream linear density lambda(phi1)

Notes
-----
1. This is the *strict baseline* analysis, intentionally close to Bonaca+2020.
2. The background model is a local linear shape in phi2, normalized over a
   finite fitting window. This is adequate for the strict baseline, but it is
   not the final background treatment for the deeper sample.
3. We use a two-pass fit:
   - pass 1: independent fits in each phi1 window with histogram-mode mu init
   - pass 2: refit using a smoothed pass-1 track as the mu prior center
4. No CMD probabilities are used here: this script operates on the already
   CMD-filtered member sample from step 2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.table import Table
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import expit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_INPUT = "step2_outputs/pal5_step2_strict_members.fits"
DEFAULT_OUTDIR = "step3_outputs"

PHI1_MIN = -20.0
PHI1_MAX = 10.0
PHI1_STEP = 0.75
WINDOW_SCALE = 1.5  # effective fit window in phi1 = 1.5 * step

PASS1_PHI2_MIN = -2.5
PASS1_PHI2_MAX = 2.5
PASS2_PHI2_HALFWIDTH = 1.75

MIN_STARS_PER_BIN = 60
HIST_BIN_PHI2 = 0.05

MU_PRIOR_SIGMA_PASS1 = 0.60
MU_PRIOR_SIGMA_PASS2 = 0.35

SIGMA_MIN = 0.03
SIGMA_MAX = 1.20

MAP_BIN = 0.10
PLOT_PHI1_MIN = -25.0
PLOT_PHI1_MAX = 20.0
PLOT_PHI2_MIN = -5.0
PLOT_PHI2_MAX = 10.0

EXAMPLE_BIN_CENTERS = [-13.0, -7.0, -3.0, 0.0, 3.0, 6.0]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return coef * np.exp(-0.5 * z * z)


def linear_bg_pdf(x: np.ndarray, raw_tilt: float, lo: float, hi: float) -> np.ndarray:
    """
    Locally linear background over [lo, hi], normalized to integrate to 1.

    We parameterize the shape as:
        p_bg(x) ∝ 1 + tilt * (x - mid)
    with tilt constrained so the density stays positive across the fit domain.
    """
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    # Keep strictly positive over the domain.
    tilt = 0.95 * np.tanh(raw_tilt) / max(half, 1e-6)
    y = 1.0 + tilt * (x - mid)
    # Because the domain is centered on mid, integral(y) = hi - lo.
    return y / (hi - lo)


def transform_bg_tilt(raw_tilt: float, lo: float, hi: float) -> float:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    _ = mid
    return 0.95 * np.tanh(raw_tilt) / max(half, 1e-6)


def transform_bg_tilt_err(raw_tilt: float, raw_err: float, lo: float, hi: float) -> float:
    half = 0.5 * (hi - lo)
    deriv = 0.95 * (1.0 - np.tanh(raw_tilt) ** 2) / max(half, 1e-6)
    return abs(deriv) * raw_err


def histogram_mode(x: np.ndarray, lo: float, hi: float, bin_size: float) -> float:
    if x.size == 0:
        return 0.0
    edges = np.arange(lo, hi + bin_size, bin_size)
    if len(edges) < 2:
        return float(np.median(x))
    hist, edges = np.histogram(x, bins=edges)
    if hist.sum() == 0:
        return float(np.median(x))
    idx = int(np.argmax(hist))
    return 0.5 * (edges[idx] + edges[idx + 1])


def interp_fill(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.array(y, dtype=float)
    m = np.isfinite(out)
    if m.sum() == 0:
        return np.zeros_like(out)
    if m.sum() == 1:
        out[~m] = out[m][0]
        return out
    out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def smooth_track_by_arm(phi1: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu_filled = interp_fill(phi1, mu)
    out = np.array(mu_filled, copy=True)

    for arm_mask in [phi1 <= 0.0, phi1 >= 0.0]:
        idx = np.where(arm_mask)[0]
        if len(idx) < 5:
            continue
        sub = mu_filled[idx]
        # Odd window length <= number of samples.
        win = min(9, len(sub) if len(sub) % 2 == 1 else len(sub) - 1)
        if win < 5:
            continue
        out[idx] = savgol_filter(sub, window_length=win, polyorder=2, mode="interp")
    return out


def polyfit_arm(phi1: np.ndarray, mu: np.ndarray, mu_err: np.ndarray, arm_mask: np.ndarray) -> Optional[np.ndarray]:
    m = arm_mask & np.isfinite(phi1) & np.isfinite(mu)
    if m.sum() < 4:
        return None
    x = phi1[m]
    y = mu[m]
    if np.isfinite(mu_err[m]).sum() >= 3:
        w = 1.0 / np.clip(mu_err[m], 0.03, None)
    else:
        w = None
    deg = 2 if len(x) >= 3 else 1
    try:
        return np.polyfit(x, y, deg=deg, w=w)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------
@dataclass
class BinFitResult:
    phi1_center: float
    phi1_lo: float
    phi1_hi: float
    fit_lo: float
    fit_hi: float
    n_stars: int
    mu_init: float
    success: bool
    status: int
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
    neglogpost: float = np.nan


def neg_log_posterior(
    theta: np.ndarray,
    phi2: np.ndarray,
    mu_prior_center: float,
    mu_prior_sigma: float,
    lo: float,
    hi: float,
) -> float:
    logit_f, mu, log_sigma, raw_tilt = theta
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma < SIGMA_MIN or sigma > SIGMA_MAX:
        return np.inf
    if not np.isfinite(mu) or mu < lo - 1.0 or mu > hi + 1.0:
        return np.inf

    f_stream = float(expit(logit_f))
    p_stream = gaussian_pdf(phi2, mu, sigma)
    p_bg = linear_bg_pdf(phi2, raw_tilt, lo, hi)
    p = f_stream * p_stream + (1.0 - f_stream) * p_bg
    if np.any(~np.isfinite(p)):
        return np.inf
    if np.any(p <= 0.0):
        return np.inf

    nlp = -np.sum(np.log(np.clip(p, 1e-300, None)))
    # Gaussian prior on mu to keep the fit close to the initialized ridge.
    nlp += 0.5 * ((mu - mu_prior_center) / mu_prior_sigma) ** 2
    return float(nlp)


def fit_single_bin(
    phi2: np.ndarray,
    phi1_center: float,
    phi1_lo: float,
    phi1_hi: float,
    mu_init: float,
    mu_prior_center: float,
    mu_prior_sigma: float,
    fit_lo: float,
    fit_hi: float,
    window_width_phi1: float,
) -> BinFitResult:
    result = BinFitResult(
        phi1_center=phi1_center,
        phi1_lo=phi1_lo,
        phi1_hi=phi1_hi,
        fit_lo=fit_lo,
        fit_hi=fit_hi,
        n_stars=int(len(phi2)),
        mu_init=float(mu_init),
        success=False,
        status=-1,
        message="not_fitted",
        mu_prior_center=float(mu_prior_center),
        mu_prior_sigma=float(mu_prior_sigma),
    )

    if len(phi2) < MIN_STARS_PER_BIN:
        result.message = "too_few_stars"
        return result

    starts: List[np.ndarray] = []
    for f0 in [0.10, 0.20, 0.35, 0.50]:
        for s0 in [0.08, 0.12, 0.18, 0.28]:
            for tilt0 in [-0.5, 0.0, 0.5]:
                starts.append(np.array([
                    math.log(f0 / (1.0 - f0)),
                    mu_init,
                    math.log(s0),
                    tilt0,
                ], dtype=float))

    best = None
    best_fun = np.inf
    for x0 in starts:
        try:
            res = minimize(
                neg_log_posterior,
                x0=x0,
                args=(phi2, mu_prior_center, mu_prior_sigma, fit_lo, fit_hi),
                method="BFGS",
                options={"gtol": 1e-5, "maxiter": 500},
            )
        except Exception:
            continue
        if np.isfinite(res.fun) and (res.fun < best_fun):
            best = res
            best_fun = float(res.fun)

    if best is None:
        result.message = "optimizer_failed"
        return result

    theta = np.asarray(best.x, dtype=float)
    logit_f, mu, log_sigma, raw_tilt = theta
    sigma = float(np.exp(log_sigma))
    f_stream = float(expit(logit_f))
    bg_tilt = transform_bg_tilt(raw_tilt, fit_lo, fit_hi)

    # Approximate covariance from BFGS inverse Hessian.
    mu_err = np.nan
    sigma_err = np.nan
    f_err = np.nan
    bg_tilt_err = np.nan
    try:
        cov = np.asarray(best.hess_inv, dtype=float)
        if cov.shape == (4, 4) and np.all(np.isfinite(cov)):
            diag = np.clip(np.diag(cov), 0.0, None)
            errs = np.sqrt(diag)
            mu_err = float(errs[1])
            sigma_err = float(sigma * errs[2])
            f_err = float(f_stream * (1.0 - f_stream) * errs[0])
            bg_tilt_err = float(transform_bg_tilt_err(raw_tilt, errs[3], fit_lo, fit_hi))
    except Exception:
        pass

    n_stream = f_stream * len(phi2)
    n_stream_err = f_err * len(phi2) if np.isfinite(f_err) else np.nan
    linear_density = n_stream / window_width_phi1
    linear_density_err = n_stream_err / window_width_phi1 if np.isfinite(n_stream_err) else np.nan

    peak_surface_density = n_stream / (window_width_phi1 * np.sqrt(2.0 * np.pi) * sigma)
    if np.isfinite(n_stream_err) and np.isfinite(sigma_err):
        rel2 = 0.0
        if n_stream > 0:
            rel2 += (n_stream_err / n_stream) ** 2
        if sigma > 0:
            rel2 += (sigma_err / sigma) ** 2
        peak_surface_density_err = abs(peak_surface_density) * np.sqrt(rel2)
    else:
        peak_surface_density_err = np.nan

    result.success = bool(best.success)
    result.status = int(getattr(best, "status", -1))
    result.message = str(getattr(best, "message", "ok"))
    result.f_stream = f_stream
    result.f_stream_err = f_err
    result.mu = mu
    result.mu_err = mu_err
    result.sigma = sigma
    result.sigma_err = sigma_err
    result.bg_tilt = bg_tilt
    result.bg_tilt_err = bg_tilt_err
    result.n_stream = n_stream
    result.n_stream_err = n_stream_err
    result.linear_density = linear_density
    result.linear_density_err = linear_density_err
    result.peak_surface_density = peak_surface_density
    result.peak_surface_density_err = peak_surface_density_err
    result.neglogpost = float(best.fun)
    return result


def run_fit_pass(
    phi1: np.ndarray,
    phi2: np.ndarray,
    centers: np.ndarray,
    window_width_phi1: float,
    pass_name: str,
    pass1_phi2_min: float,
    pass1_phi2_max: float,
    pass2_halfwidth: float,
    mu_prior_centers: Optional[np.ndarray] = None,
    mu_prior_sigma: float = MU_PRIOR_SIGMA_PASS1,
) -> List[BinFitResult]:
    results: List[BinFitResult] = []
    half_w = 0.5 * window_width_phi1

    for i, c in enumerate(centers):
        phi1_lo = c - half_w
        phi1_hi = c + half_w
        in_phi1 = (phi1 >= phi1_lo) & (phi1 < phi1_hi)
        if mu_prior_centers is None:
            fit_lo = pass1_phi2_min
            fit_hi = pass1_phi2_max
            mu_guess_center = 0.0
        else:
            mu_guess_center = float(mu_prior_centers[i])
            fit_lo = mu_guess_center - pass2_halfwidth
            fit_hi = mu_guess_center + pass2_halfwidth

        in_phi2 = (phi2 >= fit_lo) & (phi2 <= fit_hi)
        m = in_phi1 & in_phi2 & np.isfinite(phi1) & np.isfinite(phi2)
        phi2_sub = np.asarray(phi2[m], dtype=float)

        if mu_prior_centers is None:
            mu_init = histogram_mode(phi2_sub, fit_lo, fit_hi, HIST_BIN_PHI2)
            mu_prior_center = mu_init
        else:
            mu_init = mu_guess_center
            mu_prior_center = mu_guess_center

        res = fit_single_bin(
            phi2=phi2_sub,
            phi1_center=float(c),
            phi1_lo=float(phi1_lo),
            phi1_hi=float(phi1_hi),
            mu_init=float(mu_init),
            mu_prior_center=float(mu_prior_center),
            mu_prior_sigma=float(mu_prior_sigma),
            fit_lo=float(fit_lo),
            fit_hi=float(fit_hi),
            window_width_phi1=window_width_phi1,
        )
        results.append(res)

        if (i % 10) == 0 or (i == len(centers) - 1):
            print(f"[{pass_name}] bin {i+1:02d}/{len(centers)}  phi1={c:+5.2f}  N={res.n_stars:5d}  success={res.success}")

    return results


# -----------------------------------------------------------------------------
# Output conversion
# -----------------------------------------------------------------------------

def results_to_table(pass1: List[BinFitResult], final: List[BinFitResult]) -> Table:
    if len(pass1) != len(final):
        raise ValueError("pass1/final result lengths do not match")

    rows: Dict[str, List] = {
        "phi1_center": [], "phi1_lo": [], "phi1_hi": [],
        "fit_lo": [], "fit_hi": [], "n_stars": [],
        "mu_init_pass1": [], "mu_pass1": [], "sigma_pass1": [], "f_stream_pass1": [],
        "mu": [], "mu_err": [], "sigma": [], "sigma_err": [],
        "f_stream": [], "f_stream_err": [],
        "bg_tilt": [], "bg_tilt_err": [],
        "n_stream": [], "n_stream_err": [],
        "linear_density": [], "linear_density_err": [],
        "peak_surface_density": [], "peak_surface_density_err": [],
        "mu_prior_center": [], "mu_prior_sigma": [],
        "neglogpost": [], "success": [], "status": [], "message": [],
    }

    for r1, r2 in zip(pass1, final):
        rows["phi1_center"].append(r2.phi1_center)
        rows["phi1_lo"].append(r2.phi1_lo)
        rows["phi1_hi"].append(r2.phi1_hi)
        rows["fit_lo"].append(r2.fit_lo)
        rows["fit_hi"].append(r2.fit_hi)
        rows["n_stars"].append(r2.n_stars)
        rows["mu_init_pass1"].append(r1.mu_init)
        rows["mu_pass1"].append(r1.mu)
        rows["sigma_pass1"].append(r1.sigma)
        rows["f_stream_pass1"].append(r1.f_stream)
        rows["mu"].append(r2.mu)
        rows["mu_err"].append(r2.mu_err)
        rows["sigma"].append(r2.sigma)
        rows["sigma_err"].append(r2.sigma_err)
        rows["f_stream"].append(r2.f_stream)
        rows["f_stream_err"].append(r2.f_stream_err)
        rows["bg_tilt"].append(r2.bg_tilt)
        rows["bg_tilt_err"].append(r2.bg_tilt_err)
        rows["n_stream"].append(r2.n_stream)
        rows["n_stream_err"].append(r2.n_stream_err)
        rows["linear_density"].append(r2.linear_density)
        rows["linear_density_err"].append(r2.linear_density_err)
        rows["peak_surface_density"].append(r2.peak_surface_density)
        rows["peak_surface_density_err"].append(r2.peak_surface_density_err)
        rows["mu_prior_center"].append(r2.mu_prior_center)
        rows["mu_prior_sigma"].append(r2.mu_prior_sigma)
        rows["neglogpost"].append(r2.neglogpost)
        rows["success"].append(r2.success)
        rows["status"].append(r2.status)
        rows["message"].append(r2.message)

    return Table(rows)


def append_track_polynomials(tab: Table) -> Dict[str, Optional[np.ndarray]]:
    phi1 = np.asarray(tab["phi1_center"], dtype=float)
    mu = np.asarray(tab["mu"], dtype=float)
    mu_err = np.asarray(tab["mu_err"], dtype=float)
    success = np.asarray(tab["success"], dtype=bool)

    lead_mask = success & np.isfinite(mu) & (phi1 >= 0.75)
    trail_mask = success & np.isfinite(mu) & (phi1 <= -0.75)

    coeff_trail = polyfit_arm(phi1, mu, mu_err, trail_mask)
    coeff_lead = polyfit_arm(phi1, mu, mu_err, lead_mask)

    poly_val = np.full(len(tab), np.nan)
    resid = np.full(len(tab), np.nan)

    if coeff_trail is not None:
        m = phi1 <= 0.0
        poly_val[m] = np.polyval(coeff_trail, phi1[m])
    if coeff_lead is not None:
        m = phi1 >= 0.0
        poly_val[m] = np.polyval(coeff_lead, phi1[m])
    good = np.isfinite(mu) & np.isfinite(poly_val)
    resid[good] = mu[good] - poly_val[good]

    tab["track_poly"] = poly_val
    tab["track_resid"] = resid
    return {"trailing": coeff_trail, "leading": coeff_lead}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_density_map_with_track(tab_members: Table, tab_fit: Table, out_png: Path, full_frame: bool = True) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)

    if full_frame:
        xlo, xhi = PLOT_PHI1_MIN, PLOT_PHI1_MAX
        ylo, yhi = PLOT_PHI2_MIN, PLOT_PHI2_MAX
        title = "strict selected sample: Pal 5 frame number density"
    else:
        xlo, xhi = PHI1_MIN, PHI1_MAX
        ylo, yhi = PASS1_PHI2_MIN, PASS1_PHI2_MAX
        title = "strict selected sample: local Pal 5 frame density"

    xedges = np.arange(xlo, xhi + MAP_BIN, MAP_BIN)
    yedges = np.arange(ylo, yhi + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(phi1, phi2, bins=[xedges, yedges])
    area = MAP_BIN * MAP_BIN
    D = H.T / area

    plt.figure(figsize=(11, 6.5))
    plt.imshow(
        D,
        origin="lower",
        aspect="auto",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        interpolation="nearest",
    )
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")

    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    mu = np.asarray(tab_fit["mu"], dtype=float)
    sig = np.asarray(tab_fit["sigma"], dtype=float)
    ok = np.isfinite(mu) & np.isfinite(sig)
    if ok.any():
        plt.plot(x[ok], mu[ok], lw=2.0)
        plt.plot(x[ok], mu[ok] + sig[ok], "--", lw=1.2)
        plt.plot(x[ok], mu[ok] - sig[ok], "--", lw=1.2)

    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_radec_map(tab_members: Table, out_png: Path) -> None:
    ra = np.asarray(tab_members["RA"], dtype=float)
    dec = np.asarray(tab_members["DEC"], dtype=float)
    xedges = np.arange(np.floor(ra.min()), np.ceil(ra.max()) + MAP_BIN, MAP_BIN)
    yedges = np.arange(np.floor(dec.min()), np.ceil(dec.max()) + MAP_BIN, MAP_BIN)
    H, xe, ye = np.histogram2d(ra, dec, bins=[xedges, yedges])
    area = MAP_BIN * MAP_BIN
    D = H.T / area

    plt.figure(figsize=(10.5, 6.5))
    plt.imshow(
        D,
        origin="lower",
        aspect="auto",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        interpolation="nearest",
    )
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
    mu = np.asarray(tab_fit["mu"], dtype=float)
    mu_err = np.asarray(tab_fit["mu_err"], dtype=float)

    plt.figure(figsize=(10, 4.8))
    plt.errorbar(x, mu, yerr=mu_err, fmt="o", ms=3.5, lw=1.0, capsize=2, alpha=0.9, label="fit")

    for arm, coeff in poly_coeffs.items():
        if coeff is None:
            continue
        if arm == "trailing":
            xx = np.linspace(PHI1_MIN, 0.0, 200)
        else:
            xx = np.linspace(0.0, PHI1_MAX, 200)
        yy = np.polyval(coeff, xx)
        plt.plot(xx, yy, "--", lw=1.5, label=f"{arm} quadratic")

    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"track $\mu(\phi_1)$ [deg]")
    plt.title("Bonaca-style 1D model: stream track")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_track_resid_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    r = np.asarray(tab_fit["track_resid"], dtype=float)
    rerr = np.asarray(tab_fit["mu_err"], dtype=float)

    plt.figure(figsize=(10, 4.2))
    plt.axhline(0.0, color="0.5", lw=1.0)
    plt.errorbar(x, r, yerr=rerr, fmt="o", ms=3.5, lw=1.0, capsize=2)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("track residual [deg]")
    plt.title("Bonaca-style 1D model: track residual from quadratic arm fits")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_width_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    y = np.asarray(tab_fit["sigma"], dtype=float)
    yerr = np.asarray(tab_fit["sigma_err"], dtype=float)

    plt.figure(figsize=(10, 4.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o", ms=3.5, lw=1.0, capsize=2)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"width $\sigma(\phi_1)$ [deg]")
    plt.title("Bonaca-style 1D model: stream width")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_density_profile_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    y = np.asarray(tab_fit["linear_density"], dtype=float)
    yerr = np.asarray(tab_fit["linear_density_err"], dtype=float)

    plt.figure(figsize=(10, 4.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o", ms=3.5, lw=1.0, capsize=2, label="linear density")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("stream stars / deg")
    plt.title("Bonaca-style 1D model: linear density profile")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_fraction_plot(tab_fit: Table, out_png: Path) -> None:
    x = np.asarray(tab_fit["phi1_center"], dtype=float)
    y = np.asarray(tab_fit["f_stream"], dtype=float)
    yerr = np.asarray(tab_fit["f_stream_err"], dtype=float)

    plt.figure(figsize=(10, 4.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o", ms=3.5, lw=1.0, capsize=2)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("stream fraction in fit window")
    plt.title("Bonaca-style 1D model: fitted stream fraction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_example_fits(tab_members: Table, tab_fit: Table, out_png: Path) -> None:
    phi1 = np.asarray(tab_members["PHI1"], dtype=float)
    phi2 = np.asarray(tab_members["PHI2"], dtype=float)
    window_width_phi1 = WINDOW_SCALE * PHI1_STEP
    half_w = 0.5 * window_width_phi1

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()

    fit_centers = np.asarray(tab_fit["phi1_center"], dtype=float)

    for ax, target in zip(axes, EXAMPLE_BIN_CENTERS):
        if len(fit_centers) == 0:
            ax.set_axis_off()
            continue
        idx = int(np.argmin(np.abs(fit_centers - target)))
        row = tab_fit[idx]
        c = float(row["phi1_center"])
        mu = float(row["mu"])
        sigma = float(row["sigma"])
        f_stream = float(row["f_stream"])
        bg_tilt = float(row["bg_tilt"])
        fit_lo = float(row["fit_lo"])
        fit_hi = float(row["fit_hi"])

        m = (phi1 >= c - half_w) & (phi1 < c + half_w) & (phi2 >= fit_lo) & (phi2 <= fit_hi)
        y = np.asarray(phi2[m], dtype=float)

        bins = np.arange(fit_lo, fit_hi + 0.05, 0.05)
        hist, edges = np.histogram(y, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax.step(mids, hist, where="mid", lw=1.2, label="data")

        if np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(f_stream):
            # Convert the fitted pdf into expected counts per histogram bin.
            raw_tilt = np.arctanh(np.clip(bg_tilt * (0.5 * (fit_hi - fit_lo)) / 0.95, -0.999, 0.999))
            p_stream = gaussian_pdf(mids, mu, sigma)
            p_bg = linear_bg_pdf(mids, raw_tilt, fit_lo, fit_hi)
            model_pdf = f_stream * p_stream + (1.0 - f_stream) * p_bg
            model_counts = len(y) * model_pdf * np.diff(edges)[0]
            ax.plot(mids, model_counts, lw=2.0, label="fit")
            ax.axvline(mu, lw=1.0)

        ax.set_title(rf"$\phi_1 \approx {c:+.2f}^\circ$")
        ax.set_xlabel(r"$\phi_2$ [deg]")
        if ax in axes[::3]:
            ax.set_ylabel("counts / bin")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Example Bonaca-style local fits in overlapping $\phi_1$ bins")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 step 3: Bonaca-style 1D stream density model")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Step 2 strict member FITS catalog")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument("--phi1-min", type=float, default=PHI1_MIN)
    p.add_argument("--phi1-max", type=float, default=PHI1_MAX)
    p.add_argument("--phi1-step", type=float, default=PHI1_STEP)
    p.add_argument("--window-scale", type=float, default=WINDOW_SCALE)
    p.add_argument("--pass1-phi2-min", type=float, default=PASS1_PHI2_MIN)
    p.add_argument("--pass1-phi2-max", type=float, default=PASS1_PHI2_MAX)
    p.add_argument("--pass2-phi2-halfwidth", type=float, default=PASS2_PHI2_HALFWIDTH)
    p.add_argument("--min-stars", type=int, default=MIN_STARS_PER_BIN)
    return p.parse_args()


def main() -> None:
    global PHI1_MIN, PHI1_MAX, PHI1_STEP, WINDOW_SCALE
    global PASS1_PHI2_MIN, PASS1_PHI2_MAX, PASS2_PHI2_HALFWIDTH, MIN_STARS_PER_BIN

    args = parse_args()
    PHI1_MIN = args.phi1_min
    PHI1_MAX = args.phi1_max
    PHI1_STEP = args.phi1_step
    WINDOW_SCALE = args.window_scale
    PASS1_PHI2_MIN = args.pass1_phi2_min
    PASS1_PHI2_MAX = args.pass1_phi2_max
    PASS2_PHI2_HALFWIDTH = args.pass2_phi2_halfwidth
    MIN_STARS_PER_BIN = args.min_stars

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"[read] {args.input}")
    tab = Table.read(args.input)
    required = ["PHI1", "PHI2", "RA", "DEC"]
    for col in required:
        if col not in tab.colnames:
            raise KeyError(f"Missing required column: {col}")

    phi1 = np.asarray(tab["PHI1"], dtype=float)
    phi2 = np.asarray(tab["PHI2"], dtype=float)
    m = np.isfinite(phi1) & np.isfinite(phi2)
    tab = tab[m]
    phi1 = phi1[m]
    phi2 = phi2[m]
    print(f"[info] finite PHI1/PHI2 rows: {len(tab):,}")

    centers = np.arange(PHI1_MIN, PHI1_MAX + 0.5 * PHI1_STEP, PHI1_STEP)
    window_width_phi1 = WINDOW_SCALE * PHI1_STEP
    print(f"[info] centers: {len(centers)} from {centers[0]:+.2f} to {centers[-1]:+.2f}")
    print(f"[info] phi1 fit window width = {window_width_phi1:.3f} deg")

    # Pass 1: independent local fits using histogram-mode initialization.
    pass1 = run_fit_pass(
        phi1=phi1,
        phi2=phi2,
        centers=centers,
        window_width_phi1=window_width_phi1,
        pass_name="pass1",
        pass1_phi2_min=PASS1_PHI2_MIN,
        pass1_phi2_max=PASS1_PHI2_MAX,
        pass2_halfwidth=PASS2_PHI2_HALFWIDTH,
        mu_prior_centers=None,
        mu_prior_sigma=MU_PRIOR_SIGMA_PASS1,
    )

    mu1 = np.array([r.mu if np.isfinite(r.mu) else r.mu_init for r in pass1], dtype=float)
    mu_prior = smooth_track_by_arm(centers, mu1)

    np.savetxt(outdir / "pal5_step3_pass1_prior_track.txt", np.c_[centers, mu1, mu_prior],
               header="phi1_center  mu_pass1_or_init  mu_smoothed_prior")

    # Pass 2: refit using smoothed pass-1 ridge as prior center and local phi2 window.
    final = run_fit_pass(
        phi1=phi1,
        phi2=phi2,
        centers=centers,
        window_width_phi1=window_width_phi1,
        pass_name="pass2",
        pass1_phi2_min=PASS1_PHI2_MIN,
        pass1_phi2_max=PASS1_PHI2_MAX,
        pass2_halfwidth=PASS2_PHI2_HALFWIDTH,
        mu_prior_centers=mu_prior,
        mu_prior_sigma=MU_PRIOR_SIGMA_PASS2,
    )

    tab_fit = results_to_table(pass1, final)
    poly_coeffs = append_track_polynomials(tab_fit)

    profile_fits = outdir / "pal5_step3_profiles.fits"
    profile_csv = outdir / "pal5_step3_profiles.csv"
    tab_fit.write(profile_fits, overwrite=True)
    tab_fit.write(profile_csv, format="ascii.csv", overwrite=True)
    print(f"[write] {profile_fits}")
    print(f"[write] {profile_csv}")

    summary = {
        "input": str(args.input),
        "n_input": int(len(tab)),
        "phi1_min": PHI1_MIN,
        "phi1_max": PHI1_MAX,
        "phi1_step": PHI1_STEP,
        "window_scale": WINDOW_SCALE,
        "window_width_phi1": window_width_phi1,
        "pass1_phi2_range": [PASS1_PHI2_MIN, PASS1_PHI2_MAX],
        "pass2_phi2_halfwidth": PASS2_PHI2_HALFWIDTH,
        "min_stars_per_bin": MIN_STARS_PER_BIN,
        "n_bins": int(len(centers)),
        "n_success": int(np.sum(np.asarray(tab_fit["success"], dtype=bool))),
        "track_poly_trailing": poly_coeffs["trailing"].tolist() if poly_coeffs["trailing"] is not None else None,
        "track_poly_leading": poly_coeffs["leading"].tolist() if poly_coeffs["leading"] is not None else None,
        "max_width_leading": float(np.nanmax(np.asarray(tab_fit["sigma"], dtype=float)[np.asarray(tab_fit["phi1_center"], dtype=float) > 0])) if np.any(np.asarray(tab_fit["phi1_center"], dtype=float) > 0) else None,
        "max_width_trailing": float(np.nanmax(np.asarray(tab_fit["sigma"], dtype=float)[np.asarray(tab_fit["phi1_center"], dtype=float) < 0])) if np.any(np.asarray(tab_fit["phi1_center"], dtype=float) < 0) else None,
    }
    with open(outdir / "pal5_step3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {outdir / 'pal5_step3_summary.json'}")

    # Plots
    save_density_map_with_track(tab, tab_fit, outdir / "qc_step3_density_phi12.png", full_frame=True)
    save_density_map_with_track(tab, tab_fit, outdir / "qc_step3_density_phi12_local.png", full_frame=False)
    save_radec_map(tab, outdir / "qc_step3_density_radec.png")
    save_track_plot(tab_fit, poly_coeffs, outdir / "qc_step3_track.png")
    save_track_resid_plot(tab_fit, outdir / "qc_step3_track_resid.png")
    save_width_plot(tab_fit, outdir / "qc_step3_width.png")
    save_density_profile_plot(tab_fit, outdir / "qc_step3_linear_density.png")
    save_fraction_plot(tab_fit, outdir / "qc_step3_stream_fraction.png")
    save_example_fits(tab, tab_fit, outdir / "qc_step3_example_local_fits.png")
    print(f"[done] outputs written to {outdir}")


if __name__ == "__main__":
    main()
