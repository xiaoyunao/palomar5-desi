#!/usr/bin/env python3
"""
Pal 5 step 4b: MSTO-weighted refined distance-gradient selection.

What is new relative to step 4?
--------------------------------
This version keeps the same overall Bonaca-style logic — coarse distance anchors
in 2-degree steps, smooth DM(phi1) interpolation, then a refined isochrone
selection — but changes *how the anchors are scored*:

1) Anchor scoring is restricted to the distance-sensitive MSTO / upper-MS region.
2) A blue residual contaminant sequence is explicitly vetoed / downweighted.
3) Raw 2-degree anchors are smoothed with a more robust arm-wise procedure.
4) QC CMD panels are still plotted over the full 16 < g0 < 24 range so that the
   user can visually inspect all contaminating structures.

The goal is to get closer to the practical spirit of Bonaca+2020 Section 2:
use coarse, high-confidence, low-free-parameter distance anchors driven mainly by
where the Pal 5 MSTO lies, then interpolate between them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

try:
    from pal5_step4_refined_dm_selection import (
        DEFAULT_PREPROC,
        DEFAULT_ISO,
        DEFAULT_STEP2_SUMMARY,
        DEFAULT_MU_PRIOR,
        RA_MIN,
        RA_MAX,
        DEC_MIN,
        DEC_MAX,
        PHI1_MIN,
        PHI1_MAX,
        PHI2_MIN,
        PHI2_MAX,
        STRICT_GR_MIN,
        STRICT_GR_MAX,
        ZLOCUS_GR_MAX,
        ZLOCUS_TOL,
        CMD_W0,
        CMD_W_SLOPE,
        CMD_W_MIN,
        CMD_W_MAX,
        CMD_W_REF,
        G_REF_TILT,
        ANCHOR_PHI1_MIN,
        ANCHOR_PHI1_MAX,
        ANCHOR_STEP,
        ANCHOR_WINDOW_HALF,
        ON_HALFWIDTH,
        OFF_INNER,
        OFF_OUTER,
        DM_SCAN_HALF,
        DM_SCAN_STEP,
        CHUNK_SIZE_DEFAULT,
        BIN_DEG_RADEC,
        BIN_DEG_PHI1,
        BIN_DEG_PHI2,
        Step2Alignment,
        AnchorFit,
        ensure_dir,
        merge_fits_list,
        finite_grz,
        zlocus_residual,
        cmd_half_width,
        read_parsec_like_isochrone,
        IsoInterp,
        load_step2_summary,
        read_mu_prior,
        build_zparent_sample,
        plot_density_map,
        plot_dm_track,
        plot_local_compare,
    )
except ImportError as e:
    raise SystemExit(
        "Could not import helpers from pal5_step4_refined_dm_selection.py. "
        "Place this step4b script in the same project root as the earlier step4 script."
    ) from e


DEFAULT_OUTDIR = "step4b_outputs"

# -----------------------------------------------------------------------------
# Anchor-scoring config: narrowed to the MSTO / upper main sequence.
# -----------------------------------------------------------------------------
SCORE_GMIN = 19.8
SCORE_GMAX = 21.7
SCORE_MODEL_GR_MIN = 0.12
SCORE_MODEL_GR_MAX = 0.58
SCORE_SIGMA_SCALE = 1.10

# Explicit veto for the blue residual sequence seen in several segments.
BLUE_VETO_GR_MAX = 0.15
BLUE_VETO_GMIN = 21.5

# Robust smoothing for raw anchors.
SMOOTH_KERNEL = 3
CLIP_MAX_ABS_DM = 0.12
CLIP_SIGMA = 2.5
CLIP_MIN_POINTS_PER_ARM = 4

# Plotting range for the QC CMDs.
QC_GMIN = 16.0
QC_GMAX = 24.0
QC_GR_MIN = -0.35
QC_GR_MAX = 1.05
QC_BIN_C = 0.02
QC_BIN_G = 0.05

# Segment diagnostics.
MIN_ON_STARS = 50
MIN_OFF_STARS = 80
MIN_LOCAL_STARS = 200
DM_ERR_FLOOR = 0.5 * DM_SCAN_STEP


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def msto_mag_weight(g: np.ndarray) -> np.ndarray:
    """Distance-sensitive weight: emphasize MSTO / upper MS, suppress faint lower MS."""
    g = np.asarray(g, dtype=float)
    w = np.zeros_like(g, dtype=float)
    core = (g >= SCORE_GMIN) & (g <= SCORE_GMAX)
    if np.any(core):
        # Primary bump around the MSTO and a weaker shoulder into the upper main sequence.
        w_core = np.exp(-0.5 * ((g[core] - 20.45) / 0.38) ** 2)
        w_shoulder = 0.45 * np.exp(-0.5 * ((g[core] - 21.10) / 0.28) ** 2)
        w[core] = w_core + w_shoulder
    return w


def blue_residual_veto(g: np.ndarray, gr: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    gr = np.asarray(gr, dtype=float)
    return (gr < BLUE_VETO_GR_MAX) & (g > BLUE_VETO_GMIN)


def score_dm_curve_msto(
    g_on: np.ndarray,
    gr_on: np.ndarray,
    g_off: np.ndarray,
    gr_off: np.ndarray,
    dm_grid: np.ndarray,
    iso_interp: IsoInterp,
    bg_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score coarse DM anchors using only the distance-sensitive part of the CMD.

    This intentionally does *not* use the full strict-selection lower main sequence.
    Instead it focuses on the MSTO / upper MS, where the turnoff location carries
    the most distance information and where the blue residual sequence can be
    explicitly downweighted.
    """
    scores = np.full(dm_grid.shape, np.nan, dtype=float)
    snrs = np.full(dm_grid.shape, np.nan, dtype=float)

    sig_on = np.clip(SCORE_SIGMA_SCALE * cmd_half_width(g_on), 0.05, 0.16)
    sig_off = np.clip(SCORE_SIGMA_SCALE * cmd_half_width(g_off), 0.05, 0.16)
    wmag_on = msto_mag_weight(g_on)
    wmag_off = msto_mag_weight(g_off)

    if not np.any(wmag_on > 0) or not np.any(wmag_off > 0):
        return scores, snrs

    veto_on = blue_residual_veto(g_on, gr_on)
    veto_off = blue_residual_veto(g_off, gr_off)

    for i, dm in enumerate(dm_grid):
        c_on = iso_interp.gr_color(g_on, np.full_like(g_on, dm))
        c_off = iso_interp.gr_color(g_off, np.full_like(g_off, dm))

        ok_on = np.isfinite(c_on) & (c_on >= SCORE_MODEL_GR_MIN) & (c_on <= SCORE_MODEL_GR_MAX)
        ok_off = np.isfinite(c_off) & (c_off >= SCORE_MODEL_GR_MIN) & (c_off <= SCORE_MODEL_GR_MAX)
        if (not np.any(ok_on)) or (not np.any(ok_off)):
            continue

        d_on = gr_on[ok_on] - c_on[ok_on]
        d_off = gr_off[ok_off] - c_off[ok_off]

        w_on = np.exp(-0.5 * (d_on / sig_on[ok_on]) ** 2) * wmag_on[ok_on]
        w_off = np.exp(-0.5 * (d_off / sig_off[ok_off]) ** 2) * wmag_off[ok_off]

        # Strongly suppress the blue residual sequence if it leaks into the score region.
        if np.any(veto_on[ok_on]):
            w_on = w_on.copy()
            w_on[veto_on[ok_on]] *= 0.05
        if np.any(veto_off[ok_off]):
            w_off = w_off.copy()
            w_off[veto_off[ok_off]] *= 0.05

        score = float(np.sum(w_on) - bg_scale * np.sum(w_off))
        var = float(np.sum(w_on ** 2) + (bg_scale ** 2) * np.sum(w_off ** 2) + 1e-6)
        snr = score / math.sqrt(var)
        scores[i] = score
        snrs[i] = snr

    return scores, snrs


def dm_err_from_curve(dm_grid: np.ndarray, snrs: np.ndarray, idx_best: int) -> float:
    best = snrs[idx_best]
    if not np.isfinite(best):
        return np.nan
    threshold = best - 1.0
    left = idx_best
    while left > 0 and np.isfinite(snrs[left - 1]) and snrs[left - 1] >= threshold:
        left -= 1
    right = idx_best
    while right < len(snrs) - 1 and np.isfinite(snrs[right + 1]) and snrs[right + 1] >= threshold:
        right += 1
    if left == idx_best and right == idx_best:
        return DM_ERR_FLOOR
    return max(DM_ERR_FLOOR, 0.5 * abs(dm_grid[right] - dm_grid[left]))


def fit_distance_anchors_msto(
    zparent: Dict[str, np.ndarray],
    iso_interp: IsoInterp,
    align: Step2Alignment,
    mu_phi1: np.ndarray,
    mu_prior: np.ndarray,
    anchor_centers: np.ndarray,
    anchor_window_half: float,
    dm_scan_half: float,
    dm_scan_step: float,
) -> List[AnchorFit]:
    phi1 = zparent["PHI1"].astype(float)
    phi2 = zparent["PHI2"].astype(float)
    g0 = zparent["G0"].astype(float)
    gr0 = zparent["GR0"].astype(float)

    out: List[AnchorFit] = []
    for center in anchor_centers:
        mu0 = float(np.interp(center, mu_phi1, mu_prior))
        dm0 = align.dm_trailing_best if center <= 0.0 else align.dm_leading_best

        m_local = np.abs(phi1 - center) <= anchor_window_half
        n_local = int(np.sum(m_local))
        if n_local < MIN_LOCAL_STARS:
            out.append(AnchorFit(center, mu0, dm0, dm0, np.nan, np.nan, np.nan, np.nan, n_local, 0, 0, False))
            continue

        dphi2 = phi2[m_local] - mu0
        g_local = g0[m_local]
        gr_local = gr0[m_local]

        m_on = np.abs(dphi2) <= ON_HALFWIDTH
        m_off = (np.abs(dphi2) >= OFF_INNER) & (np.abs(dphi2) <= OFF_OUTER)
        n_on = int(np.sum(m_on))
        n_off = int(np.sum(m_off))
        if n_on < MIN_ON_STARS or n_off < MIN_OFF_STARS:
            out.append(AnchorFit(center, mu0, dm0, dm0, np.nan, np.nan, np.nan, np.nan, n_local, n_on, n_off, False))
            continue

        g_on = g_local[m_on]
        gr_on = gr_local[m_on]
        g_off = g_local[m_off]
        gr_off = gr_local[m_off]
        bg_scale = ON_HALFWIDTH / (OFF_OUTER - OFF_INNER)

        dm_grid = np.arange(dm0 - dm_scan_half, dm0 + dm_scan_half + 0.5 * dm_scan_step, dm_scan_step)
        scores, snrs = score_dm_curve_msto(g_on, gr_on, g_off, gr_off, dm_grid, iso_interp, bg_scale)
        if not np.any(np.isfinite(snrs)):
            out.append(AnchorFit(center, mu0, dm0, dm0, np.nan, np.nan, np.nan, np.nan, n_local, n_on, n_off, False))
            continue

        idx_best = int(np.nanargmax(snrs))
        dm_best = float(dm_grid[idx_best])
        snr_best = float(snrs[idx_best])
        score_best = float(scores[idx_best])
        mask_second = np.ones_like(snrs, dtype=bool)
        mask_second[max(0, idx_best - 1):min(len(snrs), idx_best + 2)] = False
        score_second = float(np.nanmax(snrs[mask_second])) if np.any(mask_second & np.isfinite(snrs)) else np.nan
        dm_err = float(dm_err_from_curve(dm_grid, snrs, idx_best))
        success = np.isfinite(dm_best) and np.isfinite(snr_best) and (snr_best >= 2.5)

        out.append(
            AnchorFit(
                phi1_center=float(center),
                phi2_prior=mu0,
                dm_baseline=dm0,
                dm_best=dm_best,
                dm_err=dm_err,
                score_best=score_best,
                score_second=score_second,
                snr_best=snr_best,
                n_local=n_local,
                n_on=n_on,
                n_off=n_off,
                success=bool(success),
            )
        )
    return out


def weighted_moving_average(y: np.ndarray, w: np.ndarray, kernel: int) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    h = kernel // 2
    for i in range(len(y)):
        lo = max(0, i - h)
        hi = min(len(y), i + h + 1)
        yy = y[lo:hi]
        ww = w[lo:hi]
        ok = np.isfinite(yy) & np.isfinite(ww) & (ww > 0)
        if np.any(ok):
            out[i] = float(np.average(yy[ok], weights=ww[ok]))
    return out


def robust_smooth_arm(centers: np.ndarray, raw_dm: np.ndarray, raw_err: np.ndarray, raw_snr: np.ndarray) -> np.ndarray:
    """Robust low-free-parameter smoothing for one arm."""
    y = raw_dm.copy()
    # Higher weight for high-SNR and low-uncertainty anchors.
    w = np.nan_to_num(raw_snr, nan=0.0)
    if raw_err is not None:
        w = w / np.clip(np.nan_to_num(raw_err, nan=0.20), 0.03, 0.30)
    w = np.clip(w, 0.0, None)

    sm = weighted_moving_average(y, w, kernel=SMOOTH_KERNEL)
    if len(y) < CLIP_MIN_POINTS_PER_ARM or np.sum(np.isfinite(sm)) < 3:
        return sm

    resid = y - sm
    ok = np.isfinite(resid)
    if np.sum(ok) < 3:
        return sm
    med = float(np.nanmedian(resid[ok]))
    mad = float(np.nanmedian(np.abs(resid[ok] - med)))
    sigma = 1.4826 * mad if mad > 0 else 0.0
    thresh = max(CLIP_MAX_ABS_DM, CLIP_SIGMA * sigma)
    clip = ok & (np.abs(resid - med) > thresh)

    y2 = y.copy()
    y2[clip] = sm[clip]
    sm2 = weighted_moving_average(y2, w, kernel=SMOOTH_KERNEL)
    return np.where(np.isfinite(sm2), sm2, sm)


def build_dm_track_robust(
    anchors: List[AnchorFit],
    align: Step2Alignment,
    phi1_grid_fine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([a.phi1_center for a in anchors], dtype=float)
    raw_dm = np.asarray([a.dm_best if a.success else np.nan for a in anchors], dtype=float)
    raw_err = np.asarray([a.dm_err if a.success else np.nan for a in anchors], dtype=float)
    raw_snr = np.asarray([a.snr_best if a.success else np.nan for a in anchors], dtype=float)

    # Force the central anchor to the cluster DM.
    idx0 = int(np.argmin(np.abs(centers)))
    raw_dm[idx0] = align.dm_cluster_best
    raw_err[idx0] = min(np.nan_to_num(raw_err[idx0], nan=0.05), 0.05)
    raw_snr[idx0] = max(np.nan_to_num(raw_snr[idx0], nan=5.0), 5.0)

    ok = np.isfinite(raw_dm)
    if np.sum(ok) < 4:
        raise RuntimeError("Too few successful DM anchors to build DM(phi1).")

    dm_sm = raw_dm.copy()
    for arm_mask in [centers <= 0.0, centers >= 0.0]:
        idx = np.flatnonzero(ok & arm_mask)
        if idx.size == 0:
            continue
        dm_sm[idx] = robust_smooth_arm(centers[idx], raw_dm[idx], raw_err[idx], raw_snr[idx])

    ok_sm = np.isfinite(dm_sm)
    centers_ok = centers[ok_sm]
    dm_ok = dm_sm[ok_sm]
    order = np.argsort(centers_ok)
    centers_ok = centers_ok[order]
    dm_ok = dm_ok[order]

    dm_fine = np.interp(phi1_grid_fine, centers_ok, dm_ok)
    dm_coarse_interp = np.interp(centers, centers_ok, dm_ok)
    return centers, raw_dm, raw_err, dm_coarse_interp, dm_fine


# -----------------------------------------------------------------------------
# Refined member selection pass (copied / adapted from step 4 for step4b names)
# -----------------------------------------------------------------------------
def process_refined_selection_b(
    in_fits: str,
    outdir: str,
    chunk_size: int,
    strict_cfg: Dict[str, float],
    iso_interp: IsoInterp,
    dm_phi1_grid: np.ndarray,
    dm_grid_fine: np.ndarray,
) -> Tuple[str, Dict[str, int], Dict[str, np.ndarray]]:
    tmp_dir = os.path.join(outdir, "tmp_step4b_chunks")
    ensure_dir(tmp_dir)
    for name in os.listdir(tmp_dir):
        if name.endswith(".fits"):
            os.remove(os.path.join(tmp_dir, name))

    hdul = fits.open(in_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[read] {in_fits} rows={n_rows:,}")

    ra_edges = np.arange(RA_MIN, RA_MAX + BIN_DEG_RADEC, BIN_DEG_RADEC)
    dec_edges = np.arange(DEC_MIN, DEC_MAX + BIN_DEG_RADEC, BIN_DEG_RADEC)
    p1_edges = np.arange(PHI1_MIN, PHI1_MAX + BIN_DEG_PHI1, BIN_DEG_PHI1)
    p2_edges = np.arange(PHI2_MIN, PHI2_MAX + BIN_DEG_PHI2, BIN_DEG_PHI2)
    H_radec = np.zeros((len(ra_edges) - 1, len(dec_edges) - 1), dtype=np.int64)
    H_phi = np.zeros((len(p1_edges) - 1, len(p2_edges) - 1), dtype=np.int64)

    cutflow = {
        "input": 0,
        "finite_grz": 0,
        "strict_mag": 0,
        "z_locus": 0,
        "iso_variable_dm": 0,
        "refined_selected": 0,
    }

    out_files: List[str] = []
    chunk_id = 0
    gmin = strict_cfg["STRICT_GMIN"]
    gmax = strict_cfg["STRICT_GMAX"]
    ztol = strict_cfg["ZLOCUS_TOL"]

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
        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = zlocus_residual(gr0, gz0)

        m_fin = finite_grz(g0, r0, z0)
        cutflow["finite_grz"] += int(np.sum(m_fin))
        m_mag = m_fin & (g0 >= gmin) & (g0 < gmax) & (gr0 >= STRICT_GR_MIN) & (gr0 <= STRICT_GR_MAX)
        cutflow["strict_mag"] += int(np.sum(m_mag))
        m_z = m_mag & (gr0 <= ZLOCUS_GR_MAX) & np.isfinite(zres) & (np.abs(zres) <= ztol)
        cutflow["z_locus"] += int(np.sum(m_z))

        dm_local = np.interp(phi1, dm_phi1_grid, dm_grid_fine)
        c_model = iso_interp.gr_color(g0, dm_local)
        dcol = gr0 - c_model
        width = cmd_half_width(g0)
        m_iso = m_z & np.isfinite(dcol) & (np.abs(dcol) <= width)
        cutflow["iso_variable_dm"] += int(np.sum(m_iso))
        cutflow["refined_selected"] += int(np.sum(m_iso))

        if np.any(m_iso):
            sel = m_iso
            h, _, _ = np.histogram2d(ra[sel], dec[sel], bins=[ra_edges, dec_edges])
            H_radec += h.astype(np.int64)
            h, _, _ = np.histogram2d(phi1[sel], phi2[sel], bins=[p1_edges, p2_edges])
            H_phi += h.astype(np.int64)

            t = Table(sub[sel])
            t["GR0"] = gr0[sel].astype(np.float32)
            t["GZ0"] = gz0[sel].astype(np.float32)
            t["ZLOCUS_RESID"] = zres[sel].astype(np.float32)
            t["DM_LOCAL"] = dm_local[sel].astype(np.float32)
            t["ISO_DCOL"] = dcol[sel].astype(np.float32)
            t["ISO_HALF_WIDTH"] = width[sel].astype(np.float32)
            t["SEL_ZLOCUS"] = np.ones(np.sum(sel), dtype=np.int16)
            t["SEL_ISO"] = np.ones(np.sum(sel), dtype=np.int16)
            t["SEL_REFINED"] = np.ones(np.sum(sel), dtype=np.int16)
            out_chunk = os.path.join(tmp_dir, f"step4b_refined_chunk{chunk_id:04d}.fits")
            t.write(out_chunk, overwrite=True)
            out_files.append(out_chunk)

        print(f"[refined chunk {chunk_id:04d}] rows {start:,}-{stop:,} | z={np.sum(m_z):,} refined={np.sum(m_iso):,}")
        chunk_id += 1

    hdul.close()
    if not out_files:
        raise RuntimeError("Refined selection produced no output rows.")

    out_fits = os.path.join(outdir, "pal5_step4b_refined_members.fits")
    merge_fits_list(out_files, out_fits, batch=20)
    hist_data = {
        "H_radec": H_radec,
        "H_phi": H_phi,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "p1_edges": p1_edges,
        "p2_edges": p2_edges,
    }
    return out_fits, cutflow, hist_data


# -----------------------------------------------------------------------------
# QC plotting
# -----------------------------------------------------------------------------
def plot_cmd_segment_grid_full(
    out_png: str,
    zparent: Dict[str, np.ndarray],
    anchors: List[AnchorFit],
    iso_interp: IsoInterp,
    anchor_window_half: float,
    n_panels: int = 6,
) -> None:
    good = [a for a in anchors if a.success]
    if len(good) == 0:
        return
    if len(good) <= n_panels:
        chosen = good
    else:
        idx = np.linspace(0, len(good) - 1, n_panels).round().astype(int)
        chosen = [good[i] for i in idx]

    phi1 = zparent["PHI1"].astype(float)
    phi2 = zparent["PHI2"].astype(float)
    g0 = zparent["G0"].astype(float)
    gr0 = zparent["GR0"].astype(float)

    nx = 3
    ny = int(math.ceil(len(chosen) / nx))
    fig, axes = plt.subplots(ny, nx, figsize=(5.2 * nx, 5.0 * ny), squeeze=False)
    x_edges = np.arange(QC_GR_MIN, QC_GR_MAX + QC_BIN_C, QC_BIN_C)
    y_edges = np.arange(QC_GMIN, QC_GMAX + QC_BIN_G, QC_BIN_G)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, a in zip(axes.ravel(), chosen):
        ax.set_visible(True)
        mu0 = a.phi2_prior
        dphi2 = phi2 - mu0
        m_phi = np.abs(phi1 - a.phi1_center) <= anchor_window_half
        m_on = m_phi & (np.abs(dphi2) <= ON_HALFWIDTH)
        m_off = m_phi & (np.abs(dphi2) >= OFF_INNER) & (np.abs(dphi2) <= OFF_OUTER)
        bg_scale = ON_HALFWIDTH / (OFF_OUTER - OFF_INNER)

        h_on, _, _ = np.histogram2d(gr0[m_on], g0[m_on], bins=[x_edges, y_edges])
        h_off, _, _ = np.histogram2d(gr0[m_off], g0[m_off], bins=[x_edges, y_edges])
        h = np.maximum(h_on - bg_scale * h_off, 0.0)
        img = np.log10(h.T + 1.0)

        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        ax.invert_yaxis()
        ax.set_xlabel(r"$(g-r)_0$")
        ax.set_ylabel(r"$g_0$")
        ax.set_title(rf"$\phi_1 \approx {a.phi1_center:+.1f}^\circ$, DM={a.dm_best:.3f}")

        g_line = np.linspace(QC_GMIN, QC_GMAX, 500)
        c_line = iso_interp.gr_color(g_line, np.full_like(g_line, a.dm_best))
        ax.plot(c_line, g_line, color="orange", lw=1.8)

        # Highlight the actual scoring region, but keep the full plotted range.
        ax.axhspan(SCORE_GMIN, SCORE_GMAX, color="w", alpha=0.05)
        ax.axvspan(QC_GR_MIN, BLUE_VETO_GR_MAX, ymin=0.0, ymax=(QC_GMAX - BLUE_VETO_GMIN) / (QC_GMAX - QC_GMIN),
                   color="tab:blue", alpha=0.05)
        ax.hlines([SCORE_GMIN, SCORE_GMAX], xmin=QC_GR_MIN, xmax=QC_GR_MAX, colors="w", linestyles=":", lw=0.8)
        ax.vlines(BLUE_VETO_GR_MAX, ymin=BLUE_VETO_GMIN, ymax=QC_GMAX, colors="tab:blue", linestyles="--", lw=0.8)

    plt.suptitle("Step 4b: local on-stream minus off-stream CMDs with MSTO-weighted refined DM anchors", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pal 5 step 4b: MSTO-weighted refined distance-gradient selection")
    p.add_argument("--preproc", default=DEFAULT_PREPROC)
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--step2-summary", default=DEFAULT_STEP2_SUMMARY)
    p.add_argument("--mu-prior", default=DEFAULT_MU_PRIOR)
    p.add_argument("--strict-fits", default="step2_outputs/pal5_step2_strict_members.fits")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    p.add_argument("--anchor-step", type=float, default=ANCHOR_STEP)
    p.add_argument("--anchor-window-half", type=float, default=ANCHOR_WINDOW_HALF)
    p.add_argument("--dm-scan-half", type=float, default=DM_SCAN_HALF)
    p.add_argument("--dm-scan-step", type=float, default=DM_SCAN_STEP)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.outdir)
    plots_dir = os.path.join(args.outdir, "plots_step4b")
    ensure_dir(plots_dir)

    align, strict_cfg = load_step2_summary(args.step2_summary)
    iso = read_parsec_like_isochrone(args.iso)
    iso_interp = IsoInterp(iso, align.dc0, align.dc1)
    mu_phi1, mu_prior = read_mu_prior(args.mu_prior)
    zparent = build_zparent_sample(args.preproc, strict_cfg, args.chunk_size)

    anchor_centers = np.arange(ANCHOR_PHI1_MIN, ANCHOR_PHI1_MAX + 0.5 * args.anchor_step, args.anchor_step)
    anchors = fit_distance_anchors_msto(
        zparent,
        iso_interp,
        align,
        mu_phi1,
        mu_prior,
        anchor_centers,
        anchor_window_half=args.anchor_window_half,
        dm_scan_half=args.dm_scan_half,
        dm_scan_step=args.dm_scan_step,
    )

    phi1_grid_fine = np.linspace(float(np.min(mu_phi1)), float(np.max(mu_phi1)), 2000)
    centers, raw_dm, raw_err, dm_coarse_interp, dm_fine = build_dm_track_robust(anchors, align, phi1_grid_fine)

    # Save anchor table.
    anchor_table = Table()
    anchor_table["phi1_center"] = np.asarray([a.phi1_center for a in anchors], dtype=np.float32)
    anchor_table["phi2_prior"] = np.asarray([a.phi2_prior for a in anchors], dtype=np.float32)
    anchor_table["dm_baseline"] = np.asarray([a.dm_baseline for a in anchors], dtype=np.float32)
    anchor_table["dm_best_raw"] = raw_dm.astype(np.float32)
    anchor_table["dm_err"] = raw_err.astype(np.float32)
    anchor_table["dm_interp"] = dm_coarse_interp.astype(np.float32)
    anchor_table["snr_best"] = np.asarray([a.snr_best for a in anchors], dtype=np.float32)
    anchor_table["score_best"] = np.asarray([a.score_best for a in anchors], dtype=np.float32)
    anchor_table["score_second"] = np.asarray([a.score_second for a in anchors], dtype=np.float32)
    anchor_table["n_local"] = np.asarray([a.n_local for a in anchors], dtype=np.int32)
    anchor_table["n_on"] = np.asarray([a.n_on for a in anchors], dtype=np.int32)
    anchor_table["n_off"] = np.asarray([a.n_off for a in anchors], dtype=np.int32)
    anchor_table["success"] = np.asarray([a.success for a in anchors], dtype=np.int16)
    anchor_table.write(os.path.join(args.outdir, "pal5_step4b_dm_anchors.csv"), overwrite=True)

    dm_track = Table()
    dm_track["phi1_fine"] = phi1_grid_fine.astype(np.float32)
    dm_track["dm_interp"] = dm_fine.astype(np.float32)
    dm_track.write(os.path.join(args.outdir, "pal5_step4b_dm_track.csv"), overwrite=True)

    refined_fits, cutflow, hist = process_refined_selection_b(
        args.preproc,
        args.outdir,
        args.chunk_size,
        strict_cfg,
        iso_interp,
        phi1_grid_fine,
        dm_fine,
    )

    # QC plots.
    plot_dm_track(
        os.path.join(plots_dir, "qc_step4b_dm_track.png"),
        anchors, centers, raw_dm, dm_coarse_interp, phi1_grid_fine, dm_fine, align,
    )
    plot_cmd_segment_grid_full(
        os.path.join(plots_dir, "qc_step4b_segment_cmds.png"),
        zparent, anchors, iso_interp, args.anchor_window_half,
    )
    if os.path.exists(args.strict_fits):
        plot_local_compare(
            os.path.join(plots_dir, "qc_step4b_local_compare.png"),
            args.strict_fits,
            refined_fits,
        )
    plot_density_map(
        os.path.join(plots_dir, "qc_step4b_selected_density_phi12.png"),
        hist["H_phi"], hist["p1_edges"], hist["p2_edges"],
        title="step 4b refined selection: Pal 5 frame number density",
        xlabel=r"$\phi_1$ [deg]", ylabel=r"$\phi_2$ [deg]",
    )
    plot_density_map(
        os.path.join(plots_dir, "qc_step4b_selected_density_radec.png"),
        hist["H_radec"], hist["ra_edges"], hist["dec_edges"],
        title="step 4b refined selection: RA-Dec number density",
        xlabel="RA [deg]", ylabel="Dec [deg]",
    )

    n_anchor_success = int(np.sum(np.asarray([a.success for a in anchors], dtype=bool)))
    summary = {
        "input": args.preproc,
        "iso": args.iso,
        "step2_summary": args.step2_summary,
        "mu_prior": args.mu_prior,
        "output_members": refined_fits,
        "n_zparent_total": int(zparent["count_z"][0]),
        "n_anchor_total": int(len(anchors)),
        "n_anchor_success": n_anchor_success,
        "phi1_anchor_min": float(np.min(anchor_centers)),
        "phi1_anchor_max": float(np.max(anchor_centers)),
        "anchor_step": float(args.anchor_step),
        "anchor_window_half": float(args.anchor_window_half),
        "on_halfwidth": float(ON_HALFWIDTH),
        "off_inner": float(OFF_INNER),
        "off_outer": float(OFF_OUTER),
        "dm_scan_half": float(args.dm_scan_half),
        "dm_scan_step": float(args.dm_scan_step),
        "alignment": asdict(align),
        "score_region": {
            "gmin": SCORE_GMIN,
            "gmax": SCORE_GMAX,
            "model_gr_min": SCORE_MODEL_GR_MIN,
            "model_gr_max": SCORE_MODEL_GR_MAX,
            "blue_veto_gr_max": BLUE_VETO_GR_MAX,
            "blue_veto_gmin": BLUE_VETO_GMIN,
        },
        "cutflow": cutflow,
        "dm_track": {
            "dm_cluster_best": align.dm_cluster_best,
            "dm_trailing_best_step2": align.dm_trailing_best,
            "dm_leading_best_step2": align.dm_leading_best,
            "dm_raw_min": float(np.nanmin(raw_dm)),
            "dm_raw_max": float(np.nanmax(raw_dm)),
            "dm_interp_min": float(np.nanmin(dm_fine)),
            "dm_interp_max": float(np.nanmax(dm_fine)),
            "dm_at_phi1_minus15": float(np.interp(-15.0, phi1_grid_fine, dm_fine)),
            "dm_at_phi1_0": float(np.interp(0.0, phi1_grid_fine, dm_fine)),
            "dm_at_phi1_plus8": float(np.interp(8.0, phi1_grid_fine, dm_fine)),
        },
    }
    with open(os.path.join(args.outdir, "pal5_step4b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.outdir, "pal5_step4b_cutflow.txt"), "w", encoding="utf-8") as f:
        f.write("Pal 5 step 4b MSTO-weighted refined-distance member-selection cut-flow\n")
        f.write(f"INPUT_FITS = {args.preproc}\n")
        f.write(f"ISO_FILE   = {args.iso}\n")
        f.write(f"OUTPUT_FITS = {refined_fits}\n\n")
        for key, val in cutflow.items():
            frac = 100.0 * val / max(cutflow["input"], 1)
            f.write(f"{key:24s}: {val:12d}   ({frac:8.4f}%)\n")

    with open(os.path.join(args.outdir, "pal5_step4b_report.md"), "w", encoding="utf-8") as f:
        f.write("# Pal 5 step 4b MSTO-weighted refined distance-gradient selection\n\n")
        f.write("This run refines the step-4 coarse DM(phi1) idea by scoring anchors only in the ")
        f.write("distance-sensitive MSTO / upper-MS region, downweighting the blue residual sequence, ")
        f.write("and robustly smoothing the raw 2-degree anchors before interpolation.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- z-parent sample after strict mag + z-locus: **{int(zparent['count_z'][0]):,}**\n")
        f.write(f"- successful DM anchors: **{n_anchor_success} / {len(anchors)}**\n")
        f.write(f"- refined selected members: **{cutflow['refined_selected']:,}**\n")
        f.write(f"- DM(phi1=-15 deg): **{summary['dm_track']['dm_at_phi1_minus15']:.3f}**\n")
        f.write(f"- DM(phi1=0 deg): **{summary['dm_track']['dm_at_phi1_0']:.3f}**\n")
        f.write(f"- DM(phi1=+8 deg): **{summary['dm_track']['dm_at_phi1_plus8']:.3f}**\n")
        f.write("\n## Anchor-scoring region used for DM fitting\n\n")
        f.write(f"- score window in magnitude: **{SCORE_GMIN:.1f} < g0 < {SCORE_GMAX:.1f}**\n")
        f.write(f"- model-color gate: **{SCORE_MODEL_GR_MIN:.2f} < (g-r)_iso < {SCORE_MODEL_GR_MAX:.2f}**\n")
        f.write(f"- blue-residual veto: **(g-r)_0 < {BLUE_VETO_GR_MAX:.2f}** and **g0 > {BLUE_VETO_GMIN:.1f}**\n")
        f.write("\n## QC files\n\n")
        f.write("- `plots_step4b/qc_step4b_dm_track.png`\n")
        f.write("- `plots_step4b/qc_step4b_segment_cmds.png`\n")
        f.write("- `plots_step4b/qc_step4b_local_compare.png`\n")
        f.write("- `plots_step4b/qc_step4b_selected_density_phi12.png`\n")
        f.write("- `plots_step4b/qc_step4b_selected_density_radec.png`\n")
        f.write("\nThe segment CMD figure intentionally shows the **full 16 < g0 < 24** range, ")
        f.write("even though the anchor score itself only uses the narrower MSTO-focused region.\n")

    print("[done] step 4b MSTO-weighted refined-distance selection complete")
    print(f"[outdir] {args.outdir}")


if __name__ == "__main__":
    main()
