#!/usr/bin/env python3
"""
Pal 5 step 4: Bonaca-style refined distance-gradient selection.

This script sits between the strict step-2 member selection and the step-3
Bonaca-style 1D morphology modeling.

Goal
----
Mimic the next refinement step in Bonaca+2020 Section 2:
  1) start from the preprocessed parent catalog,
  2) rebuild the strict z-locus parent sample,
  3) estimate the stream distance modulus in coarse bins along phi1,
  4) interpolate a smooth DM(phi1) track,
  5) re-run the strict isochrone selection using the varying DM(phi1),
  6) write a refined member catalog for downstream morphology modeling.

Important philosophy
--------------------
This is intentionally still a *Bonaca-style* selection stage, not a full joint
DM(phi1) + spatial generative model.  The DM track is measured in coarse bins
using local on/off-stream CMD contrast and then interpolated.  It is therefore
much closer in spirit to the paper than a global matched-filter fit, while still
being reproducible and automatable for Codex.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------
DEFAULT_PREPROC = "final_g25_preproc.fits"
DEFAULT_ISO = "pal5.dat"
DEFAULT_STEP2_SUMMARY = "step2_outputs/pal5_step2_summary.json"
DEFAULT_MU_PRIOR = "pal5_step3b_mu_prior_control.txt"
DEFAULT_STRICT_FITS = "step2_outputs/pal5_step2_strict_members.fits"
DEFAULT_OUTDIR = "step4_outputs"

RA_MIN, RA_MAX = 210.0, 260.0
DEC_MIN, DEC_MAX = -20.0, 20.0
PHI1_MIN, PHI1_MAX = -25.0, 20.0
PHI2_MIN, PHI2_MAX = -5.0, 10.0

# If the step-2 summary is unavailable, fall back to the strict defaults that
# were already used successfully in this project.
STRICT_GMIN = 20.0
STRICT_GMAX = 23.0
STRICT_GR_MIN = -0.35
STRICT_GR_MAX = 1.25
ZLOCUS_SLOPE = 1.7
ZLOCUS_INTERCEPT = -0.17
ZLOCUS_TOL = 0.10
ZLOCUS_GR_MAX = 1.20
CMD_W0 = 0.06
CMD_W_SLOPE = 0.018
CMD_W_MIN = 0.05
CMD_W_MAX = 0.14
CMD_W_REF = 20.0
G_REF_TILT = 21.0

# Coarse Bonaca-like distance anchors.
ANCHOR_PHI1_MIN = -19.0
ANCHOR_PHI1_MAX = 9.0
ANCHOR_STEP = 2.0
ANCHOR_WINDOW_HALF = 1.5     # 3 degree long boxes, matching the paper figures
ON_HALFWIDTH = 0.40
OFF_INNER = 0.80
OFF_OUTER = 1.60
DM_SCAN_HALF = 0.40
DM_SCAN_STEP = 0.01
DM_SIGMA_SCALE = 1.35        # slightly broader than the strict CMD cut
MIN_ON_STARS = 50
MIN_OFF_STARS = 80
MIN_LOCAL_STARS = 200
ANCHOR_SMOOTH_KERNEL = 3     # weighted moving average before interpolation

BIN_DEG_RADEC = 0.25
BIN_DEG_PHI1 = 0.25
BIN_DEG_PHI2 = 0.05
CHUNK_SIZE_DEFAULT = 2_000_000
RNG_SEED = 12345


@dataclass
class Step2Alignment:
    dmu: float
    dc0: float
    dc1: float
    dm_cluster_best: float
    dm_trailing_best: float
    dm_leading_best: float


@dataclass
class AnchorFit:
    phi1_center: float
    phi2_prior: float
    dm_baseline: float
    dm_best: float
    dm_err: float
    score_best: float
    score_second: float
    snr_best: float
    n_local: int
    n_on: int
    n_off: int
    success: bool


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_remove_glob(directory: str, suffix: str) -> None:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.endswith(suffix):
            try:
                os.remove(os.path.join(directory, name))
            except FileNotFoundError:
                pass


def merge_fits_list(file_list: List[str], out_fits: str, batch: int = 20) -> None:
    file_list = sorted(file_list)
    if not file_list:
        raise RuntimeError(f"No FITS chunks found to merge into {out_fits}")
    acc: Optional[Table] = None
    for i in range(0, len(file_list), batch):
        group = file_list[i:i + batch]
        tabs = [Table.read(fn) for fn in group]
        merged = vstack(tabs, metadata_conflicts="silent")
        acc = merged if acc is None else vstack([acc, merged], metadata_conflicts="silent")
        print(f"[merge] {i + len(group):,}/{len(file_list):,} chunks -> rows {len(acc):,}")
    assert acc is not None
    acc.write(out_fits, overwrite=True)
    print(f"[write] {out_fits} rows={len(acc):,}")


def finite_grz(g0: np.ndarray, r0: np.ndarray, z0: np.ndarray) -> np.ndarray:
    return np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)


def zlocus_residual(gr0: np.ndarray, gz0: np.ndarray) -> np.ndarray:
    return gz0 - (ZLOCUS_SLOPE * gr0 + ZLOCUS_INTERCEPT)


def cmd_half_width(g0: np.ndarray) -> np.ndarray:
    width = CMD_W0 + CMD_W_SLOPE * (np.asarray(g0, dtype=float) - CMD_W_REF)
    return np.clip(width, CMD_W_MIN, CMD_W_MAX)


def dm_fit_sigma(g0: np.ndarray) -> np.ndarray:
    # Use a slightly broader kernel than the strict step-2 hard cut when
    # scoring the local on/off CMD contrast.
    return np.clip(DM_SIGMA_SCALE * cmd_half_width(g0), 0.06, 0.20)


# -----------------------------------------------------------------------------
# Isochrone handling
# -----------------------------------------------------------------------------
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
    gr_abs = g_abs - r_abs
    gz_abs = g_abs - z_abs

    order = np.argsort(g_abs)
    g_abs = g_abs[order]
    gr_abs = gr_abs[order]
    gz_abs = gz_abs[order]

    # Collapse repeated g_abs values.
    g_unique: List[float] = []
    gr_unique: List[float] = []
    gz_unique: List[float] = []
    i = 0
    while i < len(g_abs):
        j = i + 1
        while j < len(g_abs) and abs(g_abs[j] - g_abs[i]) < 1e-5:
            j += 1
        g_unique.append(float(np.mean(g_abs[i:j])))
        gr_unique.append(float(np.mean(gr_abs[i:j])))
        gz_unique.append(float(np.mean(gz_abs[i:j])))
        i = j

    return {
        "g_abs": np.asarray(g_unique, dtype=float),
        "gr_abs": np.asarray(gr_unique, dtype=float),
        "gz_abs": np.asarray(gz_unique, dtype=float),
    }


class IsoInterp:
    def __init__(self, iso: Dict[str, np.ndarray], dc0: float, dc1: float):
        self.g_abs = np.asarray(iso["g_abs"], dtype=float)
        self.gr_abs = np.asarray(iso["gr_abs"], dtype=float)
        self.dc0 = float(dc0)
        self.dc1 = float(dc1)
        self.g_abs_min = float(np.min(self.g_abs))
        self.g_abs_max = float(np.max(self.g_abs))

    def gr_color(self, g_obs: np.ndarray, dm_local: np.ndarray) -> np.ndarray:
        g_obs = np.asarray(g_obs, dtype=float)
        dm_local = np.asarray(dm_local, dtype=float)
        q = g_obs - dm_local
        out = np.full_like(q, np.nan, dtype=float)
        ok = np.isfinite(q) & (q >= self.g_abs_min) & (q <= self.g_abs_max)
        if np.any(ok):
            out[ok] = np.interp(q[ok], self.g_abs, self.gr_abs)
            out[ok] = out[ok] + self.dc0 + self.dc1 * (g_obs[ok] - G_REF_TILT)
        return out


# -----------------------------------------------------------------------------
# Input helpers
# -----------------------------------------------------------------------------
def load_step2_summary(path: str) -> Tuple[Step2Alignment, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    align_d = data.get("alignment", {})
    strict_d = data.get("strict_config", {})
    align = Step2Alignment(
        dmu=float(align_d.get("dmu", 0.0)),
        dc0=float(align_d.get("dc0", 0.0)),
        dc1=float(align_d.get("dc1", 0.0)),
        dm_cluster_best=float(align_d.get("dm_cluster_best", 16.835)),
        dm_trailing_best=float(align_d.get("dm_trailing_best", 16.835 + 0.0)),
        dm_leading_best=float(align_d.get("dm_leading_best", 16.835 - 0.415)),
    )
    strict = {
        "STRICT_GMIN": float(strict_d.get("STRICT_GMIN", STRICT_GMIN)),
        "STRICT_GMAX": float(strict_d.get("STRICT_GMAX", STRICT_GMAX)),
        "ZLOCUS_TOL": float(strict_d.get("ZLOCUS_TOL", ZLOCUS_TOL)),
        "CMD_W0": float(strict_d.get("CMD_W0", CMD_W0)),
        "CMD_W_SLOPE": float(strict_d.get("CMD_W_SLOPE", CMD_W_SLOPE)),
        "CMD_W_MIN": float(strict_d.get("CMD_W_MIN", CMD_W_MIN)),
        "CMD_W_MAX": float(strict_d.get("CMD_W_MAX", CMD_W_MAX)),
    }
    return align, strict


def read_mu_prior(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Unexpected mu-prior file format: {path}")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


# -----------------------------------------------------------------------------
# Build the step-2 z-parent sample in memory
# -----------------------------------------------------------------------------
def build_zparent_sample(in_fits: str, strict_cfg: Dict[str, float], chunk_size: int) -> Dict[str, np.ndarray]:
    hdul = fits.open(in_fits, memmap=True)
    data = hdul[1].data
    n_rows = len(data)
    print(f"[read] {in_fits} rows={n_rows:,}")

    needed = ["RA", "DEC", "PHI1", "PHI2", "G0", "R0", "Z0"]
    missing = [c for c in needed if c not in data.dtype.names]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out: Dict[str, List[np.ndarray]] = {k: [] for k in ["RA", "DEC", "PHI1", "PHI2", "G0", "R0", "Z0", "GR0", "GZ0", "ZRES"]}
    count_input = 0
    count_fin = 0
    count_mag = 0
    count_z = 0

    gmin = strict_cfg["STRICT_GMIN"]
    gmax = strict_cfg["STRICT_GMAX"]
    ztol = strict_cfg["ZLOCUS_TOL"]

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        sub = data[start:stop]
        count_input += len(sub)

        ra = np.asarray(sub["RA"], dtype=np.float32)
        dec = np.asarray(sub["DEC"], dtype=np.float32)
        phi1 = np.asarray(sub["PHI1"], dtype=np.float32)
        phi2 = np.asarray(sub["PHI2"], dtype=np.float32)
        g0 = np.asarray(sub["G0"], dtype=np.float32)
        r0 = np.asarray(sub["R0"], dtype=np.float32)
        z0 = np.asarray(sub["Z0"], dtype=np.float32)
        gr0 = g0 - r0
        gz0 = g0 - z0
        zres = zlocus_residual(gr0, gz0).astype(np.float32)

        m_fin = finite_grz(g0, r0, z0)
        count_fin += int(np.sum(m_fin))

        m_mag = m_fin & (g0 >= gmin) & (g0 < gmax) & (gr0 >= STRICT_GR_MIN) & (gr0 <= STRICT_GR_MAX)
        count_mag += int(np.sum(m_mag))

        m_z = m_mag & (gr0 <= ZLOCUS_GR_MAX) & np.isfinite(zres) & (np.abs(zres) <= ztol)
        count_z += int(np.sum(m_z))

        if np.any(m_z):
            for k, arr in zip(
                ["RA", "DEC", "PHI1", "PHI2", "G0", "R0", "Z0", "GR0", "GZ0", "ZRES"],
                [ra[m_z], dec[m_z], phi1[m_z], phi2[m_z], g0[m_z], r0[m_z], z0[m_z], gr0[m_z], gz0[m_z], zres[m_z]],
            ):
                out[k].append(arr.astype(np.float32, copy=False))

        print(f"[zparent chunk] {start:,}-{stop:,} | mag={np.sum(m_mag):,} z={np.sum(m_z):,}")

    hdul.close()

    zparent = {k: np.concatenate(v) if v else np.array([], dtype=np.float32) for k, v in out.items()}
    zparent["count_input"] = np.array([count_input], dtype=np.int64)
    zparent["count_fin"] = np.array([count_fin], dtype=np.int64)
    zparent["count_mag"] = np.array([count_mag], dtype=np.int64)
    zparent["count_z"] = np.array([count_z], dtype=np.int64)
    print(f"[zparent] input={count_input:,} finite={count_fin:,} strict_mag={count_mag:,} z_locus={count_z:,}")
    return zparent


# -----------------------------------------------------------------------------
# Coarse DM-anchor fitting
# -----------------------------------------------------------------------------
def score_dm_curve(
    g_on: np.ndarray,
    gr_on: np.ndarray,
    g_off: np.ndarray,
    gr_off: np.ndarray,
    dm_grid: np.ndarray,
    iso_interp: IsoInterp,
    bg_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.full(dm_grid.shape, np.nan, dtype=float)
    snrs = np.full(dm_grid.shape, np.nan, dtype=float)

    sig_on = dm_fit_sigma(g_on)
    sig_off = dm_fit_sigma(g_off)

    for i, dm in enumerate(dm_grid):
        c_on = iso_interp.gr_color(g_on, np.full_like(g_on, dm))
        c_off = iso_interp.gr_color(g_off, np.full_like(g_off, dm))

        d_on = gr_on - c_on
        d_off = gr_off - c_off

        ok_on = np.isfinite(d_on)
        ok_off = np.isfinite(d_off)
        if (not np.any(ok_on)) or (not np.any(ok_off)):
            continue

        w_on = np.exp(-0.5 * (d_on[ok_on] / sig_on[ok_on]) ** 2)
        w_off = np.exp(-0.5 * (d_off[ok_off] / sig_off[ok_off]) ** 2)
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
        return 0.5 * DM_SCAN_STEP
    return 0.5 * abs(dm_grid[right] - dm_grid[left])


def fit_distance_anchors(
    zparent: Dict[str, np.ndarray],
    iso_interp: IsoInterp,
    align: Step2Alignment,
    mu_phi1: np.ndarray,
    mu_prior: np.ndarray,
    anchor_centers: np.ndarray,
) -> List[AnchorFit]:
    phi1 = zparent["PHI1"].astype(float)
    phi2 = zparent["PHI2"].astype(float)
    g0 = zparent["G0"].astype(float)
    gr0 = zparent["GR0"].astype(float)

    out: List[AnchorFit] = []
    for center in anchor_centers:
        mu0 = float(np.interp(center, mu_phi1, mu_prior))
        dm0 = align.dm_trailing_best if center <= 0.0 else align.dm_leading_best

        m_local = np.abs(phi1 - center) <= ANCHOR_WINDOW_HALF
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

        dm_grid = np.arange(dm0 - DM_SCAN_HALF, dm0 + DM_SCAN_HALF + 0.5 * DM_SCAN_STEP, DM_SCAN_STEP)
        scores, snrs = score_dm_curve(g_on, gr_on, g_off, gr_off, dm_grid, iso_interp, bg_scale)
        if not np.any(np.isfinite(snrs)):
            out.append(AnchorFit(center, mu0, dm0, dm0, np.nan, np.nan, np.nan, np.nan, n_local, n_on, n_off, False))
            continue

        idx_best = int(np.nanargmax(snrs))
        dm_best = float(dm_grid[idx_best])
        snr_best = float(snrs[idx_best])
        score_best = float(scores[idx_best])
        # Second best outside a narrow neighborhood.
        mask_second = np.ones_like(snrs, dtype=bool)
        mask_second[max(0, idx_best - 1):min(len(snrs), idx_best + 2)] = False
        if np.any(mask_second & np.isfinite(snrs)):
            score_second = float(np.nanmax(snrs[mask_second]))
        else:
            score_second = np.nan
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


def weighted_moving_average(x: np.ndarray, y: np.ndarray, w: np.ndarray, kernel: int = 3) -> np.ndarray:
    if kernel <= 1 or len(y) < 3:
        return y.copy()
    out = np.full_like(y, np.nan, dtype=float)
    h = kernel // 2
    for i in range(len(y)):
        lo = max(0, i - h)
        hi = min(len(y), i + h + 1)
        ww = w[lo:hi]
        yy = y[lo:hi]
        ok = np.isfinite(ww) & np.isfinite(yy) & (ww > 0)
        if np.any(ok):
            out[i] = float(np.average(yy[ok], weights=ww[ok]))
        elif np.isfinite(y[i]):
            out[i] = y[i]
    return out


def build_dm_track(
    anchors: List[AnchorFit],
    align: Step2Alignment,
    phi1_grid_fine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([a.phi1_center for a in anchors], dtype=float)
    raw_dm = np.asarray([a.dm_best if a.success else np.nan for a in anchors], dtype=float)
    raw_err = np.asarray([a.dm_err if a.success else np.nan for a in anchors], dtype=float)
    snr = np.asarray([a.snr_best if a.success else np.nan for a in anchors], dtype=float)

    # Force a central anchor to the cluster DM if we are close to phi1=0.
    idx0 = int(np.argmin(np.abs(centers)))
    raw_dm[idx0] = align.dm_cluster_best
    raw_err[idx0] = min(np.nan_to_num(raw_err[idx0], nan=0.05), 0.05)
    snr[idx0] = max(np.nan_to_num(snr[idx0], nan=5.0), 5.0)

    ok = np.isfinite(raw_dm)
    if np.sum(ok) < 4:
        raise RuntimeError("Too few successful DM anchors to build an interpolated DM(phi1) track.")

    weights = np.nan_to_num(snr, nan=0.0)
    weights = np.clip(weights, 0.0, None)
    raw_dm_sm = raw_dm.copy()

    # Smooth each arm separately before linear interpolation.
    for arm_mask in [centers <= 0.0, centers >= 0.0]:
        idx = np.flatnonzero(ok & arm_mask)
        if idx.size == 0:
            continue
        raw_dm_sm[idx] = weighted_moving_average(centers[idx], raw_dm[idx], weights[idx], kernel=ANCHOR_SMOOTH_KERNEL)

    ok_sm = np.isfinite(raw_dm_sm)
    centers_ok = centers[ok_sm]
    dm_ok = raw_dm_sm[ok_sm]

    # Enforce sorted unique x for interpolation.
    order = np.argsort(centers_ok)
    centers_ok = centers_ok[order]
    dm_ok = dm_ok[order]

    dm_fine = np.interp(phi1_grid_fine, centers_ok, dm_ok)
    dm_coarse_interp = np.interp(centers, centers_ok, dm_ok)
    return centers, raw_dm, raw_err, dm_coarse_interp, dm_fine


# -----------------------------------------------------------------------------
# Second-pass refined selection
# -----------------------------------------------------------------------------
def process_refined_selection(
    in_fits: str,
    outdir: str,
    chunk_size: int,
    strict_cfg: Dict[str, float],
    iso_interp: IsoInterp,
    dm_phi1_grid: np.ndarray,
    dm_grid_fine: np.ndarray,
) -> Tuple[str, Dict[str, int], Dict[str, np.ndarray]]:
    tmp_dir = os.path.join(outdir, "tmp_step4_chunks")
    ensure_dir(tmp_dir)
    safe_remove_glob(tmp_dir, ".fits")

    plots_dir = os.path.join(outdir, "plots_step4")
    ensure_dir(plots_dir)

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

            out_chunk = os.path.join(tmp_dir, f"step4_refined_chunk{chunk_id:04d}.fits")
            t.write(out_chunk, overwrite=True)
            out_files.append(out_chunk)

        print(f"[refined chunk {chunk_id:04d}] rows {start:,}-{stop:,} | z={np.sum(m_z):,} refined={np.sum(m_iso):,}")
        chunk_id += 1

    hdul.close()

    if not out_files:
        raise RuntimeError("Refined selection produced no output rows.")

    out_fits = os.path.join(outdir, "pal5_step4_refined_members.fits")
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
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_density_map(out_png: str, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                     title: str, xlabel: str, ylabel: str) -> None:
    bin_x = np.median(np.diff(x_edges))
    bin_y = np.median(np.diff(y_edges))
    img = H.T / (bin_x * bin_y)
    plt.figure(figsize=(10, 6.5))
    plt.imshow(img, origin="lower", aspect="auto",
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    cb = plt.colorbar()
    cb.set_label(r"counts / deg$^2$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_dm_track(out_png: str, anchors: List[AnchorFit], centers: np.ndarray,
                  raw_dm: np.ndarray, dm_smooth: np.ndarray,
                  phi1_grid_fine: np.ndarray, dm_grid_fine: np.ndarray,
                  align: Step2Alignment) -> None:
    plt.figure(figsize=(10, 5.8))
    base = np.where(phi1_grid_fine <= 0.0, align.dm_trailing_best, align.dm_leading_best)
    plt.plot(phi1_grid_fine, base, "--", lw=1.8, label="two-arm baseline")
    ok = np.isfinite(raw_dm)
    if np.any(ok):
        yerr = np.asarray([a.dm_err for a in anchors], dtype=float)
        plt.errorbar(centers[ok], raw_dm[ok], yerr=yerr[ok], fmt="o", ms=4.5, label="raw 2° anchors")
    plt.plot(centers, dm_smooth, "s-", ms=4.2, lw=1.4, label="smoothed anchors")
    plt.plot(phi1_grid_fine, dm_grid_fine, lw=2.3, label=r"interpolated $DM(\phi_1)$")
    plt.axvline(0.0, color="0.6", lw=1)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("distance modulus")
    plt.title("Step 4: coarse Bonaca-style distance anchors and interpolated DM(track)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_cmd_segment_grid(out_png: str, zparent: Dict[str, np.ndarray], anchors: List[AnchorFit],
                          iso_interp: IsoInterp, n_panels: int = 6) -> None:
    # Pick evenly spaced successful anchors, preferring the best-S/N points.
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

    x_edges = np.arange(-0.35, 1.05 + 0.02, 0.02)
    y_edges = np.arange(18.0, 23.6 + 0.05, 0.05)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, a in zip(axes.ravel(), chosen):
        ax.set_visible(True)
        mu0 = a.phi2_prior
        dphi2 = phi2 - mu0
        m_phi = np.abs(phi1 - a.phi1_center) <= ANCHOR_WINDOW_HALF
        m_on = m_phi & (np.abs(dphi2) <= ON_HALFWIDTH)
        m_off = m_phi & (np.abs(dphi2) >= OFF_INNER) & (np.abs(dphi2) <= OFF_OUTER)
        bg_scale = ON_HALFWIDTH / (OFF_OUTER - OFF_INNER)

        h_on, _, _ = np.histogram2d(gr0[m_on], g0[m_on], bins=[x_edges, y_edges])
        h_off, _, _ = np.histogram2d(gr0[m_off], g0[m_off], bins=[x_edges, y_edges])
        h = h_on - bg_scale * h_off
        h = np.maximum(h, 0.0)
        img = np.log10(h.T + 1.0)

        ax.imshow(img, origin="lower", aspect="auto",
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        ax.invert_yaxis()
        ax.set_xlabel(r"$(g-r)_0$")
        ax.set_ylabel(r"$g_0$")
        ax.set_title(rf"$\phi_1 \approx {a.phi1_center:+.1f}^\circ$, DM={a.dm_best:.3f}")

        g_line = np.linspace(18.0, 23.5, 400)
        c_line = iso_interp.gr_color(g_line, np.full_like(g_line, a.dm_best))
        ax.plot(c_line, g_line, color="orange", lw=1.8)

    plt.suptitle("Step 4: local on-stream minus off-stream CMDs with refined DM anchors", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_local_compare(out_png: str, strict_fits: str, refined_fits: str) -> None:
    if not (os.path.exists(strict_fits) and os.path.exists(refined_fits)):
        return
    t_old = Table.read(strict_fits)
    t_new = Table.read(refined_fits)

    p1_edges = np.arange(-20.0, 10.0 + 0.25, 0.25)
    p2_edges = np.arange(-2.5, 2.5 + 0.05, 0.05)
    h_old, _, _ = np.histogram2d(np.asarray(t_old["PHI1"], dtype=float), np.asarray(t_old["PHI2"], dtype=float), bins=[p1_edges, p2_edges])
    h_new, _, _ = np.histogram2d(np.asarray(t_new["PHI1"], dtype=float), np.asarray(t_new["PHI2"], dtype=float), bins=[p1_edges, p2_edges])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharex=True, sharey=True)
    for ax, H, title in zip(axes, [h_old, h_new], ["step 2 strict", "step 4 refined DM"]):
        img = H.T / (0.25 * 0.05)
        im = ax.imshow(img, origin="lower", aspect="auto",
                       extent=[p1_edges[0], p1_edges[-1], p2_edges[0], p2_edges[-1]])
        ax.set_title(title)
        ax.set_xlabel(r"$\phi_1$ [deg]")
        ax.set_ylabel(r"$\phi_2$ [deg]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label=r"counts / deg$^2$")
    fig.suptitle("Strict selection before / after the refined Bonaca-style DM(track)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    global ANCHOR_WINDOW_HALF, ON_HALFWIDTH, OFF_INNER, OFF_OUTER, DM_SCAN_HALF, DM_SCAN_STEP
    p = argparse.ArgumentParser(description="Pal 5 step 4: refined distance-gradient selection")
    p.add_argument("--preproc", default=DEFAULT_PREPROC, help="Preprocessed parent FITS catalog")
    p.add_argument("--iso", default=DEFAULT_ISO, help="Isochrone file")
    p.add_argument("--step2-summary", default=DEFAULT_STEP2_SUMMARY, help="Step 2 summary JSON")
    p.add_argument("--mu-prior", default=DEFAULT_MU_PRIOR, help="Step 3b smoothed mu prior text file")
    p.add_argument("--strict-fits", default=DEFAULT_STRICT_FITS, help="Existing step-2 strict member FITS for comparison plots")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    p.add_argument("--anchor-step", type=float, default=ANCHOR_STEP)
    p.add_argument("--anchor-window-half", type=float, default=ANCHOR_WINDOW_HALF)
    p.add_argument("--on-halfwidth", type=float, default=ON_HALFWIDTH)
    p.add_argument("--off-inner", type=float, default=OFF_INNER)
    p.add_argument("--off-outer", type=float, default=OFF_OUTER)
    p.add_argument("--dm-scan-half", type=float, default=DM_SCAN_HALF)
    p.add_argument("--dm-scan-step", type=float, default=DM_SCAN_STEP)
    args = p.parse_args()

    ANCHOR_WINDOW_HALF = float(args.anchor_window_half)
    ON_HALFWIDTH = float(args.on_halfwidth)
    OFF_INNER = float(args.off_inner)
    OFF_OUTER = float(args.off_outer)
    DM_SCAN_HALF = float(args.dm_scan_half)
    DM_SCAN_STEP = float(args.dm_scan_step)

    ensure_dir(args.outdir)
    plots_dir = os.path.join(args.outdir, "plots_step4")
    ensure_dir(plots_dir)

    align, strict_cfg = load_step2_summary(args.step2_summary)
    strict_cfg["STRICT_GMIN"] = strict_cfg.get("STRICT_GMIN", STRICT_GMIN)
    strict_cfg["STRICT_GMAX"] = strict_cfg.get("STRICT_GMAX", STRICT_GMAX)
    strict_cfg["ZLOCUS_TOL"] = strict_cfg.get("ZLOCUS_TOL", ZLOCUS_TOL)

    iso = read_parsec_like_isochrone(args.iso)
    iso_interp = IsoInterp(iso, dc0=align.dc0, dc1=align.dc1)
    mu_phi1, mu_prior = read_mu_prior(args.mu_prior)

    zparent = build_zparent_sample(args.preproc, strict_cfg, chunk_size=args.chunk_size)
    anchor_centers = np.arange(ANCHOR_PHI1_MIN, ANCHOR_PHI1_MAX + 0.5 * args.anchor_step, args.anchor_step)
    anchors = fit_distance_anchors(zparent, iso_interp, align, mu_phi1, mu_prior, anchor_centers)

    phi1_grid_fine = np.arange(ANCHOR_PHI1_MIN, ANCHOR_PHI1_MAX + 0.001, 0.05)
    centers, raw_dm, raw_err, dm_coarse_interp, dm_fine = build_dm_track(anchors, align, phi1_grid_fine)

    # Save the anchor table.
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
    anchor_table.write(os.path.join(args.outdir, "pal5_step4_dm_anchors.csv"), overwrite=True)

    dm_track_table = Table()
    dm_track_table["phi1"] = phi1_grid_fine.astype(np.float32)
    dm_track_table["dm_interp"] = dm_fine.astype(np.float32)
    dm_track_table.write(os.path.join(args.outdir, "pal5_step4_dm_track.csv"), overwrite=True)

    refined_fits, cutflow, hist_data = process_refined_selection(
        args.preproc,
        args.outdir,
        chunk_size=args.chunk_size,
        strict_cfg=strict_cfg,
        iso_interp=iso_interp,
        dm_phi1_grid=phi1_grid_fine,
        dm_grid_fine=dm_fine,
    )

    # QC plots.
    plot_dm_track(
        os.path.join(plots_dir, "qc_step4_dm_track.png"),
        anchors, centers, raw_dm, dm_coarse_interp, phi1_grid_fine, dm_fine, align,
    )
    plot_cmd_segment_grid(
        os.path.join(plots_dir, "qc_step4_segment_cmds.png"),
        zparent, anchors, iso_interp,
    )
    plot_local_compare(
        os.path.join(plots_dir, "qc_step4_local_compare.png"),
        args.strict_fits,
        refined_fits,
    )
    plot_density_map(
        os.path.join(plots_dir, "qc_step4_selected_density_radec.png"),
        hist_data["H_radec"], hist_data["ra_edges"], hist_data["dec_edges"],
        "step 4 refined selection: RA-Dec number density",
        "RA [deg]", "Dec [deg]",
    )
    plot_density_map(
        os.path.join(plots_dir, "qc_step4_selected_density_phi12.png"),
        hist_data["H_phi"], hist_data["p1_edges"], hist_data["p2_edges"],
        "step 4 refined selection: Pal 5 frame number density",
        r"$\phi_1$ [deg]", r"$\phi_2$ [deg]",
    )

    # Summary.
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
        "anchor_window_half": float(ANCHOR_WINDOW_HALF),
        "on_halfwidth": float(ON_HALFWIDTH),
        "off_inner": float(OFF_INNER),
        "off_outer": float(OFF_OUTER),
        "dm_scan_half": float(DM_SCAN_HALF),
        "dm_scan_step": float(DM_SCAN_STEP),
        "alignment": asdict(align),
        "cutflow": cutflow,
        "dm_track": {
            "dm_cluster_best": float(align.dm_cluster_best),
            "dm_trailing_best_step2": float(align.dm_trailing_best),
            "dm_leading_best_step2": float(align.dm_leading_best),
            "dm_raw_min": float(np.nanmin(raw_dm)),
            "dm_raw_max": float(np.nanmax(raw_dm)),
            "dm_interp_min": float(np.nanmin(dm_fine)),
            "dm_interp_max": float(np.nanmax(dm_fine)),
            "dm_at_phi1_minus15": float(np.interp(-15.0, phi1_grid_fine, dm_fine)),
            "dm_at_phi1_0": float(np.interp(0.0, phi1_grid_fine, dm_fine)),
            "dm_at_phi1_plus8": float(np.interp(8.0, phi1_grid_fine, dm_fine)),
        },
    }
    with open(os.path.join(args.outdir, "pal5_step4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.outdir, "pal5_step4_cutflow.txt"), "w", encoding="utf-8") as f:
        f.write("Pal 5 step 4 refined-distance member-selection cut-flow\n")
        f.write(f"INPUT_FITS = {args.preproc}\n")
        f.write(f"ISO_FILE   = {args.iso}\n")
        f.write(f"OUTPUT_FITS = {refined_fits}\n\n")
        for key, val in cutflow.items():
            frac = 100.0 * val / max(cutflow["input"], 1)
            f.write(f"{key:24s}: {val:12d}   ({frac:8.4f}%)\n")

    with open(os.path.join(args.outdir, "pal5_step4_report.md"), "w", encoding="utf-8") as f:
        f.write("# Pal 5 step 4 refined distance-gradient selection\n\n")
        f.write("This run estimates coarse Bonaca-style distance-modulus anchors in 2-degree steps, ")
        f.write("interpolates a smooth DM(phi1) track, and re-runs the strict member selection.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- z-parent sample after strict mag + z-locus: **{int(zparent['count_z'][0]):,}**\n")
        f.write(f"- successful DM anchors: **{n_anchor_success} / {len(anchors)}**\n")
        f.write(f"- refined selected members: **{cutflow['refined_selected']:,}**\n")
        f.write(f"- DM(phi1=-15 deg): **{summary['dm_track']['dm_at_phi1_minus15']:.3f}**\n")
        f.write(f"- DM(phi1=0 deg): **{summary['dm_track']['dm_at_phi1_0']:.3f}**\n")
        f.write(f"- DM(phi1=+8 deg): **{summary['dm_track']['dm_at_phi1_plus8']:.3f}**\n")
        f.write("\n## QC files\n\n")
        f.write("- `plots_step4/qc_step4_dm_track.png`\n")
        f.write("- `plots_step4/qc_step4_segment_cmds.png`\n")
        f.write("- `plots_step4/qc_step4_local_compare.png`\n")
        f.write("- `plots_step4/qc_step4_selected_density_phi12.png`\n")
        f.write("- `plots_step4/qc_step4_selected_density_radec.png`\n")

    print("[done] step 4 refined-distance selection complete")
    print(f"[outdir] {args.outdir}")


if __name__ == "__main__":
    main()
