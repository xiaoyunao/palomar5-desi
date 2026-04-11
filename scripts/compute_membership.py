from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from pal5_common import (
    LOG2PI,
    SegmentModel,
    assign_bins,
    build_cmd_pdf,
    build_segments,
    copy_fits_with_added_columns,
    logistic_from_logratio,
    lookup_pdf2d,
    quantile_bin_edges,
    read_parsec_isochrone,
    save_npz,
)


@dataclass(frozen=True)
class BackgroundModel:
    pdf1: np.ndarray
    pdf2: np.ndarray
    c1_edges: np.ndarray
    m1_edges: np.ndarray
    c2_edges: np.ndarray
    m2_edges: np.ndarray
    phi1_edges: np.ndarray
    depth_edges: np.ndarray
    area1: float
    area2: float


def phi2_track(phi1: np.ndarray) -> np.ndarray:
    return 0.011 * phi1 * phi1 + 0.10 * phi1 + 0.10


def sigma_phi2(phi1: np.ndarray) -> np.ndarray:
    width = np.sqrt(0.20**2 + (0.05 * np.abs(phi1)) ** 2)
    return np.clip(width, 0.10, 1.00)


def log_gaussian_phi2(phi1: np.ndarray, phi2: np.ndarray, track_fn=None) -> np.ndarray:
    if track_fn is None:
        track_center = phi2_track(phi1)
    else:
        track_center = track_fn(phi1)
    sig = sigma_phi2(phi1)
    delta = phi2 - track_center
    return -0.5 * (delta / sig) ** 2 - np.log(sig) - 0.5 * LOG2PI


def build_background_model(
    data: np.ndarray,
    phi1_bins: int,
    depth_bins: int,
    sideband_inner: float,
    sideband_outer: float,
    smooth_sigma: float,
) -> BackgroundModel:
    phi1_all = np.asarray(data["PHI1"], dtype=np.float64)
    phi2_all = np.asarray(data["PHI2"], dtype=np.float64)
    depth_all = np.asarray(data["PSFDEPTH_G"], dtype=np.float64)
    g0_all = np.asarray(data["G0"], dtype=np.float64)
    r0_all = np.asarray(data["R0"], dtype=np.float64)
    z0_all = np.asarray(data["Z0"], dtype=np.float64)

    offset = np.abs(phi2_all - phi2_track(phi1_all))
    mask = (
        np.isfinite(phi1_all)
        & np.isfinite(phi2_all)
        & np.isfinite(depth_all)
        & np.isfinite(g0_all)
        & np.isfinite(r0_all)
        & np.isfinite(z0_all)
        & (offset >= sideband_inner)
        & (offset <= sideband_outer)
    )
    if not np.any(mask):
        raise RuntimeError("No off-stream stars available to build the background model")

    phi1_bg = phi1_all[mask]
    depth_bg = depth_all[mask]
    g0_bg = g0_all[mask]
    r0_bg = r0_all[mask]
    z0_bg = z0_all[mask]

    phi1_edges = np.linspace(np.nanmin(phi1_bg), np.nanmax(phi1_bg), phi1_bins + 1)
    depth_edges = quantile_bin_edges(depth_bg, depth_bins)
    pdf1 = np.empty((phi1_bins, depth_bins), dtype=object)
    pdf2 = np.empty((phi1_bins, depth_bins), dtype=object)

    c1_range = (-0.5, 1.0)
    m1_range = (14.0, 24.8)
    c2_range = (-0.5, 1.0)
    m2_range = (14.0, 24.8)

    phi1_idx = assign_bins(phi1_bg, phi1_edges)
    depth_idx = assign_bins(depth_bg, depth_edges)

    fallback1 = build_cmd_pdf(g0_bg - r0_bg, g0_bg, c1_range, m1_range, 140, 160, smooth_sigma)
    fallback2 = build_cmd_pdf(r0_bg - z0_bg, r0_bg, c2_range, m2_range, 140, 160, smooth_sigma)

    for i_phi in range(phi1_bins):
        for i_depth in range(depth_bins):
            sel = (phi1_idx == i_phi) & (depth_idx == i_depth)
            if np.sum(sel) < 10_000:
                pdf1[i_phi, i_depth] = fallback1
                pdf2[i_phi, i_depth] = fallback2
                continue
            pdf1[i_phi, i_depth] = build_cmd_pdf(
                g0_bg[sel] - r0_bg[sel],
                g0_bg[sel],
                c1_range,
                m1_range,
                140,
                160,
                smooth_sigma,
            )
            pdf2[i_phi, i_depth] = build_cmd_pdf(
                r0_bg[sel] - z0_bg[sel],
                r0_bg[sel],
                c2_range,
                m2_range,
                140,
                160,
                smooth_sigma,
            )

    c1_edges = fallback1[1]
    m1_edges = fallback1[2]
    c2_edges = fallback2[1]
    m2_edges = fallback2[2]
    area1 = np.log((c1_edges[-1] - c1_edges[0]) / (len(c1_edges) - 1) * (m1_edges[-1] - m1_edges[0]) / (len(m1_edges) - 1))
    area2 = np.log((c2_edges[-1] - c2_edges[0]) / (len(c2_edges) - 1) * (m2_edges[-1] - m2_edges[0]) / (len(m2_edges) - 1))
    return BackgroundModel(
        pdf1=pdf1,
        pdf2=pdf2,
        c1_edges=c1_edges,
        m1_edges=m1_edges,
        c2_edges=c2_edges,
        m2_edges=m2_edges,
        phi1_edges=phi1_edges,
        depth_edges=depth_edges,
        area1=area1,
        area2=area2,
    )


def logp_bg(
    g0: np.ndarray,
    r0: np.ndarray,
    z0: np.ndarray,
    phi1: np.ndarray,
    depth_g: np.ndarray,
    background: BackgroundModel,
) -> np.ndarray:
    i_phi = assign_bins(phi1, background.phi1_edges)
    i_depth = assign_bins(depth_g, background.depth_edges)
    out = np.empty_like(g0, dtype=np.float64)
    c1 = g0 - r0
    m1 = g0
    c2 = r0 - z0
    m2 = r0
    for idx in range(len(out)):
        pdf1, c1_edges, m1_edges = background.pdf1[i_phi[idx], i_depth[idx]]
        pdf2, c2_edges, m2_edges = background.pdf2[i_phi[idx], i_depth[idx]]
        p1 = lookup_pdf2d(np.array([c1[idx]]), np.array([m1[idx]]), pdf1, c1_edges, m1_edges)[0]
        p2 = lookup_pdf2d(np.array([c2[idx]]), np.array([m2[idx]]), pdf2, c2_edges, m2_edges)[0]
        out[idx] = np.log(p1) - background.area1 + np.log(p2) - background.area2
    return out


def logp_iso_segments_1cmd(
    color_obs: np.ndarray,
    mag_obs: np.ndarray,
    err_color_1: np.ndarray,
    err_color_2: np.ndarray,
    seg: SegmentModel,
    distance_modulus: float,
    sigma_int: float,
    star_block: int = 20000,
    seg_block: int = 2048,
) -> np.ndarray:
    c0 = seg.c0
    m0 = seg.m0 + distance_modulus
    dc = seg.dc
    dm = seg.dm

    v1 = err_color_1 * err_color_1
    v2 = err_color_2 * err_color_2
    sig2 = sigma_int * sigma_int

    var_c = v1 + v2 + sig2
    var_m = v1 + sig2
    cov_cm = v1
    det = np.maximum(var_c * var_m - cov_cm * cov_cm, 1e-20)

    inv00 = var_m / det
    inv11 = var_c / det
    inv01 = -cov_cm / det
    lognorm = -LOG2PI - 0.5 * np.log(det)

    out = np.full(color_obs.shape[0], -np.inf, dtype=np.float64)
    nseg = c0.shape[0]
    for s0 in range(0, color_obs.shape[0], star_block):
        s1 = min(s0 + star_block, color_obs.shape[0])
        color_block = color_obs[s0:s1]
        mag_block = mag_obs[s0:s1]
        inv00_block = inv00[s0:s1]
        inv11_block = inv11[s0:s1]
        inv01_block = inv01[s0:s1]
        lognorm_block = lognorm[s0:s1]
        best = np.full(s1 - s0, -np.inf, dtype=np.float64)

        for j0 in range(0, nseg, seg_block):
            j1 = min(j0 + seg_block, nseg)
            c0b = c0[j0:j1][None, :]
            m0b = m0[j0:j1][None, :]
            dcb = dc[j0:j1][None, :]
            dmb = dm[j0:j1][None, :]

            dx = color_block[:, None] - c0b
            dy = mag_block[:, None] - m0b
            inv00x = inv00_block[:, None]
            inv11x = inv11_block[:, None]
            inv01x = inv01_block[:, None]

            denom = np.maximum(inv00x * dcb * dcb + inv11x * dmb * dmb + 2.0 * inv01x * dcb * dmb, 1e-30)
            numer = inv00x * dcb * dx + inv11x * dmb * dy + inv01x * (dcb * dy + dmb * dx)
            t = np.clip(numer / denom, 0.0, 1.0)

            rx = dx - t * dcb
            ry = dy - t * dmb
            chi2 = inv00x * rx * rx + inv11x * ry * ry + 2.0 * inv01x * rx * ry
            if sigma_int > 0.0:
                chi2 += (rx * rx + ry * ry) / (sigma_int * sigma_int)
            block_best = np.max(0.3 * lognorm_block[:, None] - 0.5 * chi2, axis=1)
            best = np.maximum(best, block_best)
        out[s0:s1] = best
    return out


def logp_iso(
    g0: np.ndarray,
    r0: np.ndarray,
    z0: np.ndarray,
    err_g: np.ndarray,
    err_r: np.ndarray,
    err_z: np.ndarray,
    seg1: SegmentModel,
    seg2: SegmentModel,
    distance_modulus: float,
    sigma_int: float,
) -> np.ndarray:
    lp1 = logp_iso_segments_1cmd(g0 - r0, g0, err_g, err_r, seg1, distance_modulus, sigma_int)
    lp2 = logp_iso_segments_1cmd(r0 - z0, r0, err_r, err_z, seg2, distance_modulus, sigma_int)
    return 0.5 * (lp1 + lp2)


def fit_track_from_cmd(
    phi1: np.ndarray,
    phi2: np.ndarray,
    p_iso: np.ndarray,
    min_prob: float,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi1_all = phi1
    phi2_all = phi2
    base = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(p_iso)
    if not np.any(base):
        raise RuntimeError("No finite stars available to fit a data-driven track")
    threshold = min_prob
    mask = base & (p_iso >= threshold)
    if np.sum(mask) < 100:
        threshold = max(0.2, float(np.nanquantile(p_iso[base], 0.90)))
        mask = base & (p_iso >= threshold)
    if np.sum(mask) < 100:
        mask = base & (p_iso >= float(np.nanquantile(p_iso[base], 0.75)))
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    weights = p_iso[mask]
    if phi1.size < 100:
        centers = np.arange(np.floor(np.nanmin(phi1_all[base])), np.ceil(np.nanmax(phi1_all[base])) + bin_width, bin_width)
        centers = 0.5 * (centers[:-1] + centers[1:])
        return centers, phi2_track(centers)

    edges = np.arange(np.floor(np.min(phi1)), np.ceil(np.max(phi1)) + bin_width, bin_width)
    centers = []
    track = []
    for left, right in zip(edges[:-1], edges[1:]):
        sel = (phi1 >= left) & (phi1 < right)
        if np.sum(sel) < 30:
            continue
        w_sel = weights[sel]
        if not np.any(np.isfinite(w_sel)) or np.nansum(w_sel) <= 0:
            continue
        centers.append(0.5 * (left + right))
        track.append(np.average(phi2[sel], weights=w_sel))
    centers = np.asarray(centers)
    track = np.asarray(track)
    if centers.size < 6:
        centers = np.arange(np.floor(np.nanmin(phi1_all[base])), np.ceil(np.nanmax(phi1_all[base])) + bin_width, bin_width)
        centers = 0.5 * (centers[:-1] + centers[1:])
        return centers, phi2_track(centers)
    return centers, track


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pal 5 photometric and spatial membership probabilities.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--isochrone", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diag-output", required=True)
    parser.add_argument("--distance-modulus", type=float, default=16.635)
    parser.add_argument("--sigma-int-mag", type=float, default=0.03)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--phi1-bins", type=int, default=18)
    parser.add_argument("--depth-bins", type=int, default=4)
    parser.add_argument("--sideband-inner", type=float, default=1.5)
    parser.add_argument("--sideband-outer", type=float, default=4.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--logk-cmd", type=float, default=0.0)
    parser.add_argument("--logk-mem", type=float, default=0.0)
    args = parser.parse_args()

    with fits.open(args.input, memmap=True) as hdul:
        data = hdul[1].data
        n_rows = len(data)
        print(f"input rows: {n_rows:,}")

        iso = read_parsec_isochrone(args.isochrone)
        seg1 = build_segments(iso.c1, iso.m1)
        seg2 = build_segments(iso.c2, iso.m2)

        background = build_background_model(
            data,
            phi1_bins=args.phi1_bins,
            depth_bins=args.depth_bins,
            sideband_inner=args.sideband_inner,
            sideband_outer=args.sideband_outer,
            smooth_sigma=args.smooth_sigma,
        )
        print("built local background CMD model")

        phi1_all = np.asarray(data["PHI1"], dtype=np.float64)
        phi2_all = np.asarray(data["PHI2"], dtype=np.float64)
        g0_all = np.asarray(data["G0"], dtype=np.float64)
        r0_all = np.asarray(data["R0"], dtype=np.float64)
        z0_all = np.asarray(data["Z0"], dtype=np.float64)
        eg_all = np.asarray(data["MAGERR_G"], dtype=np.float64)
        er_all = np.asarray(data["MAGERR_R"], dtype=np.float64)
        ez_all = np.asarray(data["MAGERR_Z"], dtype=np.float64)
        depth_all = np.asarray(data["PSFDEPTH_G"], dtype=np.float64)

        logp_iso_all = np.full(n_rows, np.nan, dtype=np.float32)
        logp_bg_all = np.full(n_rows, np.nan, dtype=np.float32)
        p_iso_all = np.full(n_rows, np.nan, dtype=np.float32)

        for start in range(0, n_rows, args.chunk_size):
            stop = min(start + args.chunk_size, n_rows)
            sel = slice(start, stop)
            ok = (
                np.isfinite(phi1_all[sel])
                & np.isfinite(phi2_all[sel])
                & np.isfinite(g0_all[sel])
                & np.isfinite(r0_all[sel])
                & np.isfinite(z0_all[sel])
                & np.isfinite(eg_all[sel])
                & np.isfinite(er_all[sel])
                & np.isfinite(ez_all[sel])
                & np.isfinite(depth_all[sel])
                & (eg_all[sel] > 0.0)
                & (er_all[sel] > 0.0)
                & (ez_all[sel] > 0.0)
            )
            if not np.any(ok):
                continue

            idx = np.where(ok)[0] + start
            logp_iso_chunk = logp_iso(
                g0_all[idx],
                r0_all[idx],
                z0_all[idx],
                eg_all[idx],
                er_all[idx],
                ez_all[idx],
                seg1,
                seg2,
                args.distance_modulus,
                args.sigma_int_mag,
            )
            logp_bg_chunk = logp_bg(
                g0_all[idx],
                r0_all[idx],
                z0_all[idx],
                phi1_all[idx],
                depth_all[idx],
                background,
            )
            p_iso_chunk = logistic_from_logratio(logp_iso_chunk, logp_bg_chunk + args.logk_cmd)
            logp_iso_all[idx] = logp_iso_chunk.astype(np.float32)
            logp_bg_all[idx] = logp_bg_chunk.astype(np.float32)
            p_iso_all[idx] = p_iso_chunk.astype(np.float32)
            print(f"computed photometric likelihoods for rows {start:,}-{stop:,}")

        track_phi1, track_phi2 = fit_track_from_cmd(phi1_all, phi2_all, p_iso_all, min_prob=0.5, bin_width=1.0)
        poly_coeff = np.polyfit(track_phi1, track_phi2, deg=min(3, len(track_phi1) - 1))
        poly_model = np.poly1d(poly_coeff)
        logphi_all = log_gaussian_phi2(phi1_all, phi2_all, track_fn=poly_model)
        logphi_ref = np.max(logphi_all[np.isfinite(logphi_all)])
        logphi_all -= logphi_ref
        p_mem_all = logistic_from_logratio(logp_iso_all.astype(np.float64) + logphi_all, logp_bg_all.astype(np.float64) + args.logk_mem)

    copy_fits_with_added_columns(
        args.input,
        args.output,
        [
            fits.Column(name="LOGP_ISO", format="E", array=logp_iso_all),
            fits.Column(name="LOGP_BG", format="E", array=logp_bg_all),
            fits.Column(name="LOGP_PHI2", format="E", array=logphi_all.astype(np.float32)),
            fits.Column(name="P_ISO", format="E", array=p_iso_all),
            fits.Column(name="P_MEM", format="E", array=p_mem_all.astype(np.float32)),
        ],
    )
    save_npz(
        args.diag_output,
        track_phi1=track_phi1,
        track_phi2=track_phi2,
        poly_coeff=poly_coeff,
        spline_phi1=np.linspace(np.min(track_phi1), np.max(track_phi1), 512),
        spline_phi2=poly_model(np.linspace(np.min(track_phi1), np.max(track_phi1), 512)),
        bg_phi1_edges=background.phi1_edges,
        bg_depth_edges=background.depth_edges,
    )
    print(f"wrote {args.output}")
    print(f"wrote diagnostics {args.diag_output}")


if __name__ == "__main__":
    main()
