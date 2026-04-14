#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from gala.coordinates import Pal5PriceWhelan18
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d


@dataclass
class Step2Config:
    gmin: float
    gmax: float
    zloc_tol: float
    cmd_w0: float
    cmd_w_slope: float
    cmd_w_min: float
    cmd_w_max: float
    dm_cluster_best: float
    dm_trailing_best: float
    dm_leading_best: float
    dc0: float
    dc1: float
    dmu: float


@dataclass
class IsochroneData:
    g_abs: np.ndarray
    r_abs: np.ndarray
    z_abs: np.ndarray


@dataclass
class AnchorPoint:
    phi1: float
    dm: float
    err: float
    kind: str
    weight: float
    meta: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Refine the Pal 5 distance-modulus track by combining the step4b-like "
            "photometric MSTO anchors with Price-Whelan+2019 RR Lyrae weak priors, "
            "then re-run the strict variable-DM member selection."
        )
    )
    p.add_argument("--preproc", default="final_g25_preproc.fits")
    p.add_argument("--step2-summary", default="step2_outputs/pal5_step2_summary.json")
    p.add_argument("--iso", default="pal5.dat")
    p.add_argument("--mu-prior-file", default="step3b_outputs_control/pal5_step3b_mu_prior.txt")
    p.add_argument(
        "--rrl-anchor-csv",
        default="pal5_rrl_price_whelan_2019_subset.csv",
        help="Curated subset of the Price-Whelan+2019 Table 2 catalog.",
    )
    p.add_argument(
        "--rrl-cache-csv",
        default="step4c_outputs/pal5_step4c_rrl_enriched.csv",
        help="Cache for the Gaia-position-enriched RRL subset.",
    )
    p.add_argument("--output-dir", default="step4c_outputs")
    p.add_argument("--output-members", default="step4c_outputs/pal5_step4c_rrlprior_members.fits")

    p.add_argument("--phi1-min", type=float, default=-20.0)
    p.add_argument("--phi1-max", type=float, default=10.0)
    p.add_argument("--anchor-step", type=float, default=2.0)
    p.add_argument("--anchor-window-half", type=float, default=1.5)
    p.add_argument("--on-halfwidth", type=float, default=0.4)
    p.add_argument("--off-inner", type=float, default=0.8)
    p.add_argument("--off-outer", type=float, default=1.6)

    p.add_argument("--dm-scan-half", type=float, default=0.45)
    p.add_argument("--dm-scan-step", type=float, default=0.01)

    p.add_argument("--msto-gmin", type=float, default=19.8)
    p.add_argument("--msto-gmax", type=float, default=21.7)
    p.add_argument("--anchor-blue-cut", type=float, default=0.15)
    p.add_argument("--anchor-blue-gmin", type=float, default=21.5)
    p.add_argument(
        "--anchor-blue-downweight",
        type=float,
        default=0.2,
        help="Downweight for the blue residual sequence when scoring photometric anchors.",
    )
    p.add_argument("--cmd-alignment-pivot", type=float, default=20.5)

    p.add_argument(
        "--rrl-stream-min-prob",
        type=float,
        default=0.8,
        help="Only stream RRLs with membership probability >= this enter the weak-prior fit.",
    )
    p.add_argument(
        "--rrl-stream-sigma-mag",
        type=float,
        default=0.14,
        help="Soft uncertainty assigned to stream RRL pseudo-DM anchors in mag.",
    )
    p.add_argument(
        "--rrl-cluster-sigma-mag",
        type=float,
        default=0.05,
        help="Soft uncertainty for the cluster anchor point in mag.",
    )
    p.add_argument(
        "--photometric-sigma-floor",
        type=float,
        default=0.12,
        help="Minimum uncertainty assigned to photometric anchors in mag.",
    )
    p.add_argument(
        "--photometric-sigma-ceiling",
        type=float,
        default=0.25,
        help="Maximum uncertainty assigned to photometric anchors in mag.",
    )
    p.add_argument(
        "--spline-smoothing-scale",
        type=float,
        default=0.8,
        help="Multiplier for the UnivariateSpline smoothing factor.",
    )
    p.add_argument(
        "--spline-clip-sigma",
        type=float,
        default=2.8,
        help="Sigma clipping threshold used on combined anchor residuals.",
    )
    p.add_argument(
        "--allow-gaia-query",
        action="store_true",
        help="Query Gaia DR2 by source_id to enrich the small RRL subset with RA/Dec if needed.",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_step2_config(path: str | Path) -> Step2Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    strict = data["strict_config"]
    align = data["alignment"]
    return Step2Config(
        gmin=float(strict["STRICT_GMIN"]),
        gmax=float(strict["STRICT_GMAX"]),
        zloc_tol=float(strict["ZLOCUS_TOL"]),
        cmd_w0=float(strict["CMD_W0"]),
        cmd_w_slope=float(strict["CMD_W_SLOPE"]),
        cmd_w_min=float(strict["CMD_W_MIN"]),
        cmd_w_max=float(strict["CMD_W_MAX"]),
        dm_cluster_best=float(align["dm_cluster_best"]),
        dm_trailing_best=float(align["dm_trailing_best"]),
        dm_leading_best=float(align["dm_leading_best"]),
        dc0=float(align["dc0"]),
        dc1=float(align["dc1"]),
        dmu=float(align["dmu"]),
    )


def load_mu_prior(path: str | Path) -> interp1d:
    arr = np.loadtxt(path)
    x = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1], dtype=float)
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")


def load_isochrone(path: str | Path) -> IsochroneData:
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Zini"):
                header = line.lstrip("#").strip().split()
                break
    if header is None:
        raise RuntimeError("Could not find the '# Zini ...' header in the isochrone file.")
    name_to_idx = {name: i for i, name in enumerate(header)}
    for req in ("label", "g_f0", "r_f0", "z_f0"):
        if req not in name_to_idx:
            raise KeyError(f"Isochrone column '{req}' not found.")

    arr = np.loadtxt(path, comments="#")
    label = arr[:, name_to_idx["label"]].astype(int)
    keep = np.isin(label, [1, 2, 3])
    arr = arr[keep]
    return IsochroneData(
        g_abs=np.asarray(arr[:, name_to_idx["g_f0"]], dtype=float),
        r_abs=np.asarray(arr[:, name_to_idx["r_f0"]], dtype=float),
        z_abs=np.asarray(arr[:, name_to_idx["z_f0"]], dtype=float),
    )


def build_cmd_interpolator(
    iso: IsochroneData,
    dm: float,
    cfg: Step2Config,
    pivot_mag: float,
) -> tuple[np.ndarray, np.ndarray]:
    g_app = iso.g_abs + dm
    c = (iso.g_abs - iso.r_abs) + cfg.dc0 + cfg.dc1 * (g_app - pivot_mag)
    order = np.argsort(g_app)
    g_sorted = g_app[order]
    c_sorted = c[order]
    uniq_mask = np.concatenate([[True], np.diff(g_sorted) > 1e-6])
    return g_sorted[uniq_mask], c_sorted[uniq_mask]


def color_residual(
    g0: np.ndarray,
    gr0: np.ndarray,
    dm: float,
    iso: IsochroneData,
    cfg: Step2Config,
    pivot_mag: float,
) -> np.ndarray:
    g_iso, c_iso = build_cmd_interpolator(iso, dm, cfg, pivot_mag)
    cintrp = np.interp(g0, g_iso, c_iso, left=np.nan, right=np.nan)
    return gr0 - cintrp


def cmd_halfwidth(g0: np.ndarray, cfg: Step2Config) -> np.ndarray:
    w = cfg.cmd_w0 + cfg.cmd_w_slope * (g0 - cfg.gmin)
    return np.clip(w, cfg.cmd_w_min, cfg.cmd_w_max)


def select_zparent(t: Table, cfg: Step2Config) -> Table:
    g0 = np.asarray(t["G0"], dtype=float)
    r0 = np.asarray(t["R0"], dtype=float)
    z0 = np.asarray(t["Z0"], dtype=float)
    gr0 = g0 - r0
    gz0 = g0 - z0
    zloc = 1.7 * gr0 - 0.17

    m = np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)
    m &= (g0 > cfg.gmin) & (g0 < cfg.gmax)
    m &= np.abs(gz0 - zloc) <= cfg.zloc_tol
    return t[m]


def default_dm_guess(phi1: float, cfg: Step2Config) -> float:
    if abs(phi1) < 1.0:
        return cfg.dm_cluster_best
    return cfg.dm_trailing_best if phi1 < 0 else cfg.dm_leading_best


def photometric_anchor_score(
    sub: Table,
    mu_center: float,
    dm: float,
    iso: IsochroneData,
    cfg: Step2Config,
    args: argparse.Namespace,
) -> float:
    g0 = np.asarray(sub["G0"], dtype=float)
    r0 = np.asarray(sub["R0"], dtype=float)
    phi2 = np.asarray(sub["PHI2"], dtype=float)
    gr0 = g0 - r0

    resid = color_residual(g0, gr0, dm, iso, cfg, args.cmd_alignment_pivot)
    wcmd = cmd_halfwidth(g0, cfg)
    base_w = np.exp(-0.5 * (resid / wcmd) ** 2)
    base_w[~np.isfinite(base_w)] = 0.0

    msto = (g0 >= args.msto_gmin) & (g0 <= args.msto_gmax)
    base_w *= msto.astype(float)

    blue_resid = (gr0 < args.anchor_blue_cut) & (g0 > args.anchor_blue_gmin)
    base_w[blue_resid] *= args.anchor_blue_downweight

    dy = phi2 - mu_center
    on = np.abs(dy) <= args.on_halfwidth
    off = ((np.abs(dy) >= args.off_inner) & (np.abs(dy) < args.off_outer))

    on_sum = float(np.sum(base_w[on]))
    off_sum = float(np.sum(base_w[off]))
    on_width = 2.0 * args.on_halfwidth
    off_width = 2.0 * (args.off_outer - args.off_inner)
    scale = on_width / off_width if off_width > 0 else 0.5
    return on_sum - scale * off_sum


def estimate_photometric_anchors(
    zparent: Table,
    mu_interp: interp1d,
    iso: IsochroneData,
    cfg: Step2Config,
    args: argparse.Namespace,
) -> tuple[list[AnchorPoint], dict[str, np.ndarray]]:
    centers = np.arange(args.phi1_min + 1.0, args.phi1_max, args.anchor_step)
    all_phi1 = np.asarray(zparent["PHI1"], dtype=float)
    anchors: list[AnchorPoint] = []
    score_cache: dict[str, np.ndarray] = {}

    for c in centers:
        local = zparent[np.abs(all_phi1 - c) <= args.anchor_window_half]
        if len(local) < 200:
            continue
        mu0 = float(mu_interp(c))
        dm0 = default_dm_guess(c, cfg)
        dm_grid = np.arange(dm0 - args.dm_scan_half, dm0 + args.dm_scan_half + 0.5 * args.dm_scan_step, args.dm_scan_step)
        scores = np.array([
            photometric_anchor_score(local, mu0, float(dm), iso, cfg, args) for dm in dm_grid
        ])
        if not np.isfinite(scores).any():
            continue
        k = int(np.nanargmax(scores))
        dm_best = float(dm_grid[k])

        rel = scores - np.nanmin(scores)
        rel[~np.isfinite(rel)] = 0.0
        if np.nanmax(rel) > 0:
            rel /= np.nanmax(rel)
        core = rel > 0.6
        if np.sum(core) >= 3:
            dm_err = float(np.sqrt(np.average((dm_grid[core] - dm_best) ** 2, weights=rel[core])))
        else:
            dm_err = 0.15
        dm_err = float(np.clip(dm_err, args.photometric_sigma_floor, args.photometric_sigma_ceiling))

        anchors.append(AnchorPoint(phi1=float(c), dm=dm_best, err=dm_err, kind="photometric", weight=1.0))
        score_cache[f"{c:+05.1f}"] = np.vstack([dm_grid, scores]).T

    return anchors, score_cache


def load_rrl_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(c, 0.5)])


def query_gaia_dr2_positions(source_ids: Iterable[int]) -> Table:
    try:
        from astroquery.gaia import Gaia
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "astroquery is required for Gaia lookups. Install it with `pip install astroquery`, "
            "or pre-populate the RRL cache CSV with RA/Dec columns."
        ) from e

    ids = list(dict.fromkeys(int(x) for x in source_ids))
    if not ids:
        raise RuntimeError("No Gaia source_ids supplied for the RRL lookup.")

    chunks: list[Table] = []
    chunk_size = 200
    for i in range(0, len(ids), chunk_size):
        subset = ids[i : i + chunk_size]
        query = (
            "SELECT source_id, ra, dec "
            "FROM gaiadr2.gaia_source "
            f"WHERE source_id IN ({','.join(str(x) for x in subset)})"
        )
        job = Gaia.launch_job_async(query, dump_to_file=False)
        chunks.append(job.get_results())
    if len(chunks) == 1:
        return chunks[0]
    return Table(np.hstack([np.array(tab) for tab in chunks]))


def enrich_rrl_subset(
    rrl_rows: list[dict[str, str]],
    cache_csv: str | Path,
    allow_gaia_query: bool,
) -> Table:
    cache_path = Path(cache_csv)
    if cache_path.exists():
        return Table.read(cache_path)

    if not allow_gaia_query:
        raise RuntimeError(
            f"RRL cache file '{cache_path}' does not exist and --allow-gaia-query was not supplied."
        )

    q = query_gaia_dr2_positions([int(r["source_id"]) for r in rrl_rows])
    qmap = {int(row["source_id"]): row for row in q}

    pal5_frame = Pal5PriceWhelan18()
    out = Table()
    cols = {k: [] for k in rrl_rows[0].keys()}
    ra_list, dec_list, phi1_list, phi2_list, dm_mag_list = [], [], [], [], []

    for row in rrl_rows:
        sid = int(row["source_id"])
        if sid not in qmap:
            raise RuntimeError(f"Gaia DR2 source_id {sid} was not returned by the Gaia query.")
        qrow = qmap[sid]
        ra = float(qrow["ra"])
        dec = float(qrow["dec"])
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(pal5_frame)
        phi1 = float(c.phi1.to_value(u.deg))
        phi2 = float(c.phi2.to_value(u.deg))
        phi1 = ((phi1 + 180.0) % 360.0) - 180.0
        dm_mag = 5.0 * math.log10(float(row["dist_kpc"])) + 10.0

        for k, v in row.items():
            cols[k].append(v)
        ra_list.append(ra)
        dec_list.append(dec)
        phi1_list.append(phi1)
        phi2_list.append(phi2)
        dm_mag_list.append(dm_mag)

    for k, v in cols.items():
        out[k] = v
    out["RA_ICRS"] = np.asarray(ra_list, dtype=float)
    out["DE_ICRS"] = np.asarray(dec_list, dtype=float)
    out["PHI1"] = np.asarray(phi1_list, dtype=float)
    out["PHI2"] = np.asarray(phi2_list, dtype=float)
    out["DM_MAG"] = np.asarray(dm_mag_list, dtype=float)

    ensure_dir(cache_path.parent)
    out.write(cache_path, overwrite=True)
    return out


def build_rrl_pseudo_anchors(rrl: Table, cfg: Step2Config, args: argparse.Namespace) -> tuple[list[AnchorPoint], dict[str, float]]:
    role = np.asarray(rrl["role"], dtype=str)
    memb = np.asarray(rrl["member_prob"], dtype=float)
    dm_mag = np.asarray(rrl["DM_MAG"], dtype=float)
    phi1 = np.asarray(rrl["PHI1"], dtype=float)

    cluster_mask = role == "cluster"
    stream_mask = (role == "stream") & (memb >= args.rrl_stream_min_prob)

    cluster_dm = weighted_median(dm_mag[cluster_mask], np.clip(memb[cluster_mask], 1e-3, None))
    zero_point = cfg.dm_cluster_best - cluster_dm

    anchors: list[AnchorPoint] = [
        AnchorPoint(
            phi1=0.0,
            dm=cfg.dm_cluster_best,
            err=args.rrl_cluster_sigma_mag,
            kind="rrl_cluster",
            weight=1.0,
            meta=f"cluster_dm_rrl={cluster_dm:.4f}, zp={zero_point:+.4f}",
        )
    ]

    for row in rrl[stream_mask]:
        prob = float(row["member_prob"])
        sig = max(args.rrl_stream_sigma_mag, 0.08) / math.sqrt(max(prob, 0.5))
        anchors.append(
            AnchorPoint(
                phi1=float(row["PHI1"]),
                dm=float(row["DM_MAG"]) + zero_point,
                err=float(sig),
                kind="rrl_stream",
                weight=float(prob),
                meta=str(row["source_id"]),
            )
        )

    info = {
        "cluster_dm_rrl_mag": float(cluster_dm),
        "zero_point_mag": float(zero_point),
        "n_rrl_total": int(len(rrl)),
        "n_rrl_stream_used": int(np.sum(stream_mask)),
    }
    return anchors, info


def robust_spline_fit(points: list[AnchorPoint], args: argparse.Namespace) -> tuple[UnivariateSpline, np.ndarray]:
    x = np.array([p.phi1 for p in points], dtype=float)
    y = np.array([p.dm for p in points], dtype=float)
    sig = np.array([p.err for p in points], dtype=float)
    w = 1.0 / np.clip(sig, 1e-3, None)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    keep = np.ones_like(x, dtype=bool)
    for _ in range(3):
        s = args.spline_smoothing_scale * np.sum(keep)
        spl = UnivariateSpline(x[keep], y[keep], w=w[keep], k=3, s=s)
        resid = (y - spl(x)) * w
        new_keep = np.abs(resid) <= args.spline_clip_sigma
        if np.all(new_keep == keep):
            keep = new_keep
            break
        keep = new_keep
    spl = UnivariateSpline(x[keep], y[keep], w=w[keep], k=3, s=args.spline_smoothing_scale * np.sum(keep))
    return spl, keep


def apply_variable_dm_selection(
    zparent: Table,
    dm_func,
    iso: IsochroneData,
    cfg: Step2Config,
    args: argparse.Namespace,
) -> Table:
    phi1 = np.asarray(zparent["PHI1"], dtype=float)
    g0 = np.asarray(zparent["G0"], dtype=float)
    r0 = np.asarray(zparent["R0"], dtype=float)
    gr0 = g0 - r0
    dm_star = np.asarray(dm_func(phi1), dtype=float)
    resid = np.empty(len(zparent), dtype=float)

    # Process in modest chunks to avoid rebuilding huge interpolation tables in loops.
    uniq_dm = np.round(dm_star / args.dm_scan_step) * args.dm_scan_step
    for dm_val in np.unique(uniq_dm):
        m = uniq_dm == dm_val
        resid[m] = color_residual(g0[m], gr0[m], float(dm_val), iso, cfg, args.cmd_alignment_pivot)

    width = cmd_halfwidth(g0, cfg)
    keep = np.isfinite(resid) & (np.abs(resid) <= width)

    out = zparent[keep]
    out["DM_TRACK"] = dm_star[keep]
    out["CMD_RESID"] = resid[keep]
    out["CMD_HALFWIDTH"] = width[keep]
    return out


def make_density_map(
    t: Table,
    xcol: str,
    ycol: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    binsize: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(t[xcol], dtype=float)
    y = np.asarray(t[ycol], dtype=float)
    xedges = np.arange(xlim[0], xlim[1] + binsize, binsize)
    yedges = np.arange(ylim[0], ylim[1] + binsize, binsize)
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    area = binsize * binsize
    return H.T / area, xedges, yedges


def plot_dm_track(
    phot_anchors: list[AnchorPoint],
    rrl_anchors: list[AnchorPoint],
    dm_func,
    cfg: Step2Config,
    out_png: str | Path,
) -> None:
    xs = np.linspace(-20, 10, 400)
    plt.figure(figsize=(10, 6))
    plt.axhline(cfg.dm_trailing_best, ls="--", color="C0", lw=1.5, label="step2 two-arm trailing")
    plt.axhline(cfg.dm_leading_best, ls=":", color="C0", lw=1.5, label="step2 two-arm leading")
    plt.axvline(0.0, color="0.5", lw=1.0)

    if phot_anchors:
        x = [p.phi1 for p in phot_anchors]
        y = [p.dm for p in phot_anchors]
        ye = [p.err for p in phot_anchors]
        plt.errorbar(x, y, yerr=ye, fmt="o", color="C1", label="MSTO photometric anchors")

    cluster = [p for p in rrl_anchors if p.kind == "rrl_cluster"]
    stream = [p for p in rrl_anchors if p.kind == "rrl_stream"]
    if cluster:
        plt.errorbar(
            [cluster[0].phi1],
            [cluster[0].dm],
            yerr=[cluster[0].err],
            fmt="s",
            color="C2",
            label="RRL cluster anchor (zero-point locked)",
        )
    if stream:
        plt.errorbar(
            [p.phi1 for p in stream],
            [p.dm for p in stream],
            yerr=[p.err for p in stream],
            fmt="^",
            color="C3",
            alpha=0.8,
            label="RRL stream weak priors",
        )

    plt.plot(xs, dm_func(xs), color="C4", lw=2.5, label="combined DM(phi1)")
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel("distance modulus")
    plt.title("step4c: MSTO photometric anchors + RR Lyrae weak priors")
    plt.legend(framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_rrl_on_phi12(signal: Table, rrl: Table, mu_interp: interp1d, out_png: str | Path) -> None:
    D, xedges, yedges = make_density_map(signal, "PHI1", "PHI2", (-20, 10), (-2.5, 2.5), 0.1)
    plt.figure(figsize=(9, 5.5))
    plt.imshow(
        D,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1.0, np.nanmin(D[D > 0]) if np.any(D > 0) else 1.0), vmax=np.nanmax(D) if np.nanmax(D) > 0 else 10.0),
    )
    xs = np.linspace(-20, 10, 200)
    mu = mu_interp(xs)
    plt.plot(xs, mu, color="C0", lw=2.0, label="step3b control prior track")
    role = np.asarray(rrl["role"], dtype=str)
    plt.scatter(rrl["PHI1"][role == "cluster"], rrl["PHI2"][role == "cluster"], s=45, marker="s", color="C2", label="RRL cluster")
    plt.scatter(rrl["PHI1"][role == "stream"], rrl["PHI2"][role == "stream"], s=45, marker="^", color="C3", label="RRL stream")
    plt.xlim(-20, 10)
    plt.ylim(-2.5, 2.5)
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step4c: Price-Whelan+2019 RR Lyrae anchors over the strict sample")
    plt.colorbar(label="counts / deg$^2$")
    plt.legend(framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_segment_cmds(
    zparent: Table,
    dm_func,
    iso: IsochroneData,
    cfg: Step2Config,
    mu_interp: interp1d,
    out_png: str | Path,
    args: argparse.Namespace,
) -> None:
    centers = [-19.0, -13.0, -7.0, -3.0, 3.0, 9.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=True)
    axes = axes.ravel()

    phi1 = np.asarray(zparent["PHI1"], dtype=float)
    phi2 = np.asarray(zparent["PHI2"], dtype=float)
    g0 = np.asarray(zparent["G0"], dtype=float)
    r0 = np.asarray(zparent["R0"], dtype=float)
    gr0 = g0 - r0

    for ax, c in zip(axes, centers):
        mu0 = float(mu_interp(c))
        seg = np.abs(phi1 - c) <= 1.5
        on = seg & (np.abs(phi2 - mu0) <= 0.4)
        off = seg & (np.abs(phi2 - mu0) >= 0.8) & (np.abs(phi2 - mu0) < 1.6)

        x_on = gr0[on]
        y_on = g0[on]
        x_off = gr0[off]
        y_off = g0[off]

        xedges = np.linspace(-0.3, 1.0, 66)
        yedges = np.linspace(16.0, 24.0, 81)
        Hon, _, _ = np.histogram2d(x_on, y_on, bins=[xedges, yedges])
        Hoff, _, _ = np.histogram2d(x_off, y_off, bins=[xedges, yedges])
        diff = Hon - 0.5 * Hoff
        diff = np.clip(diff, 0, None)

        ax.imshow(
            diff.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="viridis",
        )
        dm_here = float(dm_func(c))
        g_iso, c_iso = build_cmd_interpolator(iso, dm_here, cfg, args.cmd_alignment_pivot)
        m = (g_iso >= 16.0) & (g_iso <= 24.0)
        ax.plot(c_iso[m], g_iso[m], color="orange", lw=2.0)
        ax.set_title(rf"$\phi_1\approx {c:+.1f}^\circ$, DM={dm_here:.3f}")
        ax.set_xlabel(r"$(g-r)_0$")
        ax.set_ylabel(r"$g_0$")
        ax.set_xlim(-0.3, 1.0)
        ax.set_ylim(24.0, 16.0)
    fig.suptitle("step4c: local on-stream minus off-stream CMDs with the RRL-informed DM(track)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_selected_density(t: Table, out_phi12: str | Path, out_radec: str | Path) -> None:
    D, xedges, yedges = make_density_map(t, "PHI1", "PHI2", (-25, 20), (-5, 10), 0.1)
    plt.figure(figsize=(9, 6))
    plt.imshow(
        D,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1.0, np.nanmin(D[D > 0]) if np.any(D > 0) else 1.0), vmax=np.nanmax(D) if np.nanmax(D) > 0 else 10.0),
    )
    plt.xlabel(r"$\phi_1$ [deg]")
    plt.ylabel(r"$\phi_2$ [deg]")
    plt.title("step4c selected sample: Pal 5 frame number density (log scale)")
    plt.colorbar(label="counts / deg$^2$")
    plt.tight_layout()
    plt.savefig(out_phi12, dpi=200)
    plt.close()

    D, xedges, yedges = make_density_map(t, "RA", "DEC", (210, 256), (-20, 18), 0.1)
    plt.figure(figsize=(9, 6))
    plt.imshow(
        D,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1.0, np.nanmin(D[D > 0]) if np.any(D > 0) else 1.0), vmax=np.nanmax(D) if np.nanmax(D) > 0 else 10.0),
    )
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("step4c selected sample: RA-Dec number density (log scale)")
    plt.colorbar(label="counts / deg$^2$")
    plt.tight_layout()
    plt.savefig(out_radec, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    cfg = load_step2_config(args.step2_summary)
    mu_interp = load_mu_prior(args.mu_prior_file)
    iso = load_isochrone(args.iso)

    pre = Table.read(args.preproc)
    zparent = select_zparent(pre, cfg)

    phot_anchors, score_cache = estimate_photometric_anchors(zparent, mu_interp, iso, cfg, args)

    rrl_rows = load_rrl_csv(args.rrl_anchor_csv)
    rrl = enrich_rrl_subset(rrl_rows, args.rrl_cache_csv, args.allow_gaia_query)
    rrl_anchors, rrl_info = build_rrl_pseudo_anchors(rrl, cfg, args)

    combined = phot_anchors + rrl_anchors
    spline, keep_mask = robust_spline_fit(combined, args)
    dm_func = lambda x: spline(np.asarray(x, dtype=float))

    selected = apply_variable_dm_selection(zparent, dm_func, iso, cfg, args)
    ensure_dir(Path(args.output_members).parent)
    selected.write(args.output_members, overwrite=True)

    # Save anchors and diagnostics.
    phot_tab = Table()
    phot_tab["phi1"] = [p.phi1 for p in phot_anchors]
    phot_tab["dm"] = [p.dm for p in phot_anchors]
    phot_tab["err"] = [p.err for p in phot_anchors]
    phot_tab.write(Path(args.output_dir) / "pal5_step4c_photometric_anchors.csv", overwrite=True)

    comb_tab = Table()
    comb_tab["phi1"] = [p.phi1 for p in combined]
    comb_tab["dm"] = [p.dm for p in combined]
    comb_tab["err"] = [p.err for p in combined]
    comb_tab["kind"] = [p.kind for p in combined]
    comb_tab["meta"] = [p.meta for p in combined]
    comb_tab["used_in_spline"] = np.asarray(keep_mask, dtype=int)
    comb_tab.write(Path(args.output_dir) / "pal5_step4c_combined_anchors.csv", overwrite=True)

    phi_grid = np.arange(args.phi1_min, args.phi1_max + 0.01, 0.1)
    dm_grid = dm_func(phi_grid)
    dm_tab = Table()
    dm_tab["phi1"] = phi_grid
    dm_tab["dm"] = dm_grid
    dm_tab.write(Path(args.output_dir) / "pal5_step4c_dm_track.csv", overwrite=True)

    plot_dm_track(phot_anchors, rrl_anchors, dm_func, cfg, Path(args.output_dir) / "qc_step4c_dm_track.png")
    plot_rrl_on_phi12(selected, rrl, mu_interp, Path(args.output_dir) / "qc_step4c_rrl_phi12.png")
    plot_segment_cmds(zparent, dm_func, iso, cfg, mu_interp, Path(args.output_dir) / "qc_step4c_segment_cmds.png", args)
    plot_selected_density(
        selected,
        Path(args.output_dir) / "qc_step4c_selected_density_phi12.png",
        Path(args.output_dir) / "qc_step4c_selected_density_radec.png",
    )

    with open(Path(args.output_dir) / "pal5_step4c_scores.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in score_cache.items()}, f)

    summary = {
        "input": str(args.preproc),
        "step2_summary": str(args.step2_summary),
        "iso": str(args.iso),
        "mu_prior": str(args.mu_prior_file),
        "rrl_anchor_csv": str(args.rrl_anchor_csv),
        "rrl_cache_csv": str(args.rrl_cache_csv),
        "output_members": str(args.output_members),
        "n_zparent_total": int(len(zparent)),
        "n_photometric_anchors": int(len(phot_anchors)),
        "n_combined_anchors": int(len(combined)),
        "n_combined_anchors_used": int(np.sum(keep_mask)),
        "n_rrl_total": int(rrl_info["n_rrl_total"]),
        "n_rrl_stream_used": int(rrl_info["n_rrl_stream_used"]),
        "cluster_dm_rrl_mag": float(rrl_info["cluster_dm_rrl_mag"]),
        "rrl_zero_point_shift_mag": float(rrl_info["zero_point_mag"]),
        "dm_at_phi1_minus15": float(dm_func(-15.0)),
        "dm_at_phi1_0": float(dm_func(0.0)),
        "dm_at_phi1_plus8": float(dm_func(8.0)),
        "refined_selected": int(len(selected)),
        "cutflow": {
            "input": int(len(pre)),
            "strict_mag_z_locus": int(len(zparent)),
            "iso_variable_dm_rrlprior": int(len(selected)),
        },
        "notes": [
            "Photometric anchors are MSTO-weighted, following the step4b strategy.",
            "RR Lyrae distances are converted to pseudo-photometric DM anchors by locking the global zero-point to the step2 cluster DM.",
            "Only stream RRLs above the configured membership threshold are used as weak priors in the DM(phi1) fit.",
            "The output members should be fed back into step3b control+MAP to define the new working baseline.",
        ],
    }
    with open(Path(args.output_dir) / "pal5_step4c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = f"""# Pal 5 step 4c: RR Lyrae weak-prior distance-gradient refinement

This run combines two sources of information for the variable-DM member selection:

1. step4b-style MSTO-weighted photometric anchors; and
2. a small Price-Whelan+2019 RR Lyrae subset used as a weak prior on the *shape* of DM(phi1).

## Summary

- z-parent sample after strict mag + z-locus: **{len(zparent):,}**
- photometric anchors used: **{len(phot_anchors)}**
- RRL stars in curated subset: **{rrl_info['n_rrl_total']}**
- stream RRL weak priors used: **{rrl_info['n_rrl_stream_used']}**
- RRL cluster DM (weighted median): **{rrl_info['cluster_dm_rrl_mag']:.3f} mag**
- zero-point shift applied to RRL DMs: **{rrl_info['zero_point_mag']:+.3f} mag**
- DM(phi1=-15 deg): **{dm_func(-15.0):.3f}**
- DM(phi1=0 deg): **{dm_func(0.0):.3f}**
- DM(phi1=+8 deg): **{dm_func(8.0):.3f}**
- refined selected members: **{len(selected):,}**

## Files

- `pal5_step4c_summary.json`
- `pal5_step4c_dm_track.csv`
- `pal5_step4c_photometric_anchors.csv`
- `pal5_step4c_combined_anchors.csv`
- `qc_step4c_dm_track.png`
- `qc_step4c_rrl_phi12.png`
- `qc_step4c_segment_cmds.png`
- `qc_step4c_selected_density_phi12.png`
- `qc_step4c_selected_density_radec.png`
"""
    with open(Path(args.output_dir) / "pal5_step4c_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
