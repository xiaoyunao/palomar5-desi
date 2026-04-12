#!/usr/bin/env python3
"""
Palomar 5 preprocessing pipeline (Phase 0 baseline).

This script performs:
1. sky-region pre-cut in RA/Dec,
2. optional circular hole mask around the cluster center,
3. strict point-source cleaning aligned with Bonaca+2020,
4. extinction correction, preferring catalog transmission columns when present,
5. coordinate transform to the Pal 5 stream frame,
6. a single final faint-limit selection (G0 < 25),
7. merge of chunk products,
8. diagnostic plots and summary reports.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from dustmaps.sfd import SFDQuery
from gala.coordinates import Pal5PriceWhelan18
from matplotlib.colors import LogNorm


INPUT_FITS = "/pscratch/sd/y/yunao/palomar5/desi-dr10-palomar5-cat.fits"
OUTPUT_FITS = "pal5_preprocessed_glt25.fits"
SUMMARY_JSON = "pal5_preprocessed_summary.json"

TMP_DIR = "./tmp_pal5_preproc"
PLOT_DIR = "./diagnostics_pal5_preproc"
REPORT_DIR = "./reports_preproc"

ROW_CHUNK = 5_000_000
COORD_BATCH = 500_000
MERGE_BATCH = 20
PLOT_CHUNK = 2_000_000

RA_MIN, RA_MAX = 210.0, 260.0
DEC_MIN, DEC_MAX = -20.0, 20.0

PHI1_MIN, PHI1_MAX = -25.0, 20.0
PHI2_MIN, PHI2_MAX = -5.0, 10.0

APPLY_CLUSTER_HOLE = True
MASK_CENTER_RADEC = (229.64, 2.08)
MASK_RADIUS_DEG = 0.30

R_G, R_R, R_Z = 3.30, 2.31, 1.28

KEEP_TYPES = ("PSF",)
REQUIRE_ALLMASK_GR_ZERO = True
REQUIRE_BRIGHTSTAR_CLEAN = True
REQUIRE_BRICK_PRIMARY_IF_PRESENT = False

BRIGHTBLOB_REJECT_MASK = (1 << 0) | (1 << 1)
MASKBITS_BRIGHT_REJECT_MASK = (1 << 1) | (1 << 11)
USE_MASKBITS_BRIGHT_FALLBACK = True

G0_MIN = 12.0
G0_MAX = 25.0
USE_FAINT_DEPTH_GUARD = True
FAINT_BREAK_G = 24.0
DEPTH5_G_MIN_FOR_FAINT = 25.0

PSFDEPTH_IS_IVAR = True

DENSITY_BIN_DEG = 0.25
CMD_X_RANGE = (-0.3, 1.3)
CMD_Y_RANGE = (25.2, 12.0)
CMD_DX = 0.02
CMD_DY = 0.05
CC_X_RANGE = (-0.3, 1.3)
CC_Y_RANGE = (-0.4, 2.5)
CC_DX = 0.02
CC_DY = 0.02

PAL5_FRAME = Pal5PriceWhelan18()
SFD = SFDQuery()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("pal5_preprocess")

_RA0_RAD = np.deg2rad(MASK_CENTER_RADEC[0])
_DEC0_RAD = np.deg2rad(MASK_CENTER_RADEC[1])
_CLUSTER_UNIT_VECTOR = np.array(
    [
        np.cos(_DEC0_RAD) * np.cos(_RA0_RAD),
        np.cos(_DEC0_RAD) * np.sin(_RA0_RAD),
        np.sin(_DEC0_RAD),
    ],
    dtype=np.float64,
)
_CLUSTER_COS_RADIUS = np.cos(np.deg2rad(MASK_RADIUS_DEG))


def ensure_directories() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def build_name_map(names: Sequence[str]) -> Dict[str, str]:
    return {name.upper(): name for name in names}


def resolve_required(name_map: Mapping[str, str], *candidates: str) -> str:
    for candidate in candidates:
        key = candidate.upper()
        if key in name_map:
            return name_map[key]
    raise KeyError(f"Required column not found. Tried: {candidates}")


def resolve_optional(name_map: Mapping[str, str], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        key = candidate.upper()
        if key in name_map:
            return name_map[key]
    return None


def decode_type_column(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        out = np.char.decode(values, "ascii", errors="ignore")
    elif values.dtype.kind == "O":
        out = np.array(
            [x.decode("ascii", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x) for x in values],
            dtype="U8",
        )
    else:
        out = values.astype("U8")
    return np.char.strip(np.char.upper(out))


def outside_cluster_hole(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    cos_angle = (
        x * _CLUSTER_UNIT_VECTOR[0]
        + y * _CLUSTER_UNIT_VECTOR[1]
        + z * _CLUSTER_UNIT_VECTOR[2]
    )
    return cos_angle <= _CLUSTER_COS_RADIUS


def psfdepth_to_5sigma_abmag(psfdepth: np.ndarray) -> np.ndarray:
    psfdepth = np.asarray(psfdepth, dtype=np.float64)
    depth_mag = np.full(psfdepth.shape, np.nan, dtype=np.float32)
    good = np.isfinite(psfdepth) & (psfdepth > 0)
    if np.any(good):
        depth_mag[good] = (
            -2.5 * (np.log10(5.0 / np.sqrt(psfdepth[good])) - 9.0)
        ).astype(np.float32)
    return depth_mag


def get_depth5_g_mag(psfdepth_g: np.ndarray) -> np.ndarray:
    if PSFDEPTH_IS_IVAR:
        return psfdepth_to_5sigma_abmag(psfdepth_g)
    return np.asarray(psfdepth_g, dtype=np.float32)


def compute_pal5_coords(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    batch_size: int = COORD_BATCH,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(ra_deg)
    phi1 = np.empty(n, dtype=np.float32)
    phi2 = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        coords = SkyCoord(ra_deg[start:stop] * u.deg, dec_deg[start:stop] * u.deg, frame="icrs")
        coords_pal5 = coords.transform_to(PAL5_FRAME)
        p1 = coords_pal5.phi1.to_value(u.deg)
        p2 = coords_pal5.phi2.to_value(u.deg)
        phi1[start:stop] = (((p1 + 180.0) % 360.0) - 180.0).astype(np.float32)
        phi2[start:stop] = p2.astype(np.float32)

    return phi1, phi2


def compute_extinction(
    sub: fits.fitsrec.FITS_rec,
    columns: Mapping[str, Optional[str]],
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mwg = columns["MW_TRANSMISSION_G"]
    mwr = columns["MW_TRANSMISSION_R"]
    mwz = columns["MW_TRANSMISSION_Z"]

    if mwg is not None and mwr is not None and mwz is not None:
        tg = np.asarray(sub[mwg], dtype=np.float64)
        tr = np.asarray(sub[mwr], dtype=np.float64)
        tz = np.asarray(sub[mwz], dtype=np.float64)
        good = (
            np.isfinite(tg) & (tg > 0) & (tg <= 1)
            & np.isfinite(tr) & (tr > 0) & (tr <= 1)
            & np.isfinite(tz) & (tz > 0) & (tz <= 1)
        )
        if np.all(good):
            ag = (-2.5 * np.log10(tg)).astype(np.float32)
            ar = (-2.5 * np.log10(tr)).astype(np.float32)
            az = (-2.5 * np.log10(tz)).astype(np.float32)
            ebv = (ag / R_G).astype(np.float32)
            return ebv, ag, ar, az

    coords = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    ebv = SFD(coords).astype(np.float32)
    ag = (ebv * R_G).astype(np.float32)
    ar = (ebv * R_R).astype(np.float32)
    az = (ebv * R_Z).astype(np.float32)
    return ebv, ag, ar, az


def make_cutflow_dict() -> "OrderedDict[str, int]":
    return OrderedDict(
        [
            ("input", 0),
            ("after_radec_box", 0),
            ("after_cluster_hole", 0),
            ("after_type_psf", 0),
            ("after_legacy_quality", 0),
            ("after_finite_grz", 0),
            ("after_pal5_window", 0),
            ("after_g0_depth_limit", 0),
        ]
    )


def accumulate_cutflow(total: "OrderedDict[str, int]", part: Mapping[str, int]) -> None:
    for key in total:
        total[key] += int(part.get(key, 0))


def select_output_columns(table: Table) -> Table:
    preferred = [
        "ID", "RELEASE", "BRICKID", "BRICKNAME", "OBJID", "REF_ID", "REF_CAT",
        "RA", "DEC",
        "MAG_G", "MAG_R", "MAG_Z", "MAG_W1", "MAG_W2",
        "MAGERR_G", "MAGERR_R", "MAGERR_Z", "MAGERR_W1", "MAGERR_W2",
        "TYPE", "BRICK_PRIMARY", "MASKBITS", "FITBITS",
        "ALLMASK_G", "ALLMASK_R", "ALLMASK_Z",
        "ANYMASK_G", "ANYMASK_R", "ANYMASK_Z",
        "BRIGHTSTARINBLOB", "BRIGHTBLOB",
        "MW_TRANSMISSION_G", "MW_TRANSMISSION_R", "MW_TRANSMISSION_Z",
        "NOBS_G", "NOBS_R", "NOBS_Z",
        "PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_Z", "DEPTH5_G",
        "GAIA_PHOT_G_MEAN_MAG", "GAIA_PHOT_BP_MEAN_MAG", "GAIA_PHOT_RP_MEAN_MAG",
        "PARALLAX", "PARALLAX_IVAR", "PMRA", "PMRA_IVAR", "PMDEC", "PMDEC_IVAR",
        "MJD_MIN", "MJD_MAX",
        "EBV_SFD", "A_G", "A_R", "A_Z",
        "G0", "R0", "Z0", "GR0", "RZ0", "GZ0",
        "PHI1", "PHI2",
    ]
    keep = [col for col in preferred if col in table.colnames]
    return table[keep]


def process_chunk(
    sub: fits.fitsrec.FITS_rec,
    chunk_id: int,
    columns: Mapping[str, Optional[str]],
) -> Tuple[int, "OrderedDict[str, int]", Optional[str]]:
    cutflow = make_cutflow_dict()
    cutflow["input"] = len(sub)

    ra = np.asarray(sub[columns["RA"]], dtype=np.float64)
    dec = np.asarray(sub[columns["DEC"]], dtype=np.float64)

    mask = (
        (ra > RA_MIN)
        & (ra < RA_MAX)
        & (dec > DEC_MIN)
        & (dec < DEC_MAX)
    )
    if not np.any(mask):
        return 0, cutflow, None
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]
    cutflow["after_radec_box"] = len(sub)

    if APPLY_CLUSTER_HOLE:
        mask = outside_cluster_hole(ra, dec)
        if not np.any(mask):
            return 0, cutflow, None
        sub = sub[mask]
        ra = ra[mask]
        dec = dec[mask]
    cutflow["after_cluster_hole"] = len(sub)

    type_values = decode_type_column(np.asarray(sub[columns["TYPE"]]))
    mask = np.isin(type_values, KEEP_TYPES)
    if not np.any(mask):
        return 0, cutflow, None
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]
    cutflow["after_type_psf"] = len(sub)

    quality_mask = np.ones(len(sub), dtype=bool)
    if REQUIRE_BRICK_PRIMARY_IF_PRESENT and columns["BRICK_PRIMARY"] is not None:
        quality_mask &= np.asarray(sub[columns["BRICK_PRIMARY"]], dtype=bool)
    if REQUIRE_ALLMASK_GR_ZERO:
        if columns["ALLMASK_G"] is not None:
            quality_mask &= (np.asarray(sub[columns["ALLMASK_G"]]) == 0)
        if columns["ALLMASK_R"] is not None:
            quality_mask &= (np.asarray(sub[columns["ALLMASK_R"]]) == 0)
    if REQUIRE_BRIGHTSTAR_CLEAN:
        if columns["BRIGHTSTARINBLOB"] is not None:
            quality_mask &= ~np.asarray(sub[columns["BRIGHTSTARINBLOB"]], dtype=bool)
        elif columns["BRIGHTBLOB"] is not None:
            brightblob = np.asarray(sub[columns["BRIGHTBLOB"]])
            quality_mask &= ((brightblob & BRIGHTBLOB_REJECT_MASK) == 0)
        elif USE_MASKBITS_BRIGHT_FALLBACK and columns["MASKBITS"] is not None:
            maskbits = np.asarray(sub[columns["MASKBITS"]])
            quality_mask &= ((maskbits & MASKBITS_BRIGHT_REJECT_MASK) == 0)
    if not np.any(quality_mask):
        return 0, cutflow, None
    sub = sub[quality_mask]
    ra = ra[quality_mask]
    dec = dec[quality_mask]
    cutflow["after_legacy_quality"] = len(sub)

    g = np.asarray(sub[columns["MAG_G"]], dtype=np.float32)
    r = np.asarray(sub[columns["MAG_R"]], dtype=np.float32)
    z = np.asarray(sub[columns["MAG_Z"]], dtype=np.float32)
    psfdepth_g_raw = np.asarray(sub[columns["PSFDEPTH_G"]], dtype=np.float32)

    mask = np.isfinite(g) & np.isfinite(r) & np.isfinite(z) & np.isfinite(psfdepth_g_raw)
    if not np.any(mask):
        return 0, cutflow, None
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]
    g = g[mask]
    r = r[mask]
    z = z[mask]
    psfdepth_g_raw = psfdepth_g_raw[mask]
    cutflow["after_finite_grz"] = len(sub)

    ebv, ag, ar, az = compute_extinction(sub, columns, ra, dec)
    phi1, phi2 = compute_pal5_coords(ra, dec, batch_size=COORD_BATCH)

    g0 = (g - ag).astype(np.float32)
    r0 = (r - ar).astype(np.float32)
    z0 = (z - az).astype(np.float32)
    gr0 = (g0 - r0).astype(np.float32)
    rz0 = (r0 - z0).astype(np.float32)
    gz0 = (g0 - z0).astype(np.float32)
    depth5_g = get_depth5_g_mag(psfdepth_g_raw)

    mask = (
        (phi1 > PHI1_MIN)
        & (phi1 < PHI1_MAX)
        & (phi2 > PHI2_MIN)
        & (phi2 < PHI2_MAX)
    )
    if not np.any(mask):
        return 0, cutflow, None

    sub = sub[mask]
    ebv = ebv[mask]
    ag = ag[mask]
    ar = ar[mask]
    az = az[mask]
    g0 = g0[mask]
    r0 = r0[mask]
    z0 = z0[mask]
    gr0 = gr0[mask]
    rz0 = rz0[mask]
    gz0 = gz0[mask]
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    depth5_g = depth5_g[mask]
    cutflow["after_pal5_window"] = len(sub)

    final_mask = (g0 > G0_MIN) & (g0 < G0_MAX)
    if USE_FAINT_DEPTH_GUARD:
        bright_ok = g0 <= FAINT_BREAK_G
        faint_ok = (g0 > FAINT_BREAK_G) & np.isfinite(depth5_g) & (depth5_g >= DEPTH5_G_MIN_FOR_FAINT)
        final_mask &= (bright_ok | faint_ok)
    if not np.any(final_mask):
        return 0, cutflow, None

    sub = sub[final_mask]
    ebv = ebv[final_mask]
    ag = ag[final_mask]
    ar = ar[final_mask]
    az = az[final_mask]
    g0 = g0[final_mask]
    r0 = r0[final_mask]
    z0 = z0[final_mask]
    gr0 = gr0[final_mask]
    rz0 = rz0[final_mask]
    gz0 = gz0[final_mask]
    phi1 = phi1[final_mask]
    phi2 = phi2[final_mask]
    depth5_g = depth5_g[final_mask]
    cutflow["after_g0_depth_limit"] = len(sub)

    table = Table(sub)
    table["EBV_SFD"] = ebv.astype(np.float32)
    table["A_G"] = ag.astype(np.float32)
    table["A_R"] = ar.astype(np.float32)
    table["A_Z"] = az.astype(np.float32)
    table["DEPTH5_G"] = depth5_g.astype(np.float32)
    table["G0"] = g0.astype(np.float32)
    table["R0"] = r0.astype(np.float32)
    table["Z0"] = z0.astype(np.float32)
    table["GR0"] = gr0.astype(np.float32)
    table["RZ0"] = rz0.astype(np.float32)
    table["GZ0"] = gz0.astype(np.float32)
    table["PHI1"] = phi1.astype(np.float32)
    table["PHI2"] = phi2.astype(np.float32)

    table = select_output_columns(table)
    tmp_path = os.path.join(TMP_DIR, f"tmp_pal5_preproc_chunk{chunk_id:04d}.fits")
    table.write(tmp_path, overwrite=True)
    return len(table), cutflow, tmp_path


def merge_fits_list(file_list: Sequence[str], output_path: str, batch_size: int = MERGE_BATCH) -> None:
    file_list = sorted(file_list)
    if not file_list:
        raise RuntimeError(f"No temporary FITS files found for merge into {output_path}")

    accumulator: Optional[Table] = None
    for i in range(0, len(file_list), batch_size):
        group = file_list[i : i + batch_size]
        tables = [Table.read(path) for path in group]
        merged_group = vstack(tables, metadata_conflicts="silent")
        accumulator = merged_group if accumulator is None else vstack([accumulator, merged_group], metadata_conflicts="silent")
        LOGGER.info("Merged %d / %d chunk files -> %s rows so far", i + len(group), len(file_list), f"{len(accumulator):,}")

    accumulator.write(output_path, overwrite=True)
    LOGGER.info("Wrote merged FITS: %s (%s rows)", output_path, f"{len(accumulator):,}")


def write_cutflow_report(cutflow: Mapping[str, int], output_txt: str, output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(cutflow, fp, indent=2)

    input_n = max(int(cutflow.get("input", 0)), 1)
    lines = []
    lines.append("Pal 5 preprocessing cut-flow\n")
    lines.append(f"INPUT_FITS = {INPUT_FITS}\n")
    lines.append(f"OUTPUT_FITS = {OUTPUT_FITS}\n\n")
    for key, value in cutflow.items():
        frac = value / input_n
        lines.append(f"{key:24s} : {value:12d}   ({frac:8.4%})\n")

    with open(output_txt, "w", encoding="utf-8") as fp:
        fp.writelines(lines)


def write_summary(cutflow: Mapping[str, int], output_json: str) -> None:
    summary = {
        "input_fits": INPUT_FITS,
        "output_fits": OUTPUT_FITS,
        "diagnostics_dir": PLOT_DIR,
        "tmp_dir": TMP_DIR,
        "report_dir": REPORT_DIR,
        "cuts": {key: int(value) for key, value in cutflow.items()},
    }
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def plot_cutflow(cutflow: Mapping[str, int], output_png: str) -> None:
    keys = list(cutflow.keys())
    values = [int(cutflow[k]) for k in keys]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(keys)), values, marker="o")
    plt.xticks(range(len(keys)), keys, rotation=30, ha="right")
    plt.ylabel("Number of objects")
    plt.title("Preprocessing cut-flow")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def make_edges(vmin: float, vmax: float, step: float) -> np.ndarray:
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    return np.arange(lo, hi + step, step, dtype=np.float64)


def accumulate_hist2d(
    data: fits.fitsrec.FITS_rec,
    xcol: str,
    ycol: str,
    xedges: np.ndarray,
    yedges: np.ndarray,
    chunk_size: int = PLOT_CHUNK,
) -> np.ndarray:
    n = len(data)
    hist = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x = np.asarray(data[xcol][start:stop], dtype=np.float64)
        y = np.asarray(data[ycol][start:stop], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            h, _, _ = np.histogram2d(x[mask], y[mask], bins=[xedges, yedges])
            hist += h.astype(np.int64)
    return hist


def imshow_hist2d(
    hist: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: str,
    unit_label: str,
    invert_x: bool = False,
    log_scale: bool = True,
    invert_y: bool = False,
) -> None:
    image = hist.T.astype(float)
    image = np.ma.masked_less_equal(image, 0.0)

    plt.figure(figsize=(8, 6))
    kwargs = dict(
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    if log_scale and np.any(~image.mask):
        vmin = max(float(image.min()), 1.0)
        vmax = float(image.max())
        plt.imshow(image, norm=LogNorm(vmin=vmin, vmax=vmax), **kwargs)
    else:
        plt.imshow(image, **kwargs)
    if invert_x:
        plt.gca().invert_xaxis()
    if invert_y:
        plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(unit_label)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def make_qc_plots(fits_path: str) -> None:
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)

        ra_edges = make_edges(RA_MIN, RA_MAX, DENSITY_BIN_DEG)
        dec_edges = make_edges(DEC_MIN, DEC_MAX, DENSITY_BIN_DEG)
        hist_radec = accumulate_hist2d(data, "RA", "DEC", ra_edges, dec_edges)
        area = DENSITY_BIN_DEG * DENSITY_BIN_DEG
        imshow_hist2d(
            hist_radec / area,
            ra_edges,
            dec_edges,
            title=f"RA-Dec number density (bin = {DENSITY_BIN_DEG:.2f} deg)",
            xlabel="RA [deg]",
            ylabel="Dec [deg]",
            output_png=os.path.join(PLOT_DIR, "density_radec.png"),
            unit_label=r"counts / deg$^2$",
            invert_x=True,
        )

        phi1_edges = make_edges(PHI1_MIN, PHI1_MAX, DENSITY_BIN_DEG)
        phi2_edges = make_edges(PHI2_MIN, PHI2_MAX, DENSITY_BIN_DEG)
        hist_phi = accumulate_hist2d(data, "PHI1", "PHI2", phi1_edges, phi2_edges)
        imshow_hist2d(
            hist_phi / area,
            phi1_edges,
            phi2_edges,
            title=f"Pal 5 frame number density (bin = {DENSITY_BIN_DEG:.2f} deg)",
            xlabel=r"$\phi_1$ [deg]",
            ylabel=r"$\phi_2$ [deg]",
            output_png=os.path.join(PLOT_DIR, "density_phi12.png"),
            unit_label=r"counts / deg$^2$",
        )

        if {"GR0", "G0"}.issubset(names):
            cmd_xedges = np.arange(CMD_X_RANGE[0], CMD_X_RANGE[1] + CMD_DX, CMD_DX)
            cmd_yedges = np.arange(min(CMD_Y_RANGE), max(CMD_Y_RANGE) + CMD_DY, CMD_DY)
            hist_cmd = accumulate_hist2d(data, "GR0", "G0", cmd_xedges, cmd_yedges)
            imshow_hist2d(
                hist_cmd,
                cmd_xedges,
                cmd_yedges,
                title="Dereddened CMD",
                xlabel=r"$(g-r)_0$",
                ylabel=r"$g_0$",
                output_png=os.path.join(PLOT_DIR, "cmd_gr_g0.png"),
                unit_label="counts / bin",
                invert_y=True,
            )

        if {"GR0", "GZ0"}.issubset(names):
            cc_xedges = np.arange(CC_X_RANGE[0], CC_X_RANGE[1] + CC_DX, CC_DX)
            cc_yedges = np.arange(CC_Y_RANGE[0], CC_Y_RANGE[1] + CC_DY, CC_DY)
            hist_cc = accumulate_hist2d(data, "GR0", "GZ0", cc_xedges, cc_yedges)
            plt.figure(figsize=(7, 6))
            image = np.ma.masked_less_equal(hist_cc.T.astype(float), 0.0)
            plt.imshow(
                image,
                origin="lower",
                aspect="auto",
                extent=[cc_xedges[0], cc_xedges[-1], cc_yedges[0], cc_yedges[-1]],
                norm=LogNorm(vmin=max(float(image.min()), 1.0), vmax=float(image.max())),
            )
            xline = np.linspace(CC_X_RANGE[0], CC_X_RANGE[1], 300)
            plt.plot(xline, 1.7 * xline - 0.17, linestyle="--", linewidth=1.5)
            plt.xlabel(r"$(g-r)_0$")
            plt.ylabel(r"$(g-z)_0$")
            plt.title("Dereddened color-color diagram")
            cbar = plt.colorbar()
            cbar.set_label("counts / bin")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "colorcolor_gr_gz.png"), dpi=200)
            plt.close()

        if "DEPTH5_G" in names:
            depth = np.asarray(data["DEPTH5_G"], dtype=np.float32)
            depth = depth[np.isfinite(depth)]
            if depth.size > 0:
                plt.figure(figsize=(7, 5))
                plt.hist(depth, bins=100)
                plt.axvline(DEPTH5_G_MIN_FOR_FAINT, linestyle="--", linewidth=1.5)
                plt.xlabel(r"local $g$ 5$\sigma$ depth [AB mag]")
                plt.ylabel("Number of objects")
                plt.title("Depth distribution of the retained sample")
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, "gdepth5sig_hist.png"), dpi=200)
                plt.close()


def main() -> None:
    ensure_directories()
    for path in glob.glob(os.path.join(TMP_DIR, "tmp_pal5_preproc_chunk*.fits")):
        os.remove(path)

    with fits.open(INPUT_FITS, memmap=True) as hdul:
        data = hdul[1].data
        n_total = len(data)
        LOGGER.info("Input FITS: %s", INPUT_FITS)
        LOGGER.info("Total rows in input: %s", f"{n_total:,}")

        name_map = build_name_map(data.names)
        columns = {
            "RA": resolve_required(name_map, "RA"),
            "DEC": resolve_required(name_map, "DEC"),
            "TYPE": resolve_required(name_map, "TYPE"),
            "MAG_G": resolve_required(name_map, "MAG_G"),
            "MAG_R": resolve_required(name_map, "MAG_R"),
            "MAG_Z": resolve_required(name_map, "MAG_Z"),
            "PSFDEPTH_G": resolve_required(name_map, "PSFDEPTH_G"),
            "ALLMASK_G": resolve_optional(name_map, "ALLMASK_G"),
            "ALLMASK_R": resolve_optional(name_map, "ALLMASK_R"),
            "BRIGHTSTARINBLOB": resolve_optional(name_map, "BRIGHTSTARINBLOB"),
            "BRIGHTBLOB": resolve_optional(name_map, "BRIGHTBLOB"),
            "MASKBITS": resolve_optional(name_map, "MASKBITS"),
            "BRICK_PRIMARY": resolve_optional(name_map, "BRICK_PRIMARY"),
            "MW_TRANSMISSION_G": resolve_optional(name_map, "MW_TRANSMISSION_G"),
            "MW_TRANSMISSION_R": resolve_optional(name_map, "MW_TRANSMISSION_R"),
            "MW_TRANSMISSION_Z": resolve_optional(name_map, "MW_TRANSMISSION_Z"),
        }

        total_cutflow = make_cutflow_dict()
        written_rows = 0
        temp_files: List[str] = []

        chunk_id = 0
        for start in range(0, n_total, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, n_total)
            sub = data[start:stop]
            n_written, cutflow, tmp_path = process_chunk(sub, chunk_id, columns)
            accumulate_cutflow(total_cutflow, cutflow)
            written_rows += n_written
            if tmp_path is not None:
                temp_files.append(tmp_path)
            LOGGER.info(
                "Chunk %04d rows %s-%s -> kept %s | cumulative kept %s",
                chunk_id,
                f"{start:,}",
                f"{stop:,}",
                f"{n_written:,}",
                f"{written_rows:,}",
            )
            chunk_id += 1

    if not temp_files:
        raise RuntimeError("No rows survived preprocessing. Check the cuts and input columns.")

    merge_fits_list(temp_files, OUTPUT_FITS, batch_size=MERGE_BATCH)
    write_cutflow_report(
        total_cutflow,
        os.path.join(REPORT_DIR, "preprocess_cutflow.txt"),
        os.path.join(REPORT_DIR, "preprocess_cutflow.json"),
    )
    write_summary(total_cutflow, SUMMARY_JSON)
    plot_cutflow(total_cutflow, os.path.join(PLOT_DIR, "qc_cutflow.png"))
    make_qc_plots(OUTPUT_FITS)
    LOGGER.info("Done. Final output FITS: %s", OUTPUT_FITS)


if __name__ == "__main__":
    main()
