from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table


LOG2PI = math.log(2.0 * math.pi)


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values)
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def gaussian_filter2d(array: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(array, dtype=np.float64)
    kernel = _gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=0, arr=np.asarray(array, dtype=np.float64))
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=tmp)


@dataclass(frozen=True)
class IsochroneData:
    c1: np.ndarray
    m1: np.ndarray
    c2: np.ndarray
    m2: np.ndarray


@dataclass(frozen=True)
class SegmentModel:
    c0: np.ndarray
    m0: np.ndarray
    dc: np.ndarray
    dm: np.ndarray
    logw: np.ndarray


def read_parsec_isochrone(path: str | Path) -> IsochroneData:
    path = Path(path)
    header = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Zini"):
                header = line.lstrip("#").strip()
                break
    if header is None:
        raise RuntimeError(f"Could not find the '# Zini ...' header in {path}")

    columns = header.split()
    name_to_index = {name: idx for idx, name in enumerate(columns)}
    arr = np.loadtxt(path, comments="#")
    labels = arr[:, name_to_index["label"]].astype(int)
    keep = np.isin(labels, [1, 2, 3])
    arr = arr[keep]

    g_mag = arr[:, name_to_index["g_f0"]]
    r_mag = arr[:, name_to_index["r_f0"]]
    z_mag = arr[:, name_to_index["z_f0"]]
    return IsochroneData(
        c1=g_mag - r_mag,
        m1=g_mag,
        c2=r_mag - z_mag,
        m2=r_mag,
    )


def build_segments(c_iso: np.ndarray, m_iso: np.ndarray, jump_cut: float = 0.7) -> SegmentModel:
    c_iso = np.asarray(c_iso, dtype=np.float64)
    m_iso = np.asarray(m_iso, dtype=np.float64)
    ok = np.isfinite(c_iso) & np.isfinite(m_iso)
    c_iso = c_iso[ok]
    m_iso = m_iso[ok]

    dc = np.diff(c_iso)
    dm = np.diff(m_iso)
    ds = np.hypot(dc, dm)
    keep = (ds > 0.0) & np.isfinite(ds) & (ds < jump_cut)

    c0 = c_iso[:-1][keep]
    m0 = m_iso[:-1][keep]
    dc = (c_iso[1:] - c_iso[:-1])[keep]
    dm = (m_iso[1:] - m_iso[:-1])[keep]
    seg_len = np.maximum(np.hypot(dc, dm), 1e-12)
    if seg_len.size == 0:
        raise ValueError("Isochrone segment construction produced zero valid segments; check the input file and jump_cut")
    logw = np.log(seg_len)
    logw -= logsumexp(logw)
    return SegmentModel(c0=c0, m0=m0, dc=dc, dm=dm, logw=logw)


def build_cmd_pdf(
    color: np.ndarray,
    mag: np.ndarray,
    c_range: tuple[float, float],
    m_range: tuple[float, float],
    nbin_c: int,
    nbin_m: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, c_edges, m_edges = np.histogram2d(
        color,
        mag,
        bins=[nbin_c, nbin_m],
        range=[list(c_range), list(m_range)],
    )
    hist = gaussian_filter2d(hist, smooth_sigma)
    hist += 1e-12
    hist /= np.sum(hist)
    return hist, c_edges, m_edges


def lookup_pdf2d(
    color: np.ndarray,
    mag: np.ndarray,
    pdf: np.ndarray,
    color_edges: np.ndarray,
    mag_edges: np.ndarray,
) -> np.ndarray:
    ix = np.searchsorted(color_edges, color, side="right") - 1
    iy = np.searchsorted(mag_edges, mag, side="right") - 1
    ix = np.clip(ix, 0, len(color_edges) - 2)
    iy = np.clip(iy, 0, len(mag_edges) - 2)
    return np.maximum(pdf[ix, iy].astype(np.float64, copy=False), 1e-300)


def logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def logistic_from_logratio(log_s: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    return np.exp(log_s - logsumexp2(log_s, log_b))


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_table(table: Table, path: str | Path, overwrite: bool = True) -> None:
    path = ensure_parent_dir(path)
    table.write(path, overwrite=overwrite)


def copy_fits_with_added_columns(
    input_path: str | Path,
    output_path: str | Path,
    new_columns: Sequence[fits.Column],
) -> None:
    input_path = Path(input_path)
    output_path = ensure_parent_dir(output_path)
    with fits.open(input_path, memmap=True) as hdul:
        primary = hdul[0].copy()
        table_hdu = hdul[1]
        merged = fits.BinTableHDU.from_columns(table_hdu.columns + fits.ColDefs(list(new_columns)), header=table_hdu.header, name=table_hdu.name)
        fits.HDUList([primary, merged]).writeto(output_path, overwrite=True)


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build quantile bins from an empty array")
    if n_bins <= 1:
        lo = np.min(values)
        hi = np.max(values)
        if hi <= lo:
            hi = lo + 1e-6
        return np.array([lo, hi], dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = ensure_parent_dir(path)
    np.savez(path, **arrays)
