from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table


def fit_track(input_path: Path, output_path: Path, min_pmem: float, phi1_step: float, window: float) -> None:
    table = Table.read(input_path)
    phi1 = np.asarray(table["PHI1"], dtype=float)
    phi2 = np.asarray(table["PHI2"], dtype=float)
    p_mem = np.asarray(table["P_MEM"], dtype=float)
    mask = (
        np.isfinite(phi1)
        & np.isfinite(phi2)
        & np.isfinite(p_mem)
        & (phi1 > -21.0)
        & (phi1 < 13.0)
        & (phi2 > -3.0)
        & (phi2 < 6.0)
    )
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    p_mem = p_mem[mask]
    if phi1.size == 0:
        raise RuntimeError("No finite stars remain after the basic Pal 5 window cut")

    keep = p_mem >= min_pmem
    if np.sum(keep) < 50:
        keep = p_mem >= float(np.nanquantile(p_mem, 0.90))
    if np.sum(keep) < 50:
        keep = p_mem >= float(np.nanquantile(p_mem, 0.75))

    phi1 = phi1[keep]
    phi2 = phi2[keep]
    p_mem = p_mem[keep]
    if phi1.size == 0:
        raise RuntimeError("No stars survive the adaptive P_MEM threshold")

    phi1_bins = np.arange(np.min(phi1), np.max(phi1) + phi1_step, phi1_step)
    rows = []
    for center in phi1_bins:
        sel = np.abs(phi1 - center) < 0.5 * window
        if np.sum(sel) < 30:
            continue
        phi2_sel = phi2[sel]
        w_sel = p_mem[sel]
        if not np.any(np.isfinite(w_sel)) or np.nansum(w_sel) <= 0:
            continue
        mu = np.average(phi2_sel, weights=w_sel)
        variance = np.average((phi2_sel - mu) ** 2, weights=w_sel)
        sigma = np.sqrt(max(variance, 1e-4))
        amp = np.sum(w_sel) / max(sigma * np.sqrt(2.0 * np.pi), 1e-6)
        if not np.isfinite(mu) or not np.isfinite(sigma):
            continue
        rows.append((center, mu, sigma, amp, amp * sigma * np.sqrt(2.0 * np.pi)))

    if rows:
        out = Table(rows=rows, names=["phi1_bin", "mu", "sigma", "amp", "n_stream"])
    else:
        out = Table(
            names=["phi1_bin", "mu", "sigma", "amp", "n_stream"],
            dtype=[float, float, float, float, float],
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(output_path, overwrite=True)
    print(f"wrote {output_path} with {len(out)} bins")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the Pal 5 stream track from a probability-enriched catalog.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-pmem", type=float, default=0.5)
    parser.add_argument("--phi1-step", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=1.5)
    args = parser.parse_args()
    fit_track(Path(args.input), Path(args.output), args.min_pmem, args.phi1_step, args.window)


if __name__ == "__main__":
    main()
