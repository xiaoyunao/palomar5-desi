from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table


def load_columns(catalog_path: str):
    with fits.open(catalog_path, memmap=True) as hdul:
        data = hdul[1].data
        ra = np.asarray(data["RA"], dtype=np.float32)
        dec = np.asarray(data["DEC"], dtype=np.float32)
        phi1 = np.asarray(data["PHI1"], dtype=np.float32)
        phi2 = np.asarray(data["PHI2"], dtype=np.float32)
        p_mem = np.asarray(data["P_MEM"], dtype=np.float32)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(p_mem)
    return ra[mask], dec[mask], phi1[mask], phi2[mask], p_mem[mask]


def plot_phi12_logsum(phi1, phi2, p_mem, outpath: Path):
    hist, xedges, yedges = np.histogram2d(phi1, phi2, bins=[450, 180], range=[[-25, 20], [-5, 10]], weights=p_mem)
    image = np.full_like(hist.T, np.nan, dtype=float)
    pos = hist.T > 0
    image[pos] = np.log10(hist.T[pos])

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    im = ax.imshow(
        image,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\sum P_{\rm MEM}$")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Pal 5 Weighted Density in Stream Coordinates")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_phi12_highprob(phi1, phi2, p_mem, track_path: str, outpath: Path, threshold: float):
    mask = p_mem >= threshold
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    p_mem = p_mem[mask]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    sc = ax.scatter(phi1, phi2, c=p_mem, s=1.5, cmap="magma", vmin=threshold, vmax=1.0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="P_MEM")
    try:
        track = Table.read(track_path)
        if len(track):
            x = np.asarray(track["phi1_bin"], dtype=float)
            y = np.asarray(track["mu"], dtype=float)
            s = np.asarray(track["sigma"], dtype=float)
            ax.plot(x, y, color="cyan", lw=2.0)
            ax.fill_between(x, y - s, y + s, color="cyan", alpha=0.18)
    except Exception:
        pass
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_xlim(-25, 20)
    ax.set_ylim(-5, 10)
    ax.set_title(f"High-Probability Pal 5 Candidates (P_MEM >= {threshold:.1f})")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_radec_logsum(ra, dec, p_mem, outpath: Path):
    hist, xedges, yedges = np.histogram2d(ra, dec, bins=[420, 320], weights=p_mem)
    image = np.full_like(hist.T, np.nan, dtype=float)
    pos = hist.T > 0
    image[pos] = np.log10(hist.T[pos])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    im = ax.imshow(
        image,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\sum P_{\rm MEM}$")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Pal 5 Weighted Sky Density")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fast, interpretable quicklook plots for Pal 5 membership results.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ra, dec, phi1, phi2, p_mem = load_columns(args.catalog)

    plot_phi12_logsum(phi1, phi2, p_mem, outdir / "pal5_quick_phi12_logsum.png")
    plot_phi12_highprob(phi1, phi2, p_mem, args.track, outdir / "pal5_quick_phi12_highprob.png", args.threshold)
    plot_radec_logsum(ra, dec, p_mem, outdir / "pal5_quick_radec_logsum.png")
    print(f"wrote quicklook plots to {outdir}")


if __name__ == "__main__":
    main()
