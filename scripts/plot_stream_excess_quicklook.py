from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def running_median_background(hist2d: np.ndarray, window: int = 31) -> np.ndarray:
    pad = window // 2
    padded = np.pad(hist2d, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(hist2d, dtype=float)
    for j in range(hist2d.shape[1]):
        out[:, j] = np.median(padded[:, j : j + window], axis=1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot stream-detection quicklooks using P_ISO and local background subtraction.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--piso-threshold", type=float, default=0.9)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.catalog, memmap=True) as hdul:
        data = hdul[1].data
        phi1 = np.asarray(data["PHI1"], dtype=np.float32)
        phi2 = np.asarray(data["PHI2"], dtype=np.float32)
        p_iso = np.asarray(data["P_ISO"], dtype=np.float32)

    mask = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(p_iso)
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    p_iso = p_iso[mask]

    # 1. High-P_ISO scatter
    sel = p_iso >= args.piso_threshold
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    sc = ax.scatter(phi1[sel], phi2[sel], c=p_iso[sel], s=1.0, cmap="magma", vmin=args.piso_threshold, vmax=1.0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="P_ISO")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_xlim(-25, 20)
    ax.set_ylim(-5, 10)
    ax.set_title(f"High-Probability Photometric Candidates (P_ISO >= {args.piso_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "pal5_piso_highprob.png")
    plt.close(fig)

    # 2. Weighted P_ISO histogram
    hist, xedges, yedges = np.histogram2d(phi1, phi2, bins=[450, 180], range=[[-25, 20], [-5, 10]], weights=p_iso)
    image = np.full_like(hist.T, np.nan, dtype=float)
    pos = hist.T > 0
    image[pos] = np.log10(hist.T[pos])
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    im = ax.imshow(image, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", cmap="inferno")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\sum P_{\rm ISO}$")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Photometric Stream Weight Map")
    fig.tight_layout()
    fig.savefig(outdir / "pal5_piso_logsum.png")
    plt.close(fig)

    # 3. Local background-subtracted excess map
    counts, xedges, yedges = np.histogram2d(phi1, phi2, bins=[360, 150], range=[[-25, 20], [-5, 10]], weights=p_iso)
    counts_t = counts.T
    bg = running_median_background(counts_t, window=31)
    excess = counts_t - bg
    scale = np.nanpercentile(np.abs(excess), 99.5)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    im = ax.imshow(
        excess,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
    )
    plt.colorbar(im, ax=ax, label="Local excess in weighted counts")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Local-Background-Subtracted Stream Map")
    fig.tight_layout()
    fig.savefig(outdir / "pal5_piso_local_excess.png")
    plt.close(fig)

    print(f"wrote stream-excess quicklooks to {outdir}")


if __name__ == "__main__":
    main()
