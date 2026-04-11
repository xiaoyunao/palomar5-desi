from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def plot_phi12_map(table: Table, outdir: Path) -> None:
    phi1 = np.asarray(table["PHI1"], dtype=float)
    phi2 = np.asarray(table["PHI2"], dtype=float)
    p_mem = np.asarray(table["P_MEM"], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    sc = ax.scatter(phi1, phi2, c=p_mem, s=0.5, cmap="magma", vmin=0.0, vmax=1.0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="P_MEM")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Pal 5 Overall Stream Distribution in Stream Coordinates")
    ax.set_xlim(-25, 20)
    ax.set_ylim(-5, 10)
    fig.tight_layout()
    fig.savefig(outdir / "pal5_phi12_overall.png")
    plt.close(fig)


def plot_radec_map(table: Table, outdir: Path) -> None:
    ra = np.asarray(table["RA"], dtype=float)
    dec = np.asarray(table["DEC"], dtype=float)
    p_mem = np.asarray(table["P_MEM"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    sc = ax.scatter(ra, dec, c=p_mem, s=0.5, cmap="viridis", vmin=0.0, vmax=1.0, rasterized=True)
    plt.colorbar(sc, ax=ax, label="P_MEM")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Pal 5 Candidate Distribution on the Sky")
    fig.tight_layout()
    fig.savefig(outdir / "pal5_radec_overall.png")
    plt.close(fig)


def plot_cmd_panels(table: Table, outdir: Path) -> None:
    g0 = np.asarray(table["G0"], dtype=float)
    r0 = np.asarray(table["R0"], dtype=float)
    z0 = np.asarray(table["Z0"], dtype=float)
    p_mem = np.asarray(table["P_MEM"], dtype=float)

    mask = np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0) & np.isfinite(p_mem)
    g0 = g0[mask]
    r0 = r0[mask]
    z0 = z0[mask]
    p_mem = p_mem[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    sc1 = axes[0].scatter(g0 - r0, g0, c=p_mem, s=1.0, cmap="plasma", vmin=0.0, vmax=1.0, rasterized=True)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("G0 - R0")
    axes[0].set_ylabel("G0")
    axes[0].set_title("CMD: (G-R, G)")

    sc2 = axes[1].scatter(r0 - z0, r0, c=p_mem, s=1.0, cmap="plasma", vmin=0.0, vmax=1.0, rasterized=True)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("R0 - Z0")
    axes[1].set_ylabel("R0")
    axes[1].set_title("CMD: (R-Z, R)")

    fig.colorbar(sc2, ax=axes, label="P_MEM")
    fig.tight_layout()
    fig.savefig(outdir / "pal5_cmd_membership.png")
    plt.close(fig)


def plot_probability_hist(table: Table, outdir: Path) -> None:
    p_iso = _finite(np.asarray(table["P_ISO"], dtype=float))
    p_mem = _finite(np.asarray(table["P_MEM"], dtype=float))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.hist(p_iso, bins=100, histtype="step", linewidth=1.5, label="P_ISO", color="tab:orange", log=True)
    ax.hist(p_mem, bins=100, histtype="step", linewidth=1.5, label="P_MEM", color="tab:blue", log=True)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Membership Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pal5_probability_hist.png")
    plt.close(fig)


def plot_track_overlay(table: Table, track: Table, outdir: Path) -> None:
    phi1 = np.asarray(table["PHI1"], dtype=float)
    phi2 = np.asarray(table["PHI2"], dtype=float)
    p_mem = np.asarray(table["P_MEM"], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.scatter(phi1, phi2, c=p_mem, s=0.4, cmap="Greys", vmin=0.0, vmax=1.0, rasterized=True)
    if len(track):
        phi1_bin = np.asarray(track["phi1_bin"], dtype=float)
        mu = np.asarray(track["mu"], dtype=float)
        sigma = np.asarray(track["sigma"], dtype=float)
        ax.plot(phi1_bin, mu, color="tab:red", lw=2.0, label="Extracted track")
        ax.fill_between(phi1_bin, mu - sigma, mu + sigma, color="tab:red", alpha=0.25, label=r"$\pm 1\sigma$")
        ax.legend()
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Stream Track Overlay")
    ax.set_xlim(-25, 20)
    ax.set_ylim(-5, 10)
    fig.tight_layout()
    fig.savefig(outdir / "pal5_track_overlay.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pal 5 detection summary figures.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    table = Table.read(args.catalog)
    track = Table.read(args.track)

    plot_phi12_map(table, outdir)
    plot_radec_map(table, outdir)
    plot_cmd_panels(table, outdir)
    plot_probability_hist(table, outdir)
    plot_track_overlay(table, track, outdir)
    print(f"wrote plots to {outdir}")


if __name__ == "__main__":
    main()
