#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-make qc_step4c_segment_cmds in the original excess-Hess style, "
            "but using the full z-locus parent CMD (no strict 20<g<23 display mask)."
        )
    )
    p.add_argument("--preproc", default="final_g25_preproc.fits")
    p.add_argument("--step2-summary", default="step2_outputs/pal5_step2_summary.json")
    p.add_argument("--iso", default="pal5.dat")
    p.add_argument("--dm-track-csv", default="step4c_outputs/pal5_step4c_dm_track.csv")
    p.add_argument("--mu-prior-file", default="step3b_outputs_control/pal5_step3b_mu_prior.txt")
    p.add_argument("--output", default="step4c_outputs/qc_step4c_segment_cmds_stylefixed_fullhess.png")
    p.add_argument("--display-gmin", type=float, default=16.0)
    p.add_argument("--display-gmax", type=float, default=24.0)
    p.add_argument("--display-gr-min", type=float, default=-0.3)
    p.add_argument("--display-gr-max", type=float, default=1.0)
    p.add_argument("--segment-half", type=float, default=1.5)
    p.add_argument("--on-halfwidth", type=float, default=0.4)
    p.add_argument("--off-inner", type=float, default=0.8)
    p.add_argument("--off-outer", type=float, default=1.6)
    p.add_argument("--x-bins", type=int, default=66)
    p.add_argument("--y-bins", type=int, default=81)
    p.add_argument("--cmd-alignment-pivot", type=float, default=20.5)
    return p.parse_args()


def load_step2_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_iso(path: str):
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Zini"):
                header = line.lstrip("#").strip().split()
                break
    if header is None:
        raise RuntimeError("Could not find '# Zini ...' header in isochrone file")
    name_to_idx = {n: i for i, n in enumerate(header)}
    arr = np.loadtxt(path, comments="#")
    label = arr[:, name_to_idx["label"]].astype(int)
    keep = np.isin(label, [1, 2, 3])
    arr = arr[keep]
    return {
        "g_abs": np.asarray(arr[:, name_to_idx["g_f0"]], dtype=float),
        "r_abs": np.asarray(arr[:, name_to_idx["r_f0"]], dtype=float),
    }


def build_cmd_interpolator(
    iso: dict, dm: float, cfg: dict, pivot_mag: float
) -> tuple[np.ndarray, np.ndarray]:
    align = cfg["alignment"]
    g_app = iso["g_abs"] + dm
    c = (iso["g_abs"] - iso["r_abs"]) + float(align["dc0"]) + float(align["dc1"]) * (
        g_app - pivot_mag
    )
    order = np.argsort(g_app)
    g_sorted = g_app[order]
    c_sorted = c[order]
    uniq = np.concatenate([[True], np.diff(g_sorted) > 1e-6])
    return g_sorted[uniq], c_sorted[uniq]


def load_dm_func(path: str):
    tab = Table.read(path)
    if "phi1" not in tab.colnames or "dm" not in tab.colnames:
        raise KeyError(f"{path} must contain columns 'phi1' and 'dm'")
    return interp1d(
        np.asarray(tab["phi1"], dtype=float),
        np.asarray(tab["dm"], dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )


def load_mu_func(path: str):
    arr = np.loadtxt(path)
    return interp1d(
        np.asarray(arr[:, 0], dtype=float),
        np.asarray(arr[:, 1], dtype=float),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )


def select_display_parent(pre: Table, zloc_tol: float) -> dict:
    g0 = np.asarray(pre["G0"], dtype=float)
    r0 = np.asarray(pre["R0"], dtype=float)
    z0 = np.asarray(pre["Z0"], dtype=float)
    phi1 = np.asarray(pre["PHI1"], dtype=float)
    phi2 = np.asarray(pre["PHI2"], dtype=float)
    m = np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0) & np.isfinite(phi1) & np.isfinite(phi2)
    gr0 = g0 - r0
    gz0 = g0 - z0
    zloc = 1.7 * gr0 - 0.17
    m &= np.abs(gz0 - zloc) <= zloc_tol
    return {
        "phi1": phi1[m],
        "phi2": phi2[m],
        "g0": g0[m],
        "gr0": gr0[m],
    }


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cfg = load_step2_cfg(args.step2_summary)
    iso = load_iso(args.iso)
    dm_func = load_dm_func(args.dm_track_csv)
    mu_func = load_mu_func(args.mu_prior_file)

    pre = Table.read(args.preproc)
    parent = select_display_parent(pre, float(cfg["strict_config"]["ZLOCUS_TOL"]))

    centers = [-19.0, -13.0, -7.0, -3.0, 3.0, 9.0]
    xedges = np.linspace(args.display_gr_min, args.display_gr_max, args.x_bins)
    yedges = np.linspace(args.display_gmin, args.display_gmax, args.y_bins)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=True)
    axes = axes.ravel()

    for ax, c in zip(axes, centers):
        mu0 = float(mu_func(c))
        seg = np.abs(parent["phi1"] - c) <= args.segment_half
        on = seg & (np.abs(parent["phi2"] - mu0) <= args.on_halfwidth)
        off = seg & (np.abs(parent["phi2"] - mu0) >= args.off_inner) & (
            np.abs(parent["phi2"] - mu0) < args.off_outer
        )

        x_on = parent["gr0"][on]
        y_on = parent["g0"][on]
        x_off = parent["gr0"][off]
        y_off = parent["g0"][off]

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
        m_iso = (g_iso >= args.display_gmin) & (g_iso <= args.display_gmax)
        ax.plot(c_iso[m_iso], g_iso[m_iso], color="orange", lw=2.0)

        ax.set_title(rf"$\phi_1\approx {c:+.1f}^\circ$, DM={dm_here:.3f}")
        ax.set_xlabel(r"$(g-r)_0$")
        ax.set_ylabel(r"$g_0$")
        ax.set_xlim(args.display_gr_min, args.display_gr_max)
        ax.set_ylim(args.display_gmax, args.display_gmin)

    fig.suptitle("step4c: local on-stream minus off-stream CMDs with the RRL-informed DM(track)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"[write] {args.output}")
    print("Re-made the segment CMDs in the original excess-Hess style, but with the full display CMD range included.")


if __name__ == "__main__":
    main()
