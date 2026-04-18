#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


@dataclass
class Step2Config:
    gmin: float
    gmax: float
    zloc_tol: float
    cmd_w0: float
    cmd_w_slope: float
    cmd_w_min: float
    cmd_w_max: float
    dc0: float
    dc1: float


@dataclass
class IsochroneData:
    g_abs: np.ndarray
    r_abs: np.ndarray
    z_abs: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plotting-only patch for the step4c RR-Lyrae-prior baseline. "
            "This script does not change the science selection or morphology fit; "
            "it only regenerates diagnostic plots with a full-CMD display and log-density maps."
        )
    )
    p.add_argument("--preproc", default="final_g25_preproc.fits")
    p.add_argument("--step2-summary", default="step2_outputs/pal5_step2_summary.json")
    p.add_argument("--iso", default="pal5.dat")
    p.add_argument("--dm-track-csv", default="step4c_outputs/pal5_step4c_dm_track.csv")
    p.add_argument("--selected-members", default="step4c_outputs/pal5_step4c_rrlprior_members.fits")
    p.add_argument("--mu-prior-file", default="step3b_outputs_control/pal5_step3b_mu_prior.txt")
    p.add_argument("--output-dir", default="step4c_outputs")
    p.add_argument("--segment-window-half", type=float, default=1.5)
    p.add_argument("--on-halfwidth", type=float, default=0.4)
    p.add_argument("--off-inner", type=float, default=0.8)
    p.add_argument("--off-outer", type=float, default=1.6)
    p.add_argument("--centers", default="-19,-13,-7,-3,3,9")
    p.add_argument("--full-cmd-gmin", type=float, default=16.0)
    p.add_argument("--full-cmd-gmax", type=float, default=24.0)
    p.add_argument("--full-cmd-cmin", type=float, default=-0.35)
    p.add_argument("--full-cmd-cmax", type=float, default=1.05)
    p.add_argument("--full-cmd-bin-color", type=float, default=0.02)
    p.add_argument("--full-cmd-bin-mag", type=float, default=0.10)
    p.add_argument("--score-gmin", type=float, default=19.8)
    p.add_argument("--score-gmax", type=float, default=21.7)
    p.add_argument("--anchor-blue-cut", type=float, default=0.15)
    p.add_argument("--anchor-blue-gmin", type=float, default=21.5)
    p.add_argument("--cmd-alignment-pivot", type=float, default=20.5)
    p.add_argument("--phi12-bin", type=float, default=0.10)
    p.add_argument("--radec-bin", type=float, default=0.10)
    p.add_argument("--local-phi2-lim", type=float, default=2.5)
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
        dc0=float(align["dc0"]),
        dc1=float(align["dc1"]),
    )


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


def load_interp(path: str | Path, xcol: int = 0, ycol: int = 1) -> interp1d:
    arr = np.loadtxt(path, delimiter="," if str(path).endswith(".csv") else None)
    if arr.ndim == 1 or arr.shape[1] < max(xcol, ycol) + 1:
        tab = Table.read(path)
        x = np.asarray(tab[tab.colnames[xcol]], dtype=float)
        y = np.asarray(tab[tab.colnames[ycol]], dtype=float)
    else:
        x = np.asarray(arr[:, xcol], dtype=float)
        y = np.asarray(arr[:, ycol], dtype=float)
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")


def load_dm_interp(path: str | Path) -> interp1d:
    tab = Table.read(path)
    x = np.asarray(tab[tab.colnames[0]], dtype=float)
    y = np.asarray(tab[tab.colnames[1]], dtype=float)
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")


def cmd_halfwidth(g0: np.ndarray, cfg: Step2Config) -> np.ndarray:
    w = cfg.cmd_w0 + cfg.cmd_w_slope * (g0 - cfg.gmin)
    return np.clip(w, cfg.cmd_w_min, cfg.cmd_w_max)


def build_cmd_interpolator(
    iso: IsochroneData, dm: float, cfg: Step2Config, pivot_mag: float
) -> tuple[np.ndarray, np.ndarray]:
    g_app = iso.g_abs + dm
    c = (iso.g_abs - iso.r_abs) + cfg.dc0 + cfg.dc1 * (g_app - pivot_mag)
    order = np.argsort(g_app)
    g_sorted = g_app[order]
    c_sorted = c[order]
    uniq_mask = np.concatenate([[True], np.diff(g_sorted) > 1e-6])
    return g_sorted[uniq_mask], c_sorted[uniq_mask]


def select_zparent(t: Table, cfg: Step2Config) -> np.ndarray:
    g0 = np.asarray(t["G0"], dtype=float)
    r0 = np.asarray(t["R0"], dtype=float)
    z0 = np.asarray(t["Z0"], dtype=float)
    gr0 = g0 - r0
    gz0 = g0 - z0
    zloc = 1.7 * gr0 - 0.17

    m = np.isfinite(g0) & np.isfinite(r0) & np.isfinite(z0)
    m &= (g0 > cfg.gmin) & (g0 < cfg.gmax)
    m &= np.abs(gz0 - zloc) <= cfg.zloc_tol
    return m


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


def show_log_density(
    ax,
    D: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
):
    positive = D[D > 0]
    vmin = max(1.0, float(np.nanmin(positive)) if positive.size else 1.0)
    vmax = float(np.nanmax(D)) if np.nanmax(D) > 0 else 10.0
    im = ax.imshow(
        D,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def plot_segment_cmds_fullcmd(
    pre: Table,
    zparent_mask: np.ndarray,
    dm_func,
    iso: IsochroneData,
    cfg: Step2Config,
    mu_interp,
    out_png: str | Path,
    args: argparse.Namespace,
) -> None:
    centers = [float(x) for x in args.centers.split(",")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharey=True)
    axes = axes.ravel()

    phi1 = np.asarray(pre["PHI1"], dtype=float)
    phi2 = np.asarray(pre["PHI2"], dtype=float)
    g0 = np.asarray(pre["G0"], dtype=float)
    r0 = np.asarray(pre["R0"], dtype=float)
    gr0 = g0 - r0
    finite = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(g0) & np.isfinite(r0)
    finite &= (g0 >= args.full_cmd_gmin) & (g0 <= args.full_cmd_gmax)
    finite &= (gr0 >= args.full_cmd_cmin - 0.2) & (gr0 <= args.full_cmd_cmax + 0.2)

    xedges = np.arange(
        args.full_cmd_cmin,
        args.full_cmd_cmax + args.full_cmd_bin_color,
        args.full_cmd_bin_color,
    )
    yedges = np.arange(
        args.full_cmd_gmin,
        args.full_cmd_gmax + args.full_cmd_bin_mag,
        args.full_cmd_bin_mag,
    )

    for ax, c in zip(axes, centers):
        mu0 = float(mu_interp(c))
        seg = np.abs(phi1 - c) <= args.segment_window_half
        on = finite & seg & (np.abs(phi2 - mu0) <= args.on_halfwidth)
        off = finite & seg & (np.abs(phi2 - mu0) >= args.off_inner) & (
            np.abs(phi2 - mu0) < args.off_outer
        )

        Hfull_on, _, _ = np.histogram2d(gr0[on], g0[on], bins=[xedges, yedges])
        Hfull_on = gaussian_filter(Hfull_on, sigma=0.7)
        ax.imshow(
            np.log10(Hfull_on.T + 1.0),
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="Greys",
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(np.log10(Hfull_on + 1.0))))
            if np.any(Hfull_on > 0)
            else 1.0,
            alpha=0.95,
        )

        on_z = on & zparent_mask
        off_z = off & zparent_mask
        Hon, _, _ = np.histogram2d(gr0[on_z], g0[on_z], bins=[xedges, yedges])
        Hoff, _, _ = np.histogram2d(gr0[off_z], g0[off_z], bins=[xedges, yedges])
        scale = (
            (2.0 * args.on_halfwidth) / (2.0 * (args.off_outer - args.off_inner))
            if args.off_outer > args.off_inner
            else 0.5
        )
        diff = Hon - scale * Hoff
        diff = gaussian_filter(np.clip(diff, 0.0, None), sigma=1.0)
        if np.any(diff > 0):
            levels = np.quantile(diff[diff > 0], [0.60, 0.80, 0.92])
            levels = np.unique(levels[levels > 0])
            if levels.size:
                ax.contour(
                    0.5 * (xedges[:-1] + xedges[1:]),
                    0.5 * (yedges[:-1] + yedges[1:]),
                    diff.T,
                    levels=levels,
                    colors=["#17becf", "#1f77b4", "#d62728"][: len(levels)],
                    linewidths=1.2,
                )

        dm_here = float(dm_func(c))
        g_iso, c_iso = build_cmd_interpolator(iso, dm_here, cfg, args.cmd_alignment_pivot)
        m_all = (g_iso >= args.full_cmd_gmin) & (g_iso <= args.full_cmd_gmax)
        ax.plot(c_iso[m_all], g_iso[m_all], color="orange", lw=2.0, zorder=5)

        m_score = (g_iso >= args.score_gmin) & (g_iso <= args.score_gmax)
        if np.any(m_score):
            g_band = g_iso[m_score]
            c_band = c_iso[m_score]
            w_band = cmd_halfwidth(g_band, cfg)
            ax.fill_betweenx(
                g_band,
                c_band - w_band,
                c_band + w_band,
                color="orange",
                alpha=0.18,
                lw=0.0,
                zorder=4,
            )

        ax.axvspan(
            args.full_cmd_cmin,
            args.anchor_blue_cut,
            ymin=(args.anchor_blue_gmin - args.full_cmd_gmin)
            / (args.full_cmd_gmax - args.full_cmd_gmin),
            ymax=1.0,
            color="royalblue",
            alpha=0.10,
            zorder=1,
        )
        ax.axhspan(args.score_gmin, args.score_gmax, color="gold", alpha=0.04, zorder=0)

        ax.set_title(rf"$\phi_1\approx {c:+.1f}^\circ$, DM={dm_here:.3f}")
        ax.set_xlabel(r"$(g-r)_0$")
        ax.set_ylabel(r"$g_0$")
        ax.set_xlim(args.full_cmd_cmin, args.full_cmd_cmax)
        ax.set_ylim(args.full_cmd_gmax, args.full_cmd_gmin)
        ax.text(
            0.02,
            0.03,
            "gray: full on-stream CMD\ncyan/blue/red: strict z-parent excess\norange band: score region\nblue tint: downweighted blue residual",
            transform=ax.transAxes,
            fontsize=7.5,
            va="bottom",
            ha="left",
            color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.72, lw=0.3),
        )

    fig.suptitle(
        "step4c plotting patch: full segment CMDs from the preprocessed parent catalog\n"
        "(all 16<g0<24 sources shown; the MSTO score region is only highlighted, not used to mask the display)",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_selected_density_log(
    selected: Table,
    mu_interp,
    out_phi12: str | Path,
    out_radec: str | Path,
    out_local: str | Path,
    args: argparse.Namespace,
) -> None:
    D, xedges, yedges = make_density_map(
        selected,
        "PHI1",
        "PHI2",
        (-25, 20),
        (-5, 10),
        args.phi12_bin,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    im = show_log_density(
        ax,
        D,
        xedges,
        yedges,
        "step4c selected sample: Pal 5 frame number density (log scale)",
        r"$\phi_1$ [deg]",
        r"$\phi_2$ [deg]",
    )
    fig.colorbar(im, ax=ax, label="counts / deg$^2$")
    fig.tight_layout()
    fig.savefig(out_phi12, dpi=220)
    plt.close(fig)

    D, xedges, yedges = make_density_map(
        selected,
        "RA",
        "DEC",
        (210, 256),
        (-20, 18),
        args.radec_bin,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    im = show_log_density(
        ax,
        D,
        xedges,
        yedges,
        "step4c selected sample: RA-Dec number density (log scale)",
        "RA [deg]",
        "Dec [deg]",
    )
    fig.colorbar(im, ax=ax, label="counts / deg$^2$")
    fig.tight_layout()
    fig.savefig(out_radec, dpi=220)
    plt.close(fig)

    D, xedges, yedges = make_density_map(
        selected,
        "PHI1",
        "PHI2",
        (-20, 10),
        (-args.local_phi2_lim, args.local_phi2_lim),
        args.phi12_bin,
    )
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = show_log_density(
        ax,
        D,
        xedges,
        yedges,
        "step4c selected sample: local Pal 5 map (log scale)",
        r"$\phi_1$ [deg]",
        r"$\phi_2$ [deg]",
    )
    xs = np.linspace(-20, 10, 300)
    ax.plot(xs, mu_interp(xs), color="white", lw=1.6, alpha=0.9, label="step3b/control prior track")
    ax.legend(framealpha=0.9, loc="upper left")
    fig.colorbar(im, ax=ax, label="counts / deg$^2$")
    fig.tight_layout()
    fig.savefig(out_local, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    cfg = load_step2_config(args.step2_summary)
    iso = load_isochrone(args.iso)
    dm_func = load_dm_interp(args.dm_track_csv)
    mu_interp = load_interp(args.mu_prior_file)

    pre = Table.read(args.preproc)
    selected = Table.read(args.selected_members)
    zparent_mask = select_zparent(pre, cfg)

    plot_segment_cmds_fullcmd(
        pre,
        zparent_mask,
        dm_func,
        iso,
        cfg,
        mu_interp,
        Path(args.output_dir) / "qc_step4c_segment_cmds_fullcmd.png",
        args,
    )
    plot_selected_density_log(
        selected,
        mu_interp,
        Path(args.output_dir) / "qc_step4c_selected_density_phi12_log.png",
        Path(args.output_dir) / "qc_step4c_selected_density_radec_log.png",
        Path(args.output_dir) / "qc_step4c_selected_density_phi12_local_log.png",
        args,
    )

    summary = {
        "preproc": str(args.preproc),
        "selected_members": str(args.selected_members),
        "dm_track_csv": str(args.dm_track_csv),
        "mu_prior_file": str(args.mu_prior_file),
        "n_preproc": int(len(pre)),
        "n_selected": int(len(selected)),
        "n_zparent": int(np.sum(zparent_mask)),
        "outputs": {
            "segment_cmds_fullcmd": str(Path(args.output_dir) / "qc_step4c_segment_cmds_fullcmd.png"),
            "selected_density_phi12_log": str(Path(args.output_dir) / "qc_step4c_selected_density_phi12_log.png"),
            "selected_density_radec_log": str(Path(args.output_dir) / "qc_step4c_selected_density_radec_log.png"),
            "selected_density_phi12_local_log": str(Path(args.output_dir) / "qc_step4c_selected_density_phi12_local_log.png"),
        },
        "notes": [
            "This is a plotting-only patch. No member selection or DM track is changed.",
            "The full segment CMD panels now show all 16<g0<24 stars from the preprocessed parent catalog within each segment.",
            "The MSTO score region is overplotted as an orange band instead of being used to mask the display.",
            "Log-scale density maps are regenerated for the step4c selected sample.",
        ],
    }
    with open(Path(args.output_dir) / "pal5_step4c_plotpatch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
