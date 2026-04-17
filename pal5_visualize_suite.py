
from __future__ import annotations

"""
Pal 5 plotting suite

Purpose
-------
A standalone visualization program for the Pal 5 mock-stream / global-track project.

This script consolidates the plotting logic that used to be scattered across:
    - simulate.ipynb
    - plot_rv.ipynb
    - pal5_mock_track_fit_refactor.py outputs

What it can make
----------------
1. Stream morphology overlays in Pal 5 coordinates:
   - optional star-catalog hexbin background
   - mock particles
   - observed track
   - model track
   - optional width bands

2. Track diagnostics:
   - observed vs model phi2(phi1)
   - residuals
   - width comparison
   - model node counts
   - a Bonaca-like summary panel figure

3. MCMC diagnostics:
   - corner plot
   - walker chains
   - log-probability map

4. Orbit plots:
   - best-fit orbit in R-Z and X-Y
   - distance-grid orbit gallery

5. RV visualization from the old plot_rv notebook:
   - Aitoff all-sky no-bar vs with-bar
   - RA-Dec no-bar vs with-bar
   - no-bar distance grid
   - bar-vs-no-bar distance grid

6. Static summary figure:
   - q_z comparison with literature (from the old notebook)

Notes
-----
- The script is designed to *consume* existing products from the refactor run directory.
- Expensive parts (especially RV distance grids) are enabled by default but can be
  turned off with CLI flags if needed.
- Most figures are saved in both PNG and PDF.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
try:
    import pandas as pd
    PANDAS_ERROR = None
except Exception as e:
    pd = None
    PANDAS_ERROR = e

try:
    import matplotlib
    if hasattr(matplotlib, "use"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_ERROR = None
except Exception as e:
    matplotlib = plt = None
    MATPLOTLIB_ERROR = e

try:
    import corner
except Exception:
    corner = None

try:
    from astropy.io import fits
    from astropy.table import Table
    import astropy.coordinates as coord
    import astropy.units as u

    import gala.coordinates as gc
    import gala.dynamics as gd
    from gala.dynamics import mockstream as ms
    import gala.potential as gp
    from gala.units import galactic

    _ = coord.galactocentric_frame_defaults.set("v4.0")
    ASTRO_DEPS_ERROR = None
except Exception as e:
    fits = Table = coord = u = gc = gd = ms = gp = galactic = None
    ASTRO_DEPS_ERROR = e


# ---------------------------------------------------------------------
# Styling and small helpers
# ---------------------------------------------------------------------

if plt is not None:
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 140,
    })


PRETTY_LABELS = {
    "log10_mhalo": r"$\log_{10}(M_{\rm halo}/M_\odot)$",
    "r_s": r"$r_s\ [{\rm kpc}]$",
    "q_y": r"$q_y$",
    "q_z": r"$q_z$",
    "prog_mass": r"$M_{\rm prog}\ [M_\odot]$",
    "pm_ra_cosdec": r"$\mu_{\alpha*}\ [{\rm mas\,yr^{-1}}]$",
    "pm_dec": r"$\mu_{\delta}\ [{\rm mas\,yr^{-1}}]$",
    "distance": r"$d\ [{\rm kpc}]$",
}

TRACK_COLUMN_ALIASES = {
    "phi1": ("phi1", "phi1_center"),
    "phi2": ("phi2", "mu"),
    "phi2_err": ("phi2_err", "mu_err"),
    "width": ("width", "sigma"),
    "width_err": ("width_err", "sigma_err"),
    "counts": ("counts", "nstar", "nstars"),
    "density": ("density", "integrated_counts", "linear_density"),
}



def require_astropy_gala() -> None:
    if ASTRO_DEPS_ERROR is not None:
        raise ImportError(
            "This plotting suite requires astropy and gala in the runtime environment. "
            f"Original import error: {ASTRO_DEPS_ERROR}"
        )


def require_matplotlib() -> None:
    if MATPLOTLIB_ERROR is not None:
        raise ImportError(
            "This plotting suite requires matplotlib in the runtime environment. "
            f"Original import error: {MATPLOTLIB_ERROR}"
        )


def require_pandas() -> None:
    if PANDAS_ERROR is not None:
        raise ImportError(
            "This plotting suite requires pandas in the runtime environment. "
            f"Original import error: {PANDAS_ERROR}"
        )

def savefig_multi(fig: plt.Figure, outbase: Path, dpi: int = 300) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_core_module(core_path: Path):
    spec = importlib.util.spec_from_file_location("pal5_core_refactor", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import core module from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pal5_core_refactor"] = module
    spec.loader.exec_module(module)
    return module


def read_table_any(path: str | Path) -> Table:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return Table.read(path)


def read_best_fit_params(path: Path, summary_fallback: Path | None = None) -> dict[str, float]:
    if path.exists():
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"Expected one-row best-fit parameter file: {path}")
        return {k: float(v) for k, v in df.iloc[0].to_dict().items()}

    if summary_fallback is not None and summary_fallback.exists():
        sdf = pd.read_csv(summary_fallback)
        return {row["parameter"]: float(row["q50"]) for _, row in sdf.iterrows()}

    raise FileNotFoundError(f"Could not find {path} or summary fallback.")


def resolve_track_column(tab: Table, requested: str | None, role: str) -> str | None:
    if requested is not None:
        return requested if requested in tab.colnames else None
    for name in TRACK_COLUMN_ALIASES.get(role, ()):
        if name in tab.colnames:
            return name
    return None


def load_observed_track(path: Path,
                        phi1_col: str | None = None,
                        phi2_col: str | None = None,
                        phi2_err_col: str | None = "phi2_err",
                        width_col: str | None = "width",
                        width_err_col: str | None = "width_err",
                        counts_col: str | None = None,
                        density_col: str | None = None) -> dict[str, np.ndarray]:
    tab = read_table_any(path)
    out: dict[str, np.ndarray] = {}
    phi1_col = resolve_track_column(tab, phi1_col, "phi1")
    phi2_col = resolve_track_column(tab, phi2_col, "phi2")
    phi2_err_col = resolve_track_column(tab, phi2_err_col, "phi2_err")
    width_col = resolve_track_column(tab, width_col, "width")
    width_err_col = resolve_track_column(tab, width_err_col, "width_err")
    counts_col = resolve_track_column(tab, counts_col, "counts")
    density_col = resolve_track_column(tab, density_col, "density")
    required = [phi1_col, phi2_col]
    for name in required:
        if name is None or name not in tab.colnames:
            raise ValueError(f"Observed track file missing required column '{name}'. Available: {tab.colnames}")

    out["phi1"] = np.asarray(tab[phi1_col], dtype=float)
    out["phi2"] = np.asarray(tab[phi2_col], dtype=float)

    def maybe(col: str | None):
        if col is None or col not in tab.colnames:
            return None
        return np.asarray(tab[col], dtype=float)

    out["phi2_err"] = maybe(phi2_err_col)
    out["width"] = maybe(width_col)
    out["width_err"] = maybe(width_err_col)
    out["counts"] = maybe(counts_col)
    out["density"] = maybe(density_col)

    order = np.argsort(out["phi1"])
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.shape == out["phi1"].shape:
            out[k] = v[order]
    return out


def load_model_track(path: Path) -> dict[str, np.ndarray]:
    tab = read_table_any(path)
    out = {
        "phi1": np.asarray(tab["phi1"], dtype=float),
        "phi2_model": np.asarray(tab["phi2_model"], dtype=float),
        "phi2_model_err": np.asarray(tab["phi2_model_err"], dtype=float),
        "width_model": np.asarray(tab["width_model"], dtype=float),
        "width_model_err": np.asarray(tab["width_model_err"], dtype=float),
        "counts": np.asarray(tab["counts"], dtype=float),
        "valid": np.asarray(tab["valid"], dtype=int).astype(bool),
    }
    order = np.argsort(out["phi1"])
    for k in out:
        out[k] = out[k][order]
    return out


def load_mock_particles(path: Path) -> dict[str, np.ndarray]:
    tab = read_table_any(path)
    return {
        "ra": np.asarray(tab["ra"], dtype=float),
        "dec": np.asarray(tab["dec"], dtype=float),
        "phi1": np.asarray(tab["phi1"], dtype=float),
        "phi2": np.asarray(tab["phi2"], dtype=float),
    }


def load_star_catalog(path: Path, ra_col: str = "RA", dec_col: str = "DEC",
                      distance_col: str | None = None, max_distance: float | None = None,
                      core=None) -> dict[str, np.ndarray]:
    tab = read_table_any(path)
    if ra_col not in tab.colnames or dec_col not in tab.colnames:
        raise ValueError(f"Star catalog must contain {ra_col} and {dec_col}. Available: {tab.colnames}")

    mask = np.ones(len(tab), dtype=bool)
    if distance_col is not None and max_distance is not None and distance_col in tab.colnames:
        mask &= np.asarray(tab[distance_col], dtype=float) < max_distance

    ra = np.asarray(tab[ra_col], dtype=float)[mask]
    dec = np.asarray(tab[dec_col], dtype=float)[mask]
    if core is None:
        c = coord.ICRS(ra=ra * u.deg, dec=dec * u.deg)
        pal5 = c.transform_to(gc.Pal5PriceWhelan18())
        phi1 = pal5.phi1.wrap_at(180 * u.deg).degree
        phi2 = pal5.phi2.degree
    else:
        phi1, phi2 = core.icrs_to_pal5(ra * u.deg, dec * u.deg)

    return {"ra": ra, "dec": dec, "phi1": phi1, "phi2": phi2}


def robust_finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(np.asarray(arrays[0], dtype=float), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


# ---------------------------------------------------------------------
# Plotting: track / stream / MCMC
# ---------------------------------------------------------------------

def plot_stream_overlay_pal5(observed: dict[str, np.ndarray],
                             model_track: dict[str, np.ndarray],
                             particles: dict[str, np.ndarray],
                             outdir: Path,
                             star_catalog: dict[str, np.ndarray] | None = None) -> None:
    # with background
    fig, ax = plt.subplots(figsize=(18, 4.8))
    if star_catalog is not None:
        hb = ax.hexbin(star_catalog["phi1"], star_catalog["phi2"], gridsize=420,
                       bins="log", mincnt=1, cmap="inferno")
        cbar = fig.colorbar(hb, ax=ax, pad=0.01)
        cbar.set_label("log N")
    ax.scatter(particles["phi1"], particles["phi2"], s=4, alpha=0.08,
               color="orange", linewidths=0, label="Mock particles")
    ax.errorbar(observed["phi1"], observed["phi2"],
                yerr=observed["phi2_err"] if observed["phi2_err"] is not None else None,
                fmt="o", ms=4.5, color="deepskyblue", ecolor="deepskyblue",
                elinewidth=1, alpha=0.95, label="Observed track")
    valid = model_track["valid"]
    ax.plot(model_track["phi1"][valid], model_track["phi2_model"][valid],
            color="lime", lw=2.5, label="Model track")
    if observed.get("width") is not None:
        ax.fill_between(observed["phi1"],
                        observed["phi2"] - observed["width"],
                        observed["phi2"] + observed["width"],
                        color="deepskyblue", alpha=0.12, linewidth=0)
    finite_w = valid & np.isfinite(model_track["width_model"])
    if np.any(finite_w):
        ax.fill_between(model_track["phi1"][finite_w],
                        model_track["phi2_model"][finite_w] - model_track["width_model"][finite_w],
                        model_track["phi2_model"][finite_w] + model_track["width_model"][finite_w],
                        color="lime", alpha=0.10, linewidth=0)

    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_xlim(np.nanmin(observed["phi1"]) - 1.5, np.nanmax(observed["phi1"]) + 1.5)
    all_phi2 = [observed["phi2"], model_track["phi2_model"][valid], particles["phi2"]]
    ymin = np.nanpercentile(np.concatenate([x[np.isfinite(x)] for x in all_phi2]), 1)
    ymax = np.nanpercentile(np.concatenate([x[np.isfinite(x)] for x in all_phi2]), 99)
    ax.set_ylim(ymin - 0.6, ymax + 0.6)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Pal 5 stream overlay in Pal 5 coordinates")
    savefig_multi(fig, outdir / "01_stream_overlay_pal5")

    # particles + tracks only
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.scatter(particles["phi1"], particles["phi2"], s=5, alpha=0.10,
               color="orange", linewidths=0, label="Mock particles")
    ax.errorbar(observed["phi1"], observed["phi2"],
                yerr=observed["phi2_err"] if observed["phi2_err"] is not None else None,
                fmt="o", ms=5, color="deepskyblue", ecolor="deepskyblue",
                elinewidth=1, label="Observed track")
    ax.plot(model_track["phi1"][valid], model_track["phi2_model"][valid],
            color="black", lw=2.5, label="Model track")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_xlim(np.nanmin(observed["phi1"]) - 1.5, np.nanmax(observed["phi1"]) + 1.5)
    ax.legend(loc="upper right")
    ax.set_title("Pal 5 track fit without background catalog")
    savefig_multi(fig, outdir / "02_stream_overlay_tracks_only")


def plot_track_diagnostics(observed: dict[str, np.ndarray],
                           model_track: dict[str, np.ndarray],
                           outdir: Path) -> None:
    valid = model_track["valid"] & np.isfinite(observed["phi2"])
    phi1 = observed["phi1"][valid]
    obs = observed["phi2"][valid]
    mod = model_track["phi2_model"][valid]
    residual = mod - obs

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.errorbar(observed["phi1"], observed["phi2"],
                 yerr=observed["phi2_err"] if observed["phi2_err"] is not None else None,
                 fmt="o", ms=4.5, color="deepskyblue", ecolor="deepskyblue",
                 elinewidth=1, label="Observed")
    ax0.plot(model_track["phi1"][model_track["valid"]],
             model_track["phi2_model"][model_track["valid"]],
             color="orange", lw=2.5, label="Model")
    ax0.set_ylabel(r"$\phi_2$ [deg]")
    ax0.legend(loc="best")
    ax0.set_title("Track comparison")

    ax1.axhline(0, color="0.3", lw=1, ls="--")
    ax1.plot(phi1, residual, "o-", color="crimson", ms=4)
    ax1.set_xlabel(r"$\phi_1$ [deg]")
    ax1.set_ylabel(r"$\Delta\phi_2$ [deg]")
    ax1.set_title("Model - observed residual")
    savefig_multi(fig, outdir / "03_track_comparison_residual")


def plot_width_and_counts(observed: dict[str, np.ndarray],
                          model_track: dict[str, np.ndarray],
                          outdir: Path) -> None:
    has_obs_width = observed.get("width") is not None

    nrows = 2 if has_obs_width else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4.5 + 3.2 * (nrows - 1)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    if has_obs_width:
        ax = axes[0]
        mask_obs = np.isfinite(observed["width"])
        mask_mod = model_track["valid"] & np.isfinite(model_track["width_model"])
        ax.errorbar(observed["phi1"][mask_obs], observed["width"][mask_obs],
                    yerr=observed["width_err"][mask_obs] if observed.get("width_err") is not None else None,
                    fmt="o", ms=4.5, color="deepskyblue", ecolor="deepskyblue",
                    elinewidth=1, label="Observed width")
        ax.plot(model_track["phi1"][mask_mod], model_track["width_model"][mask_mod],
                "o-", color="orange", ms=4, lw=2, label="Model width")
        ax.set_ylabel("Width [deg]")
        ax.legend(loc="best")
        ax.set_title("Width comparison")

    ax = axes[-1]
    ax.plot(model_track["phi1"], model_track["counts"], "o-", color="black", ms=4, lw=1.8, label="Model particles / node")
    if observed.get("counts") is not None:
        mask = np.isfinite(observed["counts"])
        ax.plot(observed["phi1"][mask], observed["counts"][mask], "o-", color="deepskyblue",
                ms=4, lw=1.8, label="Observed counts")
    if observed.get("density") is not None:
        mask = np.isfinite(observed["density"])
        ax.plot(observed["phi1"][mask], observed["density"][mask], "o-", color="deepskyblue",
                ms=4, lw=1.8, label="Observed density proxy")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel("Count / proxy")
    ax.set_title("Track-node support along the stream")
    ax.legend(loc="best")

    savefig_multi(fig, outdir / "04_width_and_counts")


def plot_bonaca_style_summary(observed: dict[str, np.ndarray],
                              model_track: dict[str, np.ndarray],
                              particles: dict[str, np.ndarray],
                              outdir: Path,
                              star_catalog: dict[str, np.ndarray] | None = None) -> None:
    valid = model_track["valid"] & np.isfinite(observed["phi2"])
    phi1 = observed["phi1"][valid]
    resid = model_track["phi2_model"][valid] - observed["phi2"][valid]

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.0, 1.0, 1.0], hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    if star_catalog is not None:
        hb = ax0.hexbin(star_catalog["phi1"], star_catalog["phi2"], gridsize=320,
                        bins="log", mincnt=1, cmap="inferno")
        cbar = fig.colorbar(hb, ax=ax0, pad=0.01, fraction=0.02)
        cbar.set_label("log N")
    ax0.scatter(particles["phi1"], particles["phi2"], s=4, alpha=0.08, color="orange", linewidths=0)
    ax0.plot(observed["phi1"], observed["phi2"], "o", color="deepskyblue", ms=4, label="Observed track")
    ax0.plot(model_track["phi1"][model_track["valid"]], model_track["phi2_model"][model_track["valid"]],
             color="lime", lw=2.0, label="Model track")
    ax0.set_ylabel(r"$\phi_2$ [deg]")
    ax0.set_title("Morphology + track summary")
    ax0.legend(loc="upper right")

    ax1.plot(observed["phi1"], observed["phi2"], "o", color="deepskyblue", ms=4)
    ax1.plot(model_track["phi1"][model_track["valid"]], model_track["phi2_model"][model_track["valid"]],
             color="black", lw=2)
    ax1.set_ylabel(r"$\phi_2$")

    ax2.axhline(0, color="0.3", lw=1, ls="--")
    ax2.plot(phi1, resid, "o-", color="crimson", ms=4)
    ax2.set_ylabel(r"$\Delta\phi_2$")

    if observed.get("width") is not None and np.any(np.isfinite(observed["width"])):
        mask_obs = np.isfinite(observed["width"])
        mask_mod = model_track["valid"] & np.isfinite(model_track["width_model"])
        ax3.plot(observed["phi1"][mask_obs], observed["width"][mask_obs], "o-", color="deepskyblue", ms=4, label="Obs width")
        ax3.plot(model_track["phi1"][mask_mod], model_track["width_model"][mask_mod], "o-", color="orange", ms=4, label="Model width")
        ax3.set_ylabel("Width [deg]")
        ax3.legend(loc="best")
    else:
        ax3.plot(model_track["phi1"], model_track["counts"], "o-", color="black", ms=4, label="Model counts")
        ax3.set_ylabel("Counts")
        ax3.legend(loc="best")
    ax3.set_xlabel(r"$\phi_1$ [deg]")

    savefig_multi(fig, outdir / "05_bonaca_style_summary")


def plot_corner_and_chains(run_dir: Path, outdir: Path) -> None:
    samples_path = run_dir / "mcmc_samples.csv"
    chain_path = run_dir / "chain.npy"
    logprob_path = run_dir / "log_prob.npy"

    if samples_path.exists():
        samples_df = pd.read_csv(samples_path)
        cols = list(samples_df.columns)
        labels = [PRETTY_LABELS.get(c, c) for c in cols]
        samples = samples_df.values

        if corner is not None:
            fig = corner.corner(
                samples,
                labels=labels,
                show_titles=True,
                title_fmt=".3f",
                smooth=1.0,
                quantiles=[0.16, 0.50, 0.84],
                color="deepskyblue",
                hist_kwargs={"color": "skyblue", "alpha": 0.8},
            )
        else:
            fig, axes = plt.subplots(len(cols), len(cols), figsize=(2.6 * len(cols), 2.6 * len(cols)))
            axes = np.asarray(axes)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax = axes[i, j]
                    if i == j:
                        ax.hist(samples[:, j], bins=35, color="skyblue", alpha=0.8)
                    elif i > j:
                        ax.plot(samples[:, j], samples[:, i], "k.", ms=0.7, alpha=0.15)
                    else:
                        ax.axis("off")
                    if i == len(cols) - 1 and j <= i:
                        ax.set_xlabel(labels[j])
                    if j == 0 and i > 0:
                        ax.set_ylabel(labels[i])
        savefig_multi(fig, outdir / "06_mcmc_corner")

    if chain_path.exists():
        chain = np.load(chain_path)
        nwalkers, nsteps, ndim = chain.shape
        param_names = [PRETTY_LABELS.get(c, c) for c in pd.read_csv(samples_path, nrows=1).columns] if samples_path.exists() else [f"p{i}" for i in range(ndim)]
        fig, axes = plt.subplots(ndim, 1, figsize=(13, 2.1 * ndim), sharex=True)
        if ndim == 1:
            axes = [axes]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(chain[:, :, i].T, color="k", alpha=0.08, lw=0.6)
            ax.set_ylabel(param_names[i])
        axes[-1].set_xlabel("Step")
        fig.suptitle("Walker chains", y=0.995)
        fig.tight_layout()
        savefig_multi(fig, outdir / "07_mcmc_chains")

    if logprob_path.exists():
        logp = np.load(logprob_path)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(logp.T, color="k", alpha=0.08, lw=0.6)
        ax.set_xlabel("Step")
        ax.set_ylabel("log probability")
        ax.set_title("MCMC log-probability")
        savefig_multi(fig, outdir / "08_mcmc_logprob")


# ---------------------------------------------------------------------
# Orbit helpers and plots
# ---------------------------------------------------------------------

def build_galcen_frame(galcen_distance_kpc: float = 8.275,
                       v_sun_kms: tuple[float, float, float] = (8.4, 244.8, 8.4)) -> coord.Galactocentric:
    v_sun = coord.CartesianDifferential(np.array(v_sun_kms) * u.km / u.s)
    return coord.Galactocentric(galcen_distance=galcen_distance_kpc * u.kpc, galcen_v_sun=v_sun)


def build_base_components(params: dict[str, float]):
    bulge = gp.HernquistPotential(m=3.4e10 * u.Msun, c=0.7 * u.kpc, units=galactic)
    disk = gp.MiyamotoNagaiPotential(m=1.0e11 * u.Msun, a=6.5 * u.kpc, b=0.26 * u.kpc, units=galactic)
    halo = gp.NFWPotential(
        m=(10.0 ** params["log10_mhalo"]) * u.Msun,
        r_s=params["r_s"] * u.kpc,
        a=1.0,
        b=params.get("q_y", 1.0),
        c=params["q_z"],
        units=galactic,
    )
    return bulge, disk, halo


def build_hamiltonian(params: dict[str, float],
                      include_bar: bool = False,
                      pattern_speed: float = 42.0,
                      bar_mass: float = 1.0e10,
                      bar_alpha_deg: float = 27.0) -> gp.Hamiltonian:
    bulge, disk, halo = build_base_components(params)
    if include_bar:
        bar = gp.LongMuraliBarPotential(
            m=bar_mass * u.Msun,
            a=3.5 * u.kpc,
            b=0.5 * u.kpc,
            c=0.5 * u.kpc,
            alpha=bar_alpha_deg * u.deg,
            units=galactic,
        )
        pot = gp.CCompositePotential(disk=disk, halo=halo, bulge=bulge, bar=bar)
        rot_frame = gp.ConstantRotatingFrame(pattern_speed * u.km / u.s / u.kpc * [0, 0, -1], units=pot.units)
        return gp.Hamiltonian(pot, rot_frame)
    else:
        pot = gp.CCompositePotential(disk=disk, halo=halo, bulge=bulge)
        return gp.Hamiltonian(pot)


def progenitor_icrs_from_params(params: dict[str, float],
                                ra_deg: float = 229.018,
                                dec_deg: float = -0.124,
                                rv_kms: float = -58.7,
                                distance_override: float | None = None) -> coord.ICRS:
    return coord.ICRS(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        distance=(distance_override if distance_override is not None else params["distance"]) * u.kpc,
        pm_ra_cosdec=params["pm_ra_cosdec"] * u.mas / u.yr,
        pm_dec=params["pm_dec"] * u.mas / u.yr,
        radial_velocity=rv_kms * u.km / u.s,
    )


def integrate_orbit(params: dict[str, float],
                    include_bar: bool = False,
                    pattern_speed: float = 42.0,
                    step_myr: float = 0.15,
                    n_steps: int = 115000,
                    distance_override: float | None = None):
    ham = build_hamiltonian(params, include_bar=include_bar, pattern_speed=pattern_speed)
    galcen_frame = build_galcen_frame()
    stream_pro = progenitor_icrs_from_params(params, distance_override=distance_override)
    w0 = gd.PhaseSpacePosition(stream_pro.transform_to(galcen_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=-step_myr * u.Myr, n_steps=n_steps)
    return orbit


def plot_bestfit_orbit(params: dict[str, float], outdir: Path,
                       step_myr: float = 0.15, n_steps: int = 115000) -> None:
    orbit = integrate_orbit(params, include_bar=False, step_myr=step_myr, n_steps=n_steps)
    x = orbit.x.to_value(u.kpc)
    y = orbit.y.to_value(u.kpc)
    z = orbit.z.to_value(u.kpc)
    R = np.sqrt(x**2 + y**2)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    ax[0].plot(R, z, lw=1.2, color="black")
    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("Z [kpc]")
    ax[0].set_title("Best-fit orbit: R-Z")

    ax[1].plot(x, y, lw=1.2, color="black")
    ax[1].set_xlabel("X [kpc]")
    ax[1].set_ylabel("Y [kpc]")
    ax[1].set_title("Best-fit orbit: X-Y")

    savefig_multi(fig, outdir / "09_bestfit_orbit")


def plot_distance_grid_orbits(params: dict[str, float], outdir: Path,
                              dmin: float = 20.0, dmax: float = 25.0, n_dist: int = 10,
                              step_myr: float = 0.15, n_steps: int = 115000) -> None:
    dist_values = np.linspace(dmin, dmax, n_dist)
    nrows = math.ceil(n_dist / 2)
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.1 * nrows))
    axes = np.atleast_2d(axes)

    for i, dist in enumerate(dist_values):
        orbit = integrate_orbit(params, include_bar=False, step_myr=step_myr,
                                n_steps=n_steps, distance_override=float(dist))
        x = orbit.x.to_value(u.kpc)
        y = orbit.y.to_value(u.kpc)
        z = orbit.z.to_value(u.kpc)
        R = np.sqrt(x**2 + y**2)

        row = i // 2
        col = (i % 2) * 2
        ax_rz = axes[row, col]
        ax_xy = axes[row, col + 1]

        ax_rz.plot(R, z, lw=1.0, color="black")
        ax_rz.set_title(f"R-Z, d={dist:.1f} kpc")
        ax_rz.set_xlabel("R [kpc]")
        ax_rz.set_ylabel("Z [kpc]")

        ax_xy.plot(x, y, lw=1.0, color="black")
        ax_xy.set_title(f"X-Y, d={dist:.1f} kpc")
        ax_xy.set_xlabel("X [kpc]")
        ax_xy.set_ylabel("Y [kpc]")

    # hide unused axes if any
    total = nrows * 4
    used = n_dist * 2
    for j in range(used, total):
        rr, cc = divmod(j, 4)
        axes[rr, cc].axis("off")

    fig.suptitle("Orbit gallery vs assumed distance", y=0.995)
    fig.tight_layout()
    savefig_multi(fig, outdir / "10_orbit_distance_grid")


# ---------------------------------------------------------------------
# RV mock-stream helpers and plots
# ---------------------------------------------------------------------

def generate_stream_for_rv(params: dict[str, float],
                           include_bar: bool,
                           pattern_speed: float = 42.0,
                           step_myr: float = 1.0,
                           n_steps: int = 11500,
                           distance_override: float | None = None,
                           prog_mass: float | None = None):
    prog_mass_use = params.get("prog_mass", 3e4 if prog_mass is None else prog_mass)
    if prog_mass is not None:
        prog_mass_use = prog_mass

    ham = build_hamiltonian(params, include_bar=include_bar, pattern_speed=pattern_speed)
    galcen_frame = build_galcen_frame()
    stream_pro = progenitor_icrs_from_params(params, distance_override=distance_override)
    stream_w0 = gd.PhaseSpacePosition(stream_pro.transform_to(galcen_frame).cartesian)
    df = ms.FardalStreamDF()
    const_pro_mass = prog_mass_use * u.Msun
    const_pro_pot = gp.PlummerPotential(m=const_pro_mass, b=4 * u.pc, units=galactic)
    gen_stream = ms.MockStreamGenerator(df, ham, progenitor_potential=const_pro_pot)

    stream, _ = gen_stream.run(stream_w0, const_pro_mass,
                               dt=-step_myr * u.Myr, n_steps=n_steps,
                               release_every=1, n_particles=1, progress=False)
    sky = stream.to_coord_frame(coord.ICRS(), galactocentric_frame=galcen_frame)
    return sky


def mask_by_release_and_region(ra, dec, rv,
                               ra_min=210, ra_max=270,
                               dec_min=-40, dec_max=20,
                               frac_threshold=0.7):
    ra_deg = ra.degree
    dec_deg = dec.degree
    rv_val = rv.to_value(u.km / u.s)

    valid = ~np.isnan(ra_deg) & ~np.isnan(dec_deg) & ~np.isnan(rv_val)
    n_valid = np.sum(valid)
    valid_indices = np.where(valid)[0]
    threshold = int(frac_threshold * n_valid)

    mask_release = np.zeros(len(ra_deg), dtype=bool)
    mask_release[valid_indices[threshold:]] = True
    mask_region = (ra_deg >= ra_min) & (ra_deg <= ra_max) & (dec_deg >= dec_min) & (dec_deg <= dec_max)

    return mask_release & mask_region


def wrap_ra_to_aitoff(ra):
    ra_deg = ra.to(u.deg).value
    ra_wrap = np.where(ra_deg > 180, ra_deg - 360, ra_deg)
    return np.deg2rad(ra_wrap)


def plot_rv_aitoff(params: dict[str, float], outdir: Path,
                   pattern_speed: float = 42.0,
                   step_myr: float = 1.0, n_steps: int = 11500) -> None:
    s_nobar = generate_stream_for_rv(params, include_bar=False,
                                     step_myr=step_myr, n_steps=n_steps)
    s_bar = generate_stream_for_rv(params, include_bar=True,
                                   pattern_speed=pattern_speed,
                                   step_myr=step_myr, n_steps=n_steps)

    ra_nb, dec_nb, rv_nb = s_nobar.ra, s_nobar.dec, s_nobar.radial_velocity
    ra_b, dec_b, rv_b = s_bar.ra, s_bar.dec, s_bar.radial_velocity

    vmin = np.nanmin([np.nanmin(rv_nb.to_value(u.km/u.s)), np.nanmin(rv_b.to_value(u.km/u.s))])
    vmax = np.nanmax([np.nanmax(rv_nb.to_value(u.km/u.s)), np.nanmax(rv_b.to_value(u.km/u.s))])

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.06], hspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0], projection="aitoff")
    ax2 = fig.add_subplot(gs[0, 1], projection="aitoff")
    cax = fig.add_subplot(gs[1, :])

    sc1 = ax1.scatter(wrap_ra_to_aitoff(ra_nb), dec_nb.to(u.rad).value,
                      c=rv_nb.to_value(u.km/u.s), s=4, cmap="rainbow",
                      vmin=vmin, vmax=vmax, edgecolors="none")
    sc2 = ax2.scatter(wrap_ra_to_aitoff(ra_b), dec_b.to(u.rad).value,
                      c=rv_b.to_value(u.km/u.s), s=4, cmap="rainbow",
                      vmin=vmin, vmax=vmax, edgecolors="none")

    xticks_deg = np.arange(-150, 181, 30)
    xticks_rad = np.deg2rad(xticks_deg)
    xtick_lbls = [(x + 360) % 360 for x in xticks_deg]
    for ax in (ax1, ax2):
        ax.set_xticks(xticks_rad)
        ax.set_xticklabels(xtick_lbls)
        ax.grid(True, alpha=0.4)

    ax1.set_title("Mock stream (no bar)")
    ax2.set_title(rf"Mock stream (with bar), $\Omega_{{\rm bar}}={pattern_speed:.0f}$")

    cbar = fig.colorbar(sc1, cax=cax, orientation="horizontal")
    cbar.set_label("Radial velocity [km/s]")

    savefig_multi(fig, outdir / "11_rv_aitoff_bar_vs_nobar")


def plot_rv_radec(params: dict[str, float], outdir: Path,
                  pattern_speed: float = 42.0,
                  step_myr: float = 1.0, n_steps: int = 11500,
                  frac_threshold: float = 0.7) -> None:
    s_nobar = generate_stream_for_rv(params, include_bar=False, step_myr=step_myr, n_steps=n_steps)
    s_bar = generate_stream_for_rv(params, include_bar=True, pattern_speed=pattern_speed,
                                   step_myr=step_myr, n_steps=n_steps)

    ra_nb, dec_nb, rv_nb = s_nobar.ra, s_nobar.dec, s_nobar.radial_velocity
    ra_b, dec_b, rv_b = s_bar.ra, s_bar.dec, s_bar.radial_velocity
    mask_nb = mask_by_release_and_region(ra_nb, dec_nb, rv_nb, frac_threshold=frac_threshold)
    mask_b = mask_by_release_and_region(ra_b, dec_b, rv_b, frac_threshold=frac_threshold)

    vmin = min(np.nanmin(rv_nb[mask_nb].to_value(u.km/u.s)), np.nanmin(rv_b[mask_b].to_value(u.km/u.s)))
    vmax = max(np.nanmax(rv_nb[mask_nb].to_value(u.km/u.s)), np.nanmax(rv_b[mask_b].to_value(u.km/u.s)))

    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.06], hspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    sc1 = ax1.scatter(ra_nb[mask_nb].value, dec_nb[mask_nb].value,
                      c=rv_nb[mask_nb].value, s=4, cmap="rainbow",
                      vmin=vmin, vmax=vmax)
    sc2 = ax2.scatter(ra_b[mask_b].value, dec_b[mask_b].value,
                      c=rv_b[mask_b].value, s=4, cmap="rainbow",
                      vmin=vmin, vmax=vmax)

    for ax, title in zip((ax1, ax2), ("No bar", "With bar")):
        ax.set_xlim(210, 270)
        ax.set_ylim(-40, 20)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.scatter(229.65, 0.26, marker="*", color="red", s=150, zorder=5)
        ax.set_title(title)

    cbar = fig.colorbar(sc1, cax=cax, orientation="horizontal")
    cbar.set_label("Radial velocity [km/s]")

    savefig_multi(fig, outdir / "12_rv_radec_bar_vs_nobar")


def run_rv_distance_grid(params: dict[str, float], distances: np.ndarray,
                         include_bar: bool, pattern_speed: float,
                         step_myr: float, n_steps: int):
    results = []
    for d in distances:
        sky = generate_stream_for_rv(params, include_bar=include_bar,
                                     pattern_speed=pattern_speed,
                                     step_myr=step_myr, n_steps=n_steps,
                                     distance_override=float(d))
        results.append((float(d), sky))
    return results


def plot_rv_distance_grid_nobar(params: dict[str, float], outdir: Path,
                                dmin: float = 15.0, dmax: float = 24.5, n_dist: int = 20,
                                step_myr: float = 1.0, n_steps: int = 11500,
                                frac_threshold: float = 0.7) -> None:
    distances = np.linspace(dmin, dmax, n_dist)
    results = run_rv_distance_grid(params, distances, include_bar=False,
                                   pattern_speed=42.0, step_myr=step_myr, n_steps=n_steps)

    all_rvs = []
    for _, sky in results:
        mask = mask_by_release_and_region(sky.ra, sky.dec, sky.radial_velocity, frac_threshold=frac_threshold)
        if np.any(mask):
            all_rvs.append(sky.radial_velocity[mask].value)
    if not all_rvs:
        return

    vmin = np.min([rv.min() for rv in all_rvs])
    vmax = np.max([rv.max() for rv in all_rvs])

    ncols = 4
    nrows = math.ceil(n_dist / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, (dist, sky) in enumerate(results):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        mask = mask_by_release_and_region(sky.ra, sky.dec, sky.radial_velocity, frac_threshold=frac_threshold)
        sc = ax.scatter(sky.ra[mask].value, sky.dec[mask].value, c=sky.radial_velocity[mask].value,
                        s=4, cmap="rainbow", vmin=vmin, vmax=vmax)
        ax.set_title(f"d = {dist:.1f} kpc")
        ax.set_xlim(210, 270)
        ax.set_ylim(-40, 20)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.35)
        ax.scatter(229.65, 0.26, marker="*", color="red", s=120)

        if c == 0:
            ax.set_ylabel("Dec [deg]")
        if r == nrows - 1:
            ax.set_xlabel("RA [deg]")

    for j in range(n_dist, nrows * ncols):
        rr, cc = divmod(j, ncols)
        axes[rr, cc].axis("off")

    fig.subplots_adjust(right=0.88, hspace=0.40, wspace=0.28)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Radial velocity [km/s]")

    savefig_multi(fig, outdir / "13_rv_distance_grid_nobar")


def plot_rv_distance_grid_bar_vs_nobar(params: dict[str, float], outdir: Path,
                                       dmin: float = 15.0, dmax: float = 24.5, n_pairs: int = 10,
                                       pattern_speed: float = 42.0,
                                       step_myr: float = 1.0, n_steps: int = 11500,
                                       frac_threshold: float = 0.7) -> None:
    distances = np.linspace(dmin, dmax, n_pairs)
    nobar = run_rv_distance_grid(params, distances, include_bar=False,
                                 pattern_speed=pattern_speed, step_myr=step_myr, n_steps=n_steps)
    bar = run_rv_distance_grid(params, distances, include_bar=True,
                               pattern_speed=pattern_speed, step_myr=step_myr, n_steps=n_steps)

    vmin, vmax = -140, 140
    nrows = n_pairs
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.4 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for i in range(n_pairs):
        dist_nb, sky_nb = nobar[i]
        dist_b, sky_b = bar[i]
        assert abs(dist_nb - dist_b) < 1e-8

        ax_nb, ax_b = axes[i, 0], axes[i, 1]
        mask_nb = mask_by_release_and_region(sky_nb.ra, sky_nb.dec, sky_nb.radial_velocity, frac_threshold=frac_threshold)
        mask_b = mask_by_release_and_region(sky_b.ra, sky_b.dec, sky_b.radial_velocity, frac_threshold=frac_threshold)

        sc = ax_nb.scatter(sky_nb.ra[mask_nb].value, sky_nb.dec[mask_nb].value,
                           c=sky_nb.radial_velocity[mask_nb].value, s=4, cmap="rainbow", vmin=vmin, vmax=vmax)
        ax_b.scatter(sky_b.ra[mask_b].value, sky_b.dec[mask_b].value,
                     c=sky_b.radial_velocity[mask_b].value, s=4, cmap="rainbow", vmin=vmin, vmax=vmax)

        ax_nb.set_title(f"No bar, d={dist_nb:.1f} kpc")
        ax_b.set_title(f"With bar, d={dist_b:.1f} kpc")

        for ax in (ax_nb, ax_b):
            ax.set_xlim(210, 270)
            ax.set_ylim(-40, 20)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.35)
            ax.scatter(229.65, 0.26, marker="*", color="red", s=100)

        ax_nb.set_ylabel("Dec [deg]")
        if i == nrows - 1:
            ax_nb.set_xlabel("RA [deg]")
            ax_b.set_xlabel("RA [deg]")

    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.22)
    cbar_ax = fig.add_axes([0.90, 0.13, 0.015, 0.74])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Radial velocity [km/s]")

    savefig_multi(fig, outdir / "14_rv_distance_grid_bar_vs_nobar")


# ---------------------------------------------------------------------
# Static literature figure
# ---------------------------------------------------------------------

def plot_qz_literature_comparison(outdir: Path) -> None:
    works = ['This work', 'Huang et al. (2024)', 'Ibata et al. (2024)',
             'Palau & Miralda-Escude (2023)', 'Malhan & Ibata (2019)',
             'Bovy et al. (2016)', 'Küpper et al. (2015)']
    q_z_values = [0.93, 0.90, 0.89, 0.96, 0.82, 0.93, 0.95]
    error_left = [0.09, 0.06, 0.00, 0.02, 0.13, 0.03, 0.12]
    error_right = [0.06, 0.06, 0.00, 0.01, 0.25, 0.03, 0.16]

    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(works)))
    y_values = np.arange(len(works))

    fig, ax = plt.subplots(figsize=(10.5, 8))
    for i in range(len(works)):
        xerr = np.array([[error_left[i]], [error_right[i]]])
        color = 'red' if works[i] == 'This work' else colors[i]
        ax.errorbar(q_z_values[i], y_values[i], xerr=xerr, fmt='o',
                    color=color, ecolor=color, capsize=8, markersize=10,
                    linewidth=2.5, capthick=2.2)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_values)
    ax.set_yticklabels(works)
    ax.set_xlabel(r'$q_z$')
    ax.set_title(r'$q_z$ comparison')
    savefig_multi(fig, outdir / "15_qz_literature_comparison")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 visualization suite")

    p.add_argument("--run-dir", type=Path, default=Path("pal5_mockfit_run"),
                   help="Directory containing best_fit_params.csv, model track, particles, chains, etc.")
    p.add_argument("--track-file", type=Path, default=None,
                   help="Observed track table. If omitted, defaults to run-dir/observed_track_used.fits.")
    p.add_argument("--star-file", type=Path, default=None,
                   help="Optional filtered star catalog for hexbin background.")
    p.add_argument("--star-ra-col", type=str, default="RA")
    p.add_argument("--star-dec-col", type=str, default="DEC")
    p.add_argument("--star-distance-col", type=str, default=None)
    p.add_argument("--star-max-distance", type=float, default=None)

    p.add_argument("--core-script", type=Path, default=Path("pal5_mock_track_fit_refactor.py"),
                   help="Path to the refactor core script.")
    p.add_argument("--outdir", type=Path, default=Path("pal5_plots"))

    p.add_argument("--pattern-speed", type=float, default=42.0)
    p.add_argument("--orbit-step-myr", type=float, default=0.15)
    p.add_argument("--orbit-nsteps", type=int, default=115000)
    p.add_argument("--rv-step-myr", type=float, default=1.0)
    p.add_argument("--rv-nsteps", type=int, default=11500)

    p.add_argument("--orbit-dmin", type=float, default=20.0)
    p.add_argument("--orbit-dmax", type=float, default=25.0)
    p.add_argument("--orbit-ndist", type=int, default=10)

    p.add_argument("--rv-dmin", type=float, default=15.0)
    p.add_argument("--rv-dmax", type=float, default=24.5)
    p.add_argument("--rv-ndist", type=int, default=20)
    p.add_argument("--rv-npairs", type=int, default=10)
    p.add_argument("--rv-frac-threshold", type=float, default=0.7)

    p.add_argument("--skip-rv", action="store_true", help="Skip all RV mock-stream figures.")
    p.add_argument("--skip-rv-grids", action="store_true", help="Skip RV distance-grid figures.")
    p.add_argument("--skip-orbit-grid", action="store_true", help="Skip distance-grid orbit gallery.")
    p.add_argument("--skip-literature", action="store_true", help="Skip static q_z comparison plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_matplotlib()
    require_pandas()
    require_astropy_gala()
    args.outdir.mkdir(parents=True, exist_ok=True)

    core = load_core_module(args.core_script)

    if args.track_file is None:
        args.track_file = args.run_dir / "observed_track_used.fits"
    if not args.track_file.exists():
        raise ValueError(
            f"Observed track file not found: {args.track_file}. "
            "Provide --track-file explicitly or make sure run-dir/observed_track_used.fits exists."
        )

    observed = load_observed_track(args.track_file)
    model_track = load_model_track(args.run_dir / "best_fit_model_track.fits")
    particles = load_mock_particles(args.run_dir / "best_fit_mock_stream_particles.fits")
    params = read_best_fit_params(args.run_dir / "best_fit_params.csv",
                                  summary_fallback=args.run_dir / "mcmc_summary.csv")

    star_catalog = None
    if args.star_file is not None:
        star_catalog = load_star_catalog(args.star_file,
                                         ra_col=args.star_ra_col,
                                         dec_col=args.star_dec_col,
                                         distance_col=args.star_distance_col,
                                         max_distance=args.star_max_distance,
                                         core=core)

    plot_stream_overlay_pal5(observed, model_track, particles, args.outdir, star_catalog=star_catalog)
    plot_track_diagnostics(observed, model_track, args.outdir)
    plot_width_and_counts(observed, model_track, args.outdir)
    plot_bonaca_style_summary(observed, model_track, particles, args.outdir, star_catalog=star_catalog)
    plot_corner_and_chains(args.run_dir, args.outdir)

    plot_bestfit_orbit(params, args.outdir,
                       step_myr=args.orbit_step_myr, n_steps=args.orbit_nsteps)
    if not args.skip_orbit_grid:
        plot_distance_grid_orbits(params, args.outdir,
                                  dmin=args.orbit_dmin, dmax=args.orbit_dmax,
                                  n_dist=args.orbit_ndist,
                                  step_myr=args.orbit_step_myr, n_steps=args.orbit_nsteps)

    if not args.skip_rv:
        plot_rv_aitoff(params, args.outdir,
                       pattern_speed=args.pattern_speed,
                       step_myr=args.rv_step_myr, n_steps=args.rv_nsteps)
        plot_rv_radec(params, args.outdir,
                      pattern_speed=args.pattern_speed,
                      step_myr=args.rv_step_myr, n_steps=args.rv_nsteps,
                      frac_threshold=args.rv_frac_threshold)
        if not args.skip_rv_grids:
            plot_rv_distance_grid_nobar(params, args.outdir,
                                        dmin=args.rv_dmin, dmax=args.rv_dmax,
                                        n_dist=args.rv_ndist,
                                        step_myr=args.rv_step_myr, n_steps=args.rv_nsteps,
                                        frac_threshold=args.rv_frac_threshold)
            plot_rv_distance_grid_bar_vs_nobar(params, args.outdir,
                                               dmin=args.rv_dmin, dmax=args.rv_dmax,
                                               n_pairs=args.rv_npairs,
                                               pattern_speed=args.pattern_speed,
                                               step_myr=args.rv_step_myr, n_steps=args.rv_nsteps,
                                               frac_threshold=args.rv_frac_threshold)

    if not args.skip_literature:
        plot_qz_literature_comparison(args.outdir)

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
