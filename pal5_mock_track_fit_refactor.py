from __future__ import annotations

"""
Pal 5 mock-stream generation + global track MCMC fitting.

This refactor is designed for the current project stage:
    1) the observed Pal 5 track has already been measured elsewhere,
    2) we do NOT re-extract the observed track from the star catalog here,
    3) the likelihood compares observed and model *track centroids* directly.

Why this refactor:
    - The old notebook mixed track extraction, mock generation, plotting, and MCMC
      in one place.
    - The old likelihood compared every mock particle to the observed track spline,
      which biases the fit toward particle sampling / stream width instead of the
      global centroid track.
    - The old `dm_dt` parameter was not mapped to a physical mass-loss model.
      In gala, `n_particles` controls released tracer counts, not mass-loss rate,
      so we remove `dm_dt` from the default fit.

This repository currently has two practical observed-track input styles:
    1) a clean track table with columns like `phi1`, `phi2`, `phi2_err`,
    2) the existing step3b profile table with columns like
       `phi1_center`, `mu`, `mu_err`, `sigma`, `sigma_err`.

The loader below accepts either form through simple column aliases.
"""

import argparse
from dataclasses import dataclass, field
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
from astropy.table import Table


TRACK_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "phi1": ("phi1", "phi1_center"),
    "phi2": ("phi2", "mu"),
    "phi2_err": ("phi2_err", "mu_err"),
    "width": ("width", "sigma"),
    "width_err": ("width_err", "sigma_err"),
    "success": ("success",),
    "cluster": ("cluster_bin",),
}


@dataclass
class TrackData:
    phi1: np.ndarray
    phi2: np.ndarray
    phi2_err: np.ndarray
    width: Optional[np.ndarray] = None
    width_err: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phi1 = np.asarray(self.phi1, dtype=float)
        self.phi2 = np.asarray(self.phi2, dtype=float)
        self.phi2_err = np.asarray(self.phi2_err, dtype=float)

        if self.width is not None:
            self.width = np.asarray(self.width, dtype=float)
        if self.width_err is not None:
            self.width_err = np.asarray(self.width_err, dtype=float)

        if not (self.phi1.size == self.phi2.size == self.phi2_err.size):
            raise ValueError("phi1, phi2, phi2_err must have the same length.")

        order = np.argsort(self.phi1)
        self.phi1 = self.phi1[order]
        self.phi2 = self.phi2[order]
        self.phi2_err = self.phi2_err[order]

        if self.width is not None:
            if self.width.size != self.phi1.size:
                raise ValueError("width must have the same length as phi1.")
            self.width = self.width[order]

        if self.width_err is not None:
            if self.width_err.size != self.phi1.size:
                raise ValueError("width_err must have the same length as phi1.")
            self.width_err = self.width_err[order]

    @property
    def n(self) -> int:
        return self.phi1.size


@dataclass
class ModelTrack:
    phi1: np.ndarray
    phi2: np.ndarray
    phi2_err: np.ndarray
    width: np.ndarray
    width_err: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


@dataclass
class MockStreamResult:
    params: Dict[str, float]
    stream_icrs: Any
    pal5_frame: Any
    model_track: ModelTrack


@dataclass
class StreamModelConfig:
    ra_deg: float = 229.018
    dec_deg: float = -0.124
    radial_velocity_kms: float = -58.7

    prog_mass_msun: float = 2.0e4
    plummer_b_pc: float = 4.0

    galcen_distance_kpc: float = 8.275
    v_sun_kms: tuple[float, float, float] = (8.4, 244.8, 8.4)

    dt_myr: float = 0.5
    n_steps: int = 6000
    release_every: int = 1
    n_particles: int = 2

    track_half_window_deg: float = 0.75
    min_particles_per_node: int = 16
    track_spline_k: int = 3
    smooth_model_track: bool = True

    track_jitter_deg: float = 0.03
    min_valid_fraction: float = 0.70
    use_width_term: bool = False
    width_jitter_deg: float = 0.03

    include_static_bar: bool = False
    bar_mass_msun: float = 1.0e10
    bar_a_kpc: float = 3.5
    bar_b_kpc: float = 0.5
    bar_c_kpc: float = 0.5
    bar_alpha_deg: float = 27.0


@dataclass
class SamplerConfig:
    free_names: Sequence[str] = (
        "log10_mhalo",
        "r_s",
        "q_z",
        "pm_ra_cosdec",
        "pm_dec",
        "distance",
    )

    fixed_params: Dict[str, float] = field(
        default_factory=lambda: {
            "q_y": 1.0,
            "prog_mass": 2.0e4,
        }
    )

    initial: Dict[str, float] = field(
        default_factory=lambda: {
            "log10_mhalo": 11.75,
            "r_s": 20.0,
            "q_z": 0.93,
            "pm_ra_cosdec": -2.296,
            "pm_dec": -2.257,
            "distance": 22.9,
            "q_y": 1.0,
            "prog_mass": 2.0e4,
        }
    )

    bounds: Dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "log10_mhalo": (11.0, 12.3),
            "r_s": (5.0, 50.0),
            "q_y": (0.7, 1.3),
            "q_z": (0.7, 1.2),
            "prog_mass": (5.0e3, 1.0e5),
            "pm_ra_cosdec": (-3.5, -1.0),
            "pm_dec": (-3.5, -1.0),
            "distance": (20.0, 24.5),
        }
    )

    init_scatter: Dict[str, float] = field(
        default_factory=lambda: {
            "log10_mhalo": 0.05,
            "r_s": 1.5,
            "q_y": 0.02,
            "q_z": 0.02,
            "prog_mass": 2.0e3,
            "pm_ra_cosdec": 0.03,
            "pm_dec": 0.03,
            "distance": 0.10,
        }
    )

    nwalkers: int = 16
    burnin_steps: int = 5
    production_steps: int = 5
    random_seed: int = 42


def _read_table_any(path: str | Path) -> Table:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return Table.read(path)


def _resolve_column(tab: Table, requested: Optional[str], role: str) -> Optional[str]:
    if requested:
        return requested if requested in tab.colnames else None
    for name in TRACK_COLUMN_ALIASES.get(role, ()):
        if name in tab.colnames:
            return name
    return None


def _parse_bool_array(values: Sequence[Any]) -> np.ndarray:
    out = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
            continue
        text = str(value).strip().lower()
        out.append(text in {"1", "true", "t", "yes", "y"})
    return np.asarray(out, dtype=bool)


def _table_col_to_numpy(tab: Table, colname: Optional[str], default: Optional[float], n: int) -> Optional[np.ndarray]:
    if colname is None:
        if default is None:
            return None
        return np.full(n, float(default), dtype=float)
    if colname not in tab.colnames:
        if default is None:
            return None
        return np.full(n, float(default), dtype=float)
    return np.asarray(tab[colname], dtype=float)


def load_observed_track(
    path: str | Path,
    phi1_col: Optional[str] = None,
    phi2_col: Optional[str] = None,
    phi2_err_col: Optional[str] = None,
    width_col: Optional[str] = None,
    width_err_col: Optional[str] = None,
    success_col: Optional[str] = None,
    default_phi2_err: float = 0.05,
    default_width_err: float = 0.05,
    require_success: bool = True,
) -> TrackData:
    tab = _read_table_any(path)

    phi1_col = _resolve_column(tab, phi1_col, "phi1")
    phi2_col = _resolve_column(tab, phi2_col, "phi2")
    phi2_err_col = _resolve_column(tab, phi2_err_col, "phi2_err")
    width_col = _resolve_column(tab, width_col, "width")
    width_err_col = _resolve_column(tab, width_err_col, "width_err")
    success_col = _resolve_column(tab, success_col, "success")

    if phi1_col is None or phi2_col is None:
        raise ValueError(
            "Could not resolve track columns. "
            f"Available columns: {tab.colnames}"
        )

    phi1 = np.asarray(tab[phi1_col], dtype=float)
    phi2 = np.asarray(tab[phi2_col], dtype=float)
    phi2_err = _table_col_to_numpy(tab, phi2_err_col, default_phi2_err, len(phi1))
    width = _table_col_to_numpy(tab, width_col, None, len(phi1))
    width_err = _table_col_to_numpy(tab, width_err_col, default_width_err if width_col else None, len(phi1))

    finite = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(phi2_err)
    if width is not None:
        finite &= np.isfinite(width)
    if width_err is not None:
        finite &= np.isfinite(width_err)

    if require_success and success_col is not None:
        finite &= _parse_bool_array(tab[success_col])

    return TrackData(
        phi1=phi1[finite],
        phi2=phi2[finite],
        phi2_err=phi2_err[finite],
        width=None if width is None else width[finite],
        width_err=None if width_err is None else width_err[finite],
        meta={
            "source": str(path),
            "phi1_col": phi1_col,
            "phi2_col": phi2_col,
            "phi2_err_col": phi2_err_col,
            "width_col": width_col,
            "width_err_col": width_err_col,
            "success_col": success_col,
            "require_success": require_success,
        },
    )


def theta_to_params(theta: Sequence[float], sampler_cfg: SamplerConfig) -> Dict[str, float]:
    if len(theta) != len(sampler_cfg.free_names):
        raise ValueError("theta length does not match free_names.")
    params = dict(sampler_cfg.fixed_params)
    for name, value in zip(sampler_cfg.free_names, theta):
        params[name] = float(value)
    return params


def params_in_bounds(params: Dict[str, float], bounds: Dict[str, tuple[float, float]]) -> bool:
    for name, (low, high) in bounds.items():
        if name not in params:
            continue
        value = params[name]
        if not (low < value < high):
            return False
    return True


def make_initial_state(sampler_cfg: SamplerConfig) -> np.ndarray:
    rng = np.random.default_rng(sampler_cfg.random_seed)
    ndim = len(sampler_cfg.free_names)
    nwalkers = max(sampler_cfg.nwalkers, 4 * ndim)
    pos = np.empty((nwalkers, ndim), dtype=float)

    for i in range(nwalkers):
        trial = []
        for name in sampler_cfg.free_names:
            center = sampler_cfg.initial[name]
            scatter = sampler_cfg.init_scatter[name]
            low, high = sampler_cfg.bounds[name]
            for _ in range(1000):
                value = rng.normal(center, scatter)
                if low < value < high:
                    trial.append(value)
                    break
            else:
                trial.append(np.clip(center, low + 1e-6, high - 1e-6))
        pos[i] = trial

    return pos


def _import_astropy_coords() -> tuple[Any, Any]:
    import astropy.coordinates as coord
    import astropy.units as u
    _ = coord.galactocentric_frame_defaults.set("v4.0")
    return coord, u


def _import_gala() -> tuple[Any, Any, Any, Any, Any]:
    import gala.coordinates as gc
    import gala.dynamics as gd
    from gala.dynamics import mockstream as ms
    import gala.potential as gp
    from gala.units import galactic
    return gc, gd, ms, gp, galactic


def _import_pandas() -> Any:
    import pandas as pd
    return pd


def _import_interpolated_spline() -> Any:
    from scipy.interpolate import InterpolatedUnivariateSpline
    return InterpolatedUnivariateSpline


def build_galcen_frame(model_cfg: StreamModelConfig) -> Any:
    coord, u = _import_astropy_coords()
    v_sun = coord.CartesianDifferential(np.array(model_cfg.v_sun_kms) * u.km / u.s)
    return coord.Galactocentric(
        galcen_distance=model_cfg.galcen_distance_kpc * u.kpc,
        galcen_v_sun=v_sun,
    )


def build_potential(params: Dict[str, float], model_cfg: StreamModelConfig) -> Any:
    _, _, _, gp, galactic = _import_gala()
    _, u = _import_astropy_coords()
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

    pot = gp.CCompositePotential(disk=disk, bulge=bulge, halo=halo)
    if model_cfg.include_static_bar:
        pot["bar"] = gp.LongMuraliBarPotential(
            m=model_cfg.bar_mass_msun * u.Msun,
            a=model_cfg.bar_a_kpc * u.kpc,
            b=model_cfg.bar_b_kpc * u.kpc,
            c=model_cfg.bar_c_kpc * u.kpc,
            alpha=model_cfg.bar_alpha_deg * u.deg,
            units=galactic,
        )
    return gp.Hamiltonian(pot)


def build_progenitor_coord(params: Dict[str, float], model_cfg: StreamModelConfig) -> Any:
    coord, u = _import_astropy_coords()
    return coord.ICRS(
        ra=model_cfg.ra_deg * u.deg,
        dec=model_cfg.dec_deg * u.deg,
        distance=params["distance"] * u.kpc,
        pm_ra_cosdec=params["pm_ra_cosdec"] * u.mas / u.yr,
        pm_dec=params["pm_dec"] * u.mas / u.yr,
        radial_velocity=model_cfg.radial_velocity_kms * u.km / u.s,
    )


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    q25, q75 = np.nanpercentile(values, [25, 75])
    return 0.7413 * (q75 - q25)


def extract_model_track(
    phi1_particles: np.ndarray,
    phi2_particles: np.ndarray,
    observed_phi1_nodes: np.ndarray,
    model_cfg: StreamModelConfig,
) -> ModelTrack:
    phi1_particles = np.asarray(phi1_particles, dtype=float)
    phi2_particles = np.asarray(phi2_particles, dtype=float)
    nodes = np.asarray(observed_phi1_nodes, dtype=float)

    mu = np.full(nodes.size, np.nan, dtype=float)
    mu_err = np.full(nodes.size, np.nan, dtype=float)
    width = np.full(nodes.size, np.nan, dtype=float)
    width_err = np.full(nodes.size, np.nan, dtype=float)
    counts = np.zeros(nodes.size, dtype=int)

    for i, center in enumerate(nodes):
        mask = np.abs(phi1_particles - center) <= model_cfg.track_half_window_deg
        counts[i] = int(np.sum(mask))
        if counts[i] < model_cfg.min_particles_per_node:
            continue

        sample = phi2_particles[mask]
        mu[i] = np.nanmedian(sample)
        width[i] = robust_sigma(sample)
        mu_err[i] = width[i] / np.sqrt(max(counts[i], 1))
        width_err[i] = width[i] / np.sqrt(2 * max(counts[i] - 1, 1))

    valid = np.isfinite(mu)
    if model_cfg.smooth_model_track and np.sum(valid) >= max(4, model_cfg.track_spline_k + 1):
        InterpolatedUnivariateSpline = _import_interpolated_spline()
        spline_k = min(model_cfg.track_spline_k, int(np.sum(valid) - 1))
        track_spline = InterpolatedUnivariateSpline(nodes[valid], mu[valid], k=spline_k)
        inside = (nodes >= np.nanmin(nodes[valid])) & (nodes <= np.nanmax(nodes[valid]))
        mu[inside] = track_spline(nodes[inside])

    return ModelTrack(
        phi1=nodes,
        phi2=mu,
        phi2_err=mu_err,
        width=width,
        width_err=width_err,
        counts=counts,
        valid=np.isfinite(mu),
    )


def generate_mock_stream(
    params: Dict[str, float],
    observed: TrackData,
    model_cfg: StreamModelConfig,
) -> MockStreamResult:
    coord, u = _import_astropy_coords()
    gc, gd, ms, gp, galactic = _import_gala()
    ham = build_potential(params, model_cfg)
    galcen_frame = build_galcen_frame(model_cfg)

    prog_mass = params.get("prog_mass", model_cfg.prog_mass_msun)
    prog_potential = gp.PlummerPotential(
        m=prog_mass * u.Msun,
        b=model_cfg.plummer_b_pc * u.pc,
        units=galactic,
    )

    progenitor = build_progenitor_coord(params, model_cfg)
    prog_w0 = gd.PhaseSpacePosition(progenitor.transform_to(galcen_frame).cartesian)

    df = ms.FardalStreamDF()
    gen = ms.MockStreamGenerator(df, ham, progenitor_potential=prog_potential)

    stream_w, _ = gen.run(
        prog_w0,
        prog_mass * u.Msun,
        dt=-model_cfg.dt_myr * u.Myr,
        n_steps=model_cfg.n_steps,
        release_every=model_cfg.release_every,
        n_particles=model_cfg.n_particles,
        progress=False,
    )

    stream_icrs = stream_w.to_coord_frame(coord.ICRS(), galactocentric_frame=galcen_frame)
    pal5_frame = stream_icrs.transform_to(gc.Pal5PriceWhelan18())
    phi1 = pal5_frame.phi1.wrap_at(180 * u.deg).degree
    phi2 = pal5_frame.phi2.degree
    model_track = extract_model_track(phi1, phi2, observed.phi1, model_cfg)

    return MockStreamResult(
        params=params,
        stream_icrs=stream_icrs,
        pal5_frame=pal5_frame,
        model_track=model_track,
    )


def track_log_likelihood(observed: TrackData, model_track: ModelTrack, model_cfg: StreamModelConfig) -> float:
    valid = np.isfinite(observed.phi2) & np.isfinite(observed.phi2_err) & model_track.valid
    min_valid = max(5, int(np.ceil(model_cfg.min_valid_fraction * observed.n)))
    if int(np.sum(valid)) < min_valid:
        return -np.inf

    sigma = np.sqrt(
        observed.phi2_err[valid] ** 2
        + np.nan_to_num(model_track.phi2_err[valid], nan=0.0) ** 2
        + model_cfg.track_jitter_deg ** 2
    )
    delta = model_track.phi2[valid] - observed.phi2[valid]
    logl = -0.5 * np.sum((delta / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

    if (
        model_cfg.use_width_term
        and observed.width is not None
        and observed.width_err is not None
        and model_track.width is not None
    ):
        valid_w = valid & np.isfinite(observed.width) & np.isfinite(observed.width_err) & np.isfinite(model_track.width)
        if np.any(valid_w):
            sigma_w = np.sqrt(
                observed.width_err[valid_w] ** 2
                + np.nan_to_num(model_track.width_err[valid_w], nan=0.0) ** 2
                + model_cfg.width_jitter_deg ** 2
            )
            delta_w = model_track.width[valid_w] - observed.width[valid_w]
            logl += -0.5 * np.sum((delta_w / sigma_w) ** 2 + np.log(2 * np.pi * sigma_w**2))

    return float(logl)


def log_prior(theta: Sequence[float], sampler_cfg: SamplerConfig) -> float:
    params = theta_to_params(theta, sampler_cfg)
    return 0.0 if params_in_bounds(params, sampler_cfg.bounds) else -np.inf


def log_probability(
    theta: Sequence[float],
    observed: TrackData,
    model_cfg: StreamModelConfig,
    sampler_cfg: SamplerConfig,
) -> float:
    lp = log_prior(theta, sampler_cfg)
    if not np.isfinite(lp):
        return -np.inf

    params = theta_to_params(theta, sampler_cfg)
    try:
        mock = generate_mock_stream(params, observed, model_cfg)
        ll = track_log_likelihood(observed, mock.model_track, model_cfg)
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def summarize_samples(samples: np.ndarray, free_names: Sequence[str]) -> Any:
    pd = _import_pandas()
    records = []
    for i, name in enumerate(free_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        records.append(
            {
                "parameter": name,
                "q16": q16,
                "q50": q50,
                "q84": q84,
                "minus": q50 - q16,
                "plus": q84 - q50,
            }
        )
    return pd.DataFrame.from_records(records)


def run_mcmc(
    observed: TrackData,
    model_cfg: StreamModelConfig,
    sampler_cfg: SamplerConfig,
    outdir: str | Path,
    ncores: int = 1,
    mp_start_method: str = "spawn",
) -> tuple[Any, np.ndarray, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pos0 = make_initial_state(sampler_cfg)
    nwalkers, ndim = pos0.shape
    import emcee
    pd = _import_pandas()

    effective_ncores = max(1, int(ncores))

    pool = None
    if effective_ncores > 1:
        ctx = mp.get_context(mp_start_method)
        print(
            f"[info] enabling multiprocessing pool: "
            f"ncores={min(effective_ncores, nwalkers)} start_method={mp_start_method}"
        )
        pool = ctx.Pool(processes=min(effective_ncores, nwalkers))

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(observed, model_cfg, sampler_cfg),
            pool=pool,
        )

        state = sampler.run_mcmc(pos0, sampler_cfg.burnin_steps, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, sampler_cfg.production_steps, progress=False)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    flat_samples = sampler.get_chain(flat=True)
    samples_df = pd.DataFrame(flat_samples, columns=list(sampler_cfg.free_names))
    summary_df = summarize_samples(flat_samples, sampler_cfg.free_names)

    samples_df.to_csv(outdir / "mcmc_samples.csv", index=False)
    summary_df.to_csv(outdir / "mcmc_summary.csv", index=False)
    np.save(outdir / "chain.npy", sampler.get_chain())
    np.save(outdir / "log_prob.npy", sampler.get_log_prob())

    return sampler, flat_samples, summary_df


def profile_initial_log_probability(
    observed: TrackData,
    model_cfg: StreamModelConfig,
    sampler_cfg: SamplerConfig,
) -> float:
    import time

    theta0 = make_initial_state(sampler_cfg)[0]
    t0 = time.time()
    logp0 = log_probability(theta0, observed, model_cfg, sampler_cfg)
    dt = time.time() - t0
    total_steps = sampler_cfg.burnin_steps + sampler_cfg.production_steps
    rough_total_evals = sampler_cfg.nwalkers * total_steps
    rough_serial_hours = (dt * rough_total_evals) / 3600.0
    print(
        "[timing] initial log_probability "
        f"dt={dt:.3f}s logp={logp0:.3f} "
        f"rough_total_evals~{rough_total_evals} "
        f"rough_serial_walltime~{rough_serial_hours:.2f} hr"
    )
    return dt


def best_fit_params_from_samples(samples: np.ndarray, sampler_cfg: SamplerConfig) -> Dict[str, float]:
    med = np.median(samples, axis=0)
    return theta_to_params(med, sampler_cfg)


def save_best_fit_products(
    observed: TrackData,
    params: Dict[str, float],
    model_cfg: StreamModelConfig,
    outdir: str | Path,
) -> MockStreamResult:
    pd = _import_pandas()
    _, u = _import_astropy_coords()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mock = generate_mock_stream(params, observed, model_cfg)

    model_track_tab = Table()
    model_track_tab["phi1"] = mock.model_track.phi1
    model_track_tab["phi2_model"] = mock.model_track.phi2
    model_track_tab["phi2_model_err"] = mock.model_track.phi2_err
    model_track_tab["width_model"] = mock.model_track.width
    model_track_tab["width_model_err"] = mock.model_track.width_err
    model_track_tab["counts"] = mock.model_track.counts
    model_track_tab["valid"] = mock.model_track.valid.astype(int)
    model_track_tab.write(outdir / "best_fit_model_track.fits", overwrite=True)

    stream_tab = Table()
    stream_tab["ra"] = mock.stream_icrs.ra.degree
    stream_tab["dec"] = mock.stream_icrs.dec.degree
    stream_tab["phi1"] = mock.pal5_frame.phi1.wrap_at(180 * u.deg).degree
    stream_tab["phi2"] = mock.pal5_frame.phi2.degree
    stream_tab.write(outdir / "best_fit_mock_stream_particles.fits", overwrite=True)

    pd.DataFrame([params]).to_csv(outdir / "best_fit_params.csv", index=False)
    return mock


def run_visualization_suite(args: argparse.Namespace, outdir: Path) -> None:
    script_path = Path(args.visualize_script)
    if not script_path.is_absolute():
        script_path = Path(__file__).resolve().parent / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Visualization script not found: {script_path}")

    plots_outdir = Path(args.plots_outdir) if args.plots_outdir else outdir / "pal5_plots"
    cmd = [
        sys.executable,
        str(script_path),
        "--run-dir",
        str(outdir),
        "--track-file",
        str(outdir / "observed_track_used.fits"),
        "--core-script",
        str(Path(__file__).resolve()),
        "--outdir",
        str(plots_outdir),
    ]

    if args.plot_star_file:
        cmd.extend(["--star-file", args.plot_star_file])
        cmd.extend(["--star-ra-col", args.plot_star_ra_col])
        cmd.extend(["--star-dec-col", args.plot_star_dec_col])
        if args.plot_star_distance_col:
            cmd.extend(["--star-distance-col", args.plot_star_distance_col])
        if args.plot_star_max_distance is not None:
            cmd.extend(["--star-max-distance", str(args.plot_star_max_distance)])

    if args.plot_skip_rv:
        cmd.append("--skip-rv")
    if args.plot_skip_rv_grids:
        cmd.append("--skip-rv-grids")
    if args.plot_skip_orbit_grid:
        cmd.append("--skip-orbit-grid")
    if args.plot_skip_literature:
        cmd.append("--skip-literature")

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pal 5 mock-stream global track fit")
    p.add_argument("--track", required=True, help="Observed track table path")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--phi1-col", default=None, help="Override phi1 column name")
    p.add_argument("--phi2-col", default=None, help="Override phi2 column name")
    p.add_argument("--phi2-err-col", default=None, help="Override phi2_err column name")
    p.add_argument("--width-col", default=None, help="Override width column name")
    p.add_argument("--width-err-col", default=None, help="Override width_err column name")
    p.add_argument("--success-col", default=None, help="Override success column name")
    p.add_argument("--allow-failed", action="store_true", help="Do not filter out failed rows if a success column exists")
    p.add_argument("--default-phi2-err", type=float, default=0.05)
    p.add_argument("--default-width-err", type=float, default=0.05)
    p.add_argument("--use-width-term", action="store_true", help="Include width term in the likelihood")
    p.add_argument("--ncores", type=int, default=1, help="Number of worker processes for parallel likelihood evaluation")
    p.add_argument("--nwalkers", type=int, default=16)
    p.add_argument("--burnin", type=int, default=5)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--dt-myr", type=float, default=0.5)
    p.add_argument("--n-stream-steps", type=int, default=3000)
    p.add_argument("--release-every", type=int, default=1)
    p.add_argument("--n-particles", type=int, default=2)
    p.add_argument("--min-particles-per-node", type=int, default=16)
    p.add_argument("--min-valid-fraction", type=float, default=0.30)
    p.add_argument("--mp-start-method", choices=("spawn", "fork", "forkserver"), default="spawn")
    p.add_argument("--track-half-window-deg", type=float, default=0.75)
    p.add_argument("--track-jitter-deg", type=float, default=0.03)
    p.add_argument("--width-jitter-deg", type=float, default=0.03)
    p.add_argument("--profile-initial-logp", action="store_true", help="Time one initial log_probability call and print a rough serial walltime estimate before sampling")
    p.add_argument("--profile-only", action="store_true", help="Run the initial log_probability timing probe and exit without running MCMC")
    p.add_argument("--include-static-bar", action="store_true")
    p.add_argument("--log10-mhalo-init", type=float, default=11.75)
    p.add_argument("--r-s-init", type=float, default=20.0)
    p.add_argument("--q-z-init", type=float, default=0.93)
    p.add_argument("--pm-ra-cosdec-init", type=float, default=-2.296)
    p.add_argument("--pm-dec-init", type=float, default=-2.257)
    p.add_argument("--distance-init", type=float, default=22.9)
    p.add_argument("--prog-mass", type=float, default=2.0e4)
    p.add_argument("--q-y", type=float, default=1.0)
    p.add_argument("--make-plots", action="store_true", help="Run pal5_visualize_suite.py after best-fit products are written")
    p.add_argument("--visualize-script", default="pal5_visualize_suite.py", help="Path to the visualization suite script")
    p.add_argument("--plots-outdir", default=None, help="Visualization output directory; defaults to <outdir>/pal5_plots")
    p.add_argument("--plot-star-file", default=None, help="Optional filtered star catalog for visualization background")
    p.add_argument("--plot-star-ra-col", default="RA")
    p.add_argument("--plot-star-dec-col", default="DEC")
    p.add_argument("--plot-star-distance-col", default=None)
    p.add_argument("--plot-star-max-distance", type=float, default=None)
    p.add_argument("--plot-skip-rv", action="store_true", help="Skip all RV figures in the visualization suite")
    p.add_argument("--plot-skip-rv-grids", action="store_true", help="Skip RV distance-grid figures in the visualization suite")
    p.add_argument("--plot-skip-orbit-grid", action="store_true", help="Skip orbit distance-grid figures in the visualization suite")
    p.add_argument("--plot-skip-literature", action="store_true", help="Skip q_z literature comparison in the visualization suite")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    observed = load_observed_track(
        args.track,
        phi1_col=args.phi1_col,
        phi2_col=args.phi2_col,
        phi2_err_col=args.phi2_err_col,
        width_col=args.width_col,
        width_err_col=args.width_err_col,
        success_col=args.success_col,
        default_phi2_err=args.default_phi2_err,
        default_width_err=args.default_width_err,
        require_success=not args.allow_failed,
    )

    model_cfg = StreamModelConfig(
        dt_myr=args.dt_myr,
        n_steps=args.n_stream_steps,
        release_every=args.release_every,
        n_particles=args.n_particles,
        min_particles_per_node=args.min_particles_per_node,
        min_valid_fraction=args.min_valid_fraction,
        track_half_window_deg=args.track_half_window_deg,
        track_jitter_deg=args.track_jitter_deg,
        width_jitter_deg=args.width_jitter_deg,
        use_width_term=args.use_width_term,
        include_static_bar=args.include_static_bar,
        prog_mass_msun=args.prog_mass,
    )

    sampler_cfg = SamplerConfig(
        fixed_params={
            "q_y": args.q_y,
            "prog_mass": args.prog_mass,
        },
        initial={
            "log10_mhalo": args.log10_mhalo_init,
            "r_s": args.r_s_init,
            "q_z": args.q_z_init,
            "pm_ra_cosdec": args.pm_ra_cosdec_init,
            "pm_dec": args.pm_dec_init,
            "distance": args.distance_init,
            "q_y": args.q_y,
            "prog_mass": args.prog_mass,
        },
        nwalkers=args.nwalkers,
        burnin_steps=args.burnin,
        production_steps=args.steps,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    observed_tab = Table()
    observed_tab["phi1"] = observed.phi1
    observed_tab["phi2"] = observed.phi2
    observed_tab["phi2_err"] = observed.phi2_err
    if observed.width is not None:
        observed_tab["width"] = observed.width
    if observed.width_err is not None:
        observed_tab["width_err"] = observed.width_err
    observed_tab.meta.update(observed.meta)
    observed_tab.write(outdir / "observed_track_used.fits", overwrite=True)

    if args.profile_initial_logp:
        profile_initial_log_probability(observed, model_cfg, sampler_cfg)
        if args.profile_only:
            return

    _, samples, summary = run_mcmc(
        observed,
        model_cfg,
        sampler_cfg,
        outdir,
        ncores=max(1, args.ncores),
        mp_start_method=args.mp_start_method,
    )
    print(summary.to_string(index=False))

    best_params = best_fit_params_from_samples(samples, sampler_cfg)
    save_best_fit_products(observed, best_params, model_cfg, outdir)

    if args.make_plots:
        run_visualization_suite(args, outdir)


if __name__ == "__main__":
    main()
