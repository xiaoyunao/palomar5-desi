#!/usr/bin/env python3
"""
Step 3c: Bonaca-style comparison / figure-making for Pal 5.

This script does not re-fit the stream. It takes the outputs of step 3b and
turns them into:

1. A clean Bonaca-style profile figure (track, track residual, density, width)
2. A local phi1-phi2 density map with the fitted track overplotted
3. A MAP-vs-emcee comparison figure (if emcee products are provided)
4. A trailing-vs-leading asymmetry figure
5. Machine-readable and human-readable summary tables / reports

The baseline intent is:
- control + MAP = main morphology baseline
- control + emcee = posterior / uncertainty sanity check
- control_times_depth = diagnostic only, not default science baseline

The script is flexible and can run with only the control+MAP products.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from astropy.table import Table
except Exception:  # pragma: no cover
    Table = None


BONACA_REF = {
    "g_mag_range": "20 < g0 < 23",
    "total_stream_stars_excluding_progenitor": 3000.0,
    "total_stream_stars_excluding_progenitor_err": 100.0,
    "extent_trailing_deg": 16.0,
    "extent_leading_deg": 8.0,
    "trailing_to_leading_ratio_within_5deg": 1.5,
    "cluster_width_deg": 0.15,
    "leading_fan_width_deg": 0.40,
    "leading_fan_phi1_deg": 7.0,
    "prominent_gap_phi1_deg": [-7.0, 3.0],
    "track_wiggle_phi1_deg": [-13.0, -4.5, 3.5],
}


@dataclass
class ProfileMetrics:
    label: str
    n_bins: int
    n_success: int
    n_success_excluding_cluster: int
    trailing_extent_deg: float
    leading_extent_deg: float
    integrated_total_abs8: float
    integrated_trailing_abs8: float
    integrated_leading_abs8: float
    trailing_to_leading_abs8: float
    integrated_total_abs5: float
    integrated_trailing_abs5: float
    integrated_leading_abs5: float
    trailing_to_leading_abs5: float
    near_cluster_width_deg: float
    leading_width_max_5to8_deg: float
    leading_width_max_5to8_phi1: float
    trailing_width_max_m15to5_deg: float
    trailing_width_max_m15to5_phi1: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3c Bonaca-style comparison / figure-making")
    p.add_argument("--profiles-map", default="pal5_step3b_profiles_control.csv",
                   help="Baseline profile table, typically control+MAP")
    p.add_argument("--summary-map", default="pal5_step3b_summary_control.json",
                   help="Summary JSON for the baseline profile table")
    p.add_argument("--label-map", default="control + MAP",
                   help="Legend label for the baseline run")

    p.add_argument("--profiles-alt", default="pal5_step3b_profiles_emcee.csv",
                   help="Optional alternate profile table, typically control+emcee")
    p.add_argument("--summary-alt", default="pal5_step3b_summary_emcee.json",
                   help="Optional summary JSON for the alternate run")
    p.add_argument("--label-alt", default="control + emcee",
                   help="Legend label for the alternate run")
    p.add_argument("--no-alt", action="store_true",
                   help="Ignore the alternate run even if files exist")

    p.add_argument("--strict-fits", default="step2_outputs/pal5_step2_strict_members.fits",
                   help="Strict member catalog from step 2, used for map figures")
    p.add_argument("--no-map", action="store_true",
                   help="Skip re-making the local phi1-phi2 density map")

    p.add_argument("--outdir", default="step3c_outputs",
                   help="Output directory")
    p.add_argument("--bin-phi1", type=float, default=0.25,
                   help="Bin size for local phi1 map")
    p.add_argument("--bin-phi2", type=float, default=0.05,
                   help="Bin size for local phi2 map")
    p.add_argument("--phi2-min", type=float, default=-2.5)
    p.add_argument("--phi2-max", type=float, default=2.5)
    p.add_argument("--compare-limit", type=float, default=8.0,
                   help="Bonaca-like integrated comparison limit in |phi1|")
    p.add_argument("--inner-limit", type=float, default=5.0,
                   help="Inner comparison limit in |phi1|")
    p.add_argument("--cluster-width-window", type=float, default=1.5,
                   help="Use successful non-cluster bins with |phi1| <= this value to summarize near-cluster width")
    p.add_argument("--leading-width-min", type=float, default=5.0,
                   help="Minimum phi1 used to summarize leading fan width")
    p.add_argument("--leading-width-max", type=float, default=8.0,
                   help="Maximum phi1 used to summarize leading fan width")
    p.add_argument("--trailing-width-min", type=float, default=-15.0,
                   help="Minimum phi1 used to summarize trailing outer width")
    p.add_argument("--trailing-width-max", type=float, default=-5.0,
                   help="Maximum phi1 used to summarize trailing outer width")
    return p.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize booleans just in case
    for col in ["cluster_bin", "success", "optimizer_success", "sampler_success"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def infer_phi1_step(df: pd.DataFrame, summary: Dict) -> float:
    if "phi1_step" in summary:
        return float(summary["phi1_step"])
    vals = np.sort(df["phi1_center"].to_numpy(dtype=float))
    d = np.diff(vals)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        raise ValueError("Could not infer phi1_step from profile table")
    return float(np.median(d))


def make_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    success = df["success"].to_numpy(dtype=bool)
    cluster = df["cluster_bin"].to_numpy(dtype=bool)
    usable = success & (~cluster)
    phi1 = df["phi1_center"].to_numpy(dtype=float)
    return {
        "success": success,
        "cluster": cluster,
        "usable": usable,
        "trailing": usable & (phi1 < 0),
        "leading": usable & (phi1 > 0),
    }


def integrated_linear_density(df: pd.DataFrame, mask: np.ndarray, phi1_step: float) -> float:
    x = df.loc[mask, "linear_density"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sum(x * phi1_step))


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def summarize_metrics(
    df: pd.DataFrame,
    summary: Dict,
    label: str,
    compare_limit: float,
    inner_limit: float,
    cluster_width_window: float,
    leading_width_min: float,
    leading_width_max: float,
    trailing_width_min: float,
    trailing_width_max: float,
) -> ProfileMetrics:
    masks = make_masks(df)
    phi1 = df["phi1_center"].to_numpy(dtype=float)
    phi1_step = infer_phi1_step(df, summary)

    usable = masks["usable"]
    trailing = masks["trailing"]
    leading = masks["leading"]

    abs8 = usable & (np.abs(phi1) <= compare_limit)
    abs5 = usable & (np.abs(phi1) <= inner_limit)
    tr8 = abs8 & (phi1 < 0)
    le8 = abs8 & (phi1 > 0)
    tr5 = abs5 & (phi1 < 0)
    le5 = abs5 & (phi1 > 0)

    trailing_extent = float(np.abs(np.min(phi1[trailing]))) if np.any(trailing) else float("nan")
    leading_extent = float(np.max(phi1[leading])) if np.any(leading) else float("nan")

    near_cluster = usable & (np.abs(phi1) <= cluster_width_window)
    near_cluster_width = float(np.nanmedian(df.loc[near_cluster, "sigma"].to_numpy(dtype=float))) if np.any(near_cluster) else float("nan")

    lead_fan = usable & (phi1 >= leading_width_min) & (phi1 <= leading_width_max)
    trail_outer = usable & (phi1 >= trailing_width_min) & (phi1 <= trailing_width_max)

    if np.any(lead_fan):
        lead_idx_local = np.nanargmax(df.loc[lead_fan, "sigma"].to_numpy(dtype=float))
        lead_rows = df.loc[lead_fan, ["phi1_center", "sigma"]].reset_index(drop=True)
        leading_width_max = float(lead_rows.loc[lead_idx_local, "sigma"])
        leading_width_phi1 = float(lead_rows.loc[lead_idx_local, "phi1_center"])
    else:
        leading_width_max = float("nan")
        leading_width_phi1 = float("nan")

    if np.any(trail_outer):
        tr_idx_local = np.nanargmax(df.loc[trail_outer, "sigma"].to_numpy(dtype=float))
        tr_rows = df.loc[trail_outer, ["phi1_center", "sigma"]].reset_index(drop=True)
        trailing_width_max = float(tr_rows.loc[tr_idx_local, "sigma"])
        trailing_width_phi1 = float(tr_rows.loc[tr_idx_local, "phi1_center"])
    else:
        trailing_width_max = float("nan")
        trailing_width_phi1 = float("nan")

    return ProfileMetrics(
        label=label,
        n_bins=int(summary.get("n_bins", len(df))),
        n_success=int(np.sum(df["success"].to_numpy(dtype=bool))),
        n_success_excluding_cluster=int(np.sum(usable)),
        trailing_extent_deg=trailing_extent,
        leading_extent_deg=leading_extent,
        integrated_total_abs8=integrated_linear_density(df, abs8, phi1_step),
        integrated_trailing_abs8=integrated_linear_density(df, tr8, phi1_step),
        integrated_leading_abs8=integrated_linear_density(df, le8, phi1_step),
        trailing_to_leading_abs8=safe_ratio(
            integrated_linear_density(df, tr8, phi1_step),
            integrated_linear_density(df, le8, phi1_step),
        ),
        integrated_total_abs5=integrated_linear_density(df, abs5, phi1_step),
        integrated_trailing_abs5=integrated_linear_density(df, tr5, phi1_step),
        integrated_leading_abs5=integrated_linear_density(df, le5, phi1_step),
        trailing_to_leading_abs5=safe_ratio(
            integrated_linear_density(df, tr5, phi1_step),
            integrated_linear_density(df, le5, phi1_step),
        ),
        near_cluster_width_deg=near_cluster_width,
        leading_width_max_5to8_deg=leading_width_max,
        leading_width_max_5to8_phi1=leading_width_phi1,
        trailing_width_max_m15to5_deg=trailing_width_max,
        trailing_width_max_m15to5_phi1=trailing_width_phi1,
    )


def merge_runs(df_map: pd.DataFrame, df_alt: Optional[pd.DataFrame]) -> pd.DataFrame:
    keep_cols = [
        "phi1_center", "cluster_bin", "success", "mu", "mu_err", "sigma", "sigma_err",
        "linear_density", "linear_density_err", "peak_surface_density", "peak_surface_density_err",
        "track_poly", "track_resid", "f_stream", "f_stream_err", "n_stream", "n_stream_err",
    ]
    base = df_map[keep_cols].copy()
    base.columns = [c if c == "phi1_center" else f"map_{c}" for c in base.columns]
    base = base.rename(columns={"phi1_center": "phi1_center"})
    if df_alt is not None:
        alt = df_alt[keep_cols].copy()
        alt.columns = [c if c == "phi1_center" else f"alt_{c}" for c in alt.columns]
        alt = alt.rename(columns={"phi1_center": "phi1_center"})
        base = base.merge(alt, on="phi1_center", how="left")
    return base.sort_values("phi1_center").reset_index(drop=True)


def write_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_markdown_report(
    path: Path,
    metrics_map: ProfileMetrics,
    metrics_alt: Optional[ProfileMetrics],
    summary_map: Dict,
    summary_alt: Optional[Dict],
    merged: pd.DataFrame,
    compare_limit: float,
    inner_limit: float,
) -> None:
    def fmt(x: float, nd: int = 3) -> str:
        if x is None or (isinstance(x, float) and (not np.isfinite(x))):
            return "nan"
        return f"{x:.{nd}f}"

    lines = []
    lines.append("# Pal 5 step 3c: Bonaca-style comparison report")
    lines.append("")
    lines.append("This report compares the current Pal 5 strict-sample morphology baseline to the")
    lines.append("headline reference values described in Bonaca+2020 Figure 3 / Section 3.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Baseline run: **{metrics_map.label}**")
    lines.append(f"- eta_mode: `{summary_map.get('eta_mode', 'unknown')}`")
    lines.append(f"- sampler: `{summary_map.get('sampler', 'unknown')}`")
    lines.append(f"- Strict sample size: {summary_map.get('n_input_signal', 'unknown')}")
    if metrics_alt is not None and summary_alt is not None:
        lines.append(f"- Alternate run: **{metrics_alt.label}**")
        lines.append(f"- Alternate eta_mode: `{summary_alt.get('eta_mode', 'unknown')}`")
        lines.append(f"- Alternate sampler: `{summary_alt.get('sampler', 'unknown')}`")
    lines.append("")
    lines.append("## Bonaca reference values used for quick comparison")
    lines.append("")
    lines.append(f"- Magnitude range: {BONACA_REF['g_mag_range']}")
    lines.append(f"- Total stream stars excluding progenitor: {BONACA_REF['total_stream_stars_excluding_progenitor']:.0f} ± {BONACA_REF['total_stream_stars_excluding_progenitor_err']:.0f}")
    lines.append(f"- Stream extent: trailing ≈ {BONACA_REF['extent_trailing_deg']:.0f} deg, leading ≈ {BONACA_REF['extent_leading_deg']:.0f} deg")
    lines.append(f"- Trailing / leading within {inner_limit:.0f} deg: ≈ {BONACA_REF['trailing_to_leading_ratio_within_5deg']:.1f}")
    lines.append(f"- Near cluster width: ≈ {BONACA_REF['cluster_width_deg']:.2f} deg")
    lines.append(f"- Leading fan width: ≈ {BONACA_REF['leading_fan_width_deg']:.2f} deg at phi1 ≈ {BONACA_REF['leading_fan_phi1_deg']:.0f} deg")
    lines.append("")

    lines.append("## Derived metrics")
    lines.append("")
    header = "| metric | baseline | alternate | Bonaca ref |"
    sep = "|---|---:|---:|---:|"
    lines += [header, sep]
    alt = metrics_alt
    lines.append(f"| successful bins | {metrics_map.n_success} | {alt.n_success if alt else ''} | 41 bins modeled |")
    lines.append(f"| successful bins excluding cluster | {metrics_map.n_success_excluding_cluster} | {alt.n_success_excluding_cluster if alt else ''} | -- |")
    lines.append(f"| trailing extent [deg] | {fmt(metrics_map.trailing_extent_deg, 2)} | {fmt(alt.trailing_extent_deg, 2) if alt else ''} | {BONACA_REF['extent_trailing_deg']:.1f} |")
    lines.append(f"| leading extent [deg] | {fmt(metrics_map.leading_extent_deg, 2)} | {fmt(alt.leading_extent_deg, 2) if alt else ''} | {BONACA_REF['extent_leading_deg']:.1f} |")
    lines.append(f"| integrated stars, |phi1| < {compare_limit:.0f} | {fmt(metrics_map.integrated_total_abs8, 0)} | {fmt(alt.integrated_total_abs8, 0) if alt else ''} | {BONACA_REF['total_stream_stars_excluding_progenitor']:.0f} |")
    lines.append(f"| trailing / leading, |phi1| < {compare_limit:.0f} | {fmt(metrics_map.trailing_to_leading_abs8, 2)} | {fmt(alt.trailing_to_leading_abs8, 2) if alt else ''} | ~1.0 |")
    lines.append(f"| trailing / leading, |phi1| < {inner_limit:.0f} | {fmt(metrics_map.trailing_to_leading_abs5, 2)} | {fmt(alt.trailing_to_leading_abs5, 2) if alt else ''} | {BONACA_REF['trailing_to_leading_ratio_within_5deg']:.2f} |")
    lines.append(f"| near-cluster width [deg] | {fmt(metrics_map.near_cluster_width_deg, 3)} | {fmt(alt.near_cluster_width_deg, 3) if alt else ''} | {BONACA_REF['cluster_width_deg']:.3f} |")
    lines.append(f"| leading max width in [5, 8] [deg] | {fmt(metrics_map.leading_width_max_5to8_deg, 3)} @ {fmt(metrics_map.leading_width_max_5to8_phi1, 2)} | {f'{fmt(alt.leading_width_max_5to8_deg, 3)} @ {fmt(alt.leading_width_max_5to8_phi1, 2)}' if alt else ''} | {BONACA_REF['leading_fan_width_deg']:.3f} @ {BONACA_REF['leading_fan_phi1_deg']:.1f} |")
    lines.append(f"| trailing max width in [-15, -5] [deg] | {fmt(metrics_map.trailing_width_max_m15to5_deg, 3)} @ {fmt(metrics_map.trailing_width_max_m15to5_phi1, 2)} | {f'{fmt(alt.trailing_width_max_m15to5_deg, 3)} @ {fmt(alt.trailing_width_max_m15to5_phi1, 2)}' if alt else ''} | trailing should stay relatively thin |")
    lines.append("")

    # Add a compact note on cluster bins and like-for-like integration.
    lines.append("## Notes")
    lines.append("")
    lines.append("- The integrated star counts above use the **linear density profile times the phi1 bin spacing**.")
    lines.append("  This is the closest like-for-like quantity to Bonaca's integrated stream-star count.")
    lines.append("- Cluster bins are excluded from the like-for-like totals and from the arm polynomial fits.")
    lines.append("- Width is usually the least stable profile; use the MAP run as the baseline morphology")
    lines.append("  and use the emcee run primarily as a posterior sanity check.")
    lines.append("")

    # Include a short list of bins that look suspicious in the baseline run.
    suspicious = merged[
        (merged["map_success"].fillna(False).astype(bool))
        & (~merged["map_cluster_bin"].fillna(False).astype(bool))
        & (merged["map_sigma"] > 0.35)
    ][["phi1_center", "map_sigma", "map_track_resid"]].copy()
    if len(suspicious) > 0:
        lines.append("## Wide / suspicious baseline bins")
        lines.append("")
        lines.append("| phi1_center | sigma_map [deg] | track_resid_map [deg] |")
        lines.append("|---:|---:|---:|")
        for _, row in suspicious.iterrows():
            lines.append(f"| {row['phi1_center']:.2f} | {row['map_sigma']:.3f} | {row['map_track_resid']:.3f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _add_cluster_spans(ax: plt.Axes, df: pd.DataFrame, color: str = "0.92") -> None:
    for _, row in df.loc[df["cluster_bin"].astype(bool)].iterrows():
        ax.axvspan(row["phi1_lo"], row["phi1_hi"], color=color, zorder=0)


def _plot_profile_with_errors(ax: plt.Axes, x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                              mask: np.ndarray, **kwargs) -> None:
    if np.any(mask):
        ax.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt="o", ms=4, capsize=3, **kwargs)


def make_profile_stack_figure(
    outpath: Path,
    df_map: pd.DataFrame,
    label_map: str,
    df_alt: Optional[pd.DataFrame],
    label_alt: Optional[str],
) -> None:
    phi1 = df_map["phi1_center"].to_numpy(dtype=float)
    masks = make_masks(df_map)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    ax_track, ax_resid, ax_dens, ax_width = axes

    # Track
    _add_cluster_spans(ax_track, df_map)
    _plot_profile_with_errors(
        ax_track, phi1,
        df_map["mu"].to_numpy(dtype=float),
        df_map["mu_err"].to_numpy(dtype=float),
        masks["usable"],
        color="C0", label=label_map,
    )
    # cluster bins as squares
    cmask = masks["cluster"]
    if np.any(cmask):
        ax_track.plot(phi1[cmask], df_map.loc[cmask, "mu"], "s", color="0.5", ms=6, label="cluster bins")
    ax_track.plot(phi1[masks["trailing"]], df_map.loc[masks["trailing"], "track_poly"], "--", color="C1", lw=2, label="trailing quadratic")
    ax_track.plot(phi1[masks["leading"]], df_map.loc[masks["leading"], "track_poly"], "--", color="C2", lw=2, label="leading quadratic")
    if df_alt is not None:
        alt_masks = make_masks(df_alt)
        _plot_profile_with_errors(
            ax_track, phi1,
            df_alt["mu"].to_numpy(dtype=float),
            df_alt["mu_err"].to_numpy(dtype=float),
            alt_masks["usable"],
            color="C3", alpha=0.65, label=label_alt,
        )
    ax_track.set_ylabel(r"track $\mu(\phi_1)$ [deg]")
    ax_track.set_title("Step 3c: Bonaca-style stream profiles")
    ax_track.legend(loc="best")

    # Residual
    _add_cluster_spans(ax_resid, df_map)
    _plot_profile_with_errors(
        ax_resid, phi1,
        df_map["track_resid"].to_numpy(dtype=float),
        df_map["mu_err"].to_numpy(dtype=float),
        masks["usable"],
        color="C0", label=label_map,
    )
    if df_alt is not None:
        alt_masks = make_masks(df_alt)
        _plot_profile_with_errors(
            ax_resid, phi1,
            df_alt["track_resid"].to_numpy(dtype=float),
            df_alt["mu_err"].to_numpy(dtype=float),
            alt_masks["usable"],
            color="C3", alpha=0.65, label=label_alt,
        )
    for xref in BONACA_REF["track_wiggle_phi1_deg"]:
        ax_resid.axvline(xref, color="0.85", ls=":", lw=1)
    ax_resid.axhline(0, color="0.6", lw=1)
    ax_resid.set_ylabel("track residual [deg]")

    # Density
    _add_cluster_spans(ax_dens, df_map)
    _plot_profile_with_errors(
        ax_dens, phi1,
        df_map["linear_density"].to_numpy(dtype=float),
        df_map["linear_density_err"].to_numpy(dtype=float),
        masks["usable"],
        color="C0", label=label_map,
    )
    if np.any(cmask):
        ax_dens.plot(phi1[cmask], df_map.loc[cmask, "linear_density"], "s", color="0.5", ms=6)
    if df_alt is not None:
        alt_masks = make_masks(df_alt)
        _plot_profile_with_errors(
            ax_dens, phi1,
            df_alt["linear_density"].to_numpy(dtype=float),
            df_alt["linear_density_err"].to_numpy(dtype=float),
            alt_masks["usable"],
            color="C3", alpha=0.65, label=label_alt,
        )
    for xref in BONACA_REF["prominent_gap_phi1_deg"]:
        ax_dens.axvline(xref, color="0.85", ls=":", lw=1)
    ax_dens.set_ylabel("stream stars / deg")

    # Width
    _add_cluster_spans(ax_width, df_map)
    _plot_profile_with_errors(
        ax_width, phi1,
        df_map["sigma"].to_numpy(dtype=float),
        df_map["sigma_err"].to_numpy(dtype=float),
        masks["usable"],
        color="C0", label=label_map,
    )
    if np.any(cmask):
        ax_width.plot(phi1[cmask], df_map.loc[cmask, "sigma"], "s", color="0.5", ms=6)
    if df_alt is not None:
        alt_masks = make_masks(df_alt)
        _plot_profile_with_errors(
            ax_width, phi1,
            df_alt["sigma"].to_numpy(dtype=float),
            df_alt["sigma_err"].to_numpy(dtype=float),
            alt_masks["usable"],
            color="C3", alpha=0.65, label=label_alt,
        )
    ax_width.axhline(BONACA_REF["cluster_width_deg"], color="0.82", ls="--", lw=1, label="Bonaca near-cluster ref")
    ax_width.axvline(BONACA_REF["leading_fan_phi1_deg"], color="0.85", ls=":", lw=1)
    ax_width.set_ylabel(r"width $\sigma(\phi_1)$ [deg]")
    ax_width.set_xlabel(r"$\phi_1$ [deg]")
    ax_width.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_local_map_figure(
    outpath: Path,
    strict_fits: Path,
    df_map: pd.DataFrame,
    bin_phi1: float,
    bin_phi2: float,
    phi2_min: float,
    phi2_max: float,
) -> None:
    if Table is None:
        raise RuntimeError("astropy is required to remake the local density map")
    tab = Table.read(strict_fits)
    phi1 = np.asarray(tab["PHI1"], dtype=float)
    phi2 = np.asarray(tab["PHI2"], dtype=float)
    m = np.isfinite(phi1) & np.isfinite(phi2) & (phi2 >= phi2_min) & (phi2 <= phi2_max)
    phi1 = phi1[m]
    phi2 = phi2[m]

    p1_min = float(np.floor(np.min(phi1) / bin_phi1) * bin_phi1)
    p1_max = float(np.ceil(np.max(phi1) / bin_phi1) * bin_phi1)
    p2_min = float(np.floor(phi2_min / bin_phi2) * bin_phi2)
    p2_max = float(np.ceil(phi2_max / bin_phi2) * bin_phi2)

    e1 = np.arange(p1_min, p1_max + bin_phi1, bin_phi1)
    e2 = np.arange(p2_min, p2_max + bin_phi2, bin_phi2)
    H, _, _ = np.histogram2d(phi1, phi2, bins=[e1, e2])
    area = bin_phi1 * bin_phi2
    D = H / area

    fig, ax = plt.subplots(figsize=(12, 6.8))
    im = ax.imshow(
        D.T,
        origin="lower",
        aspect="auto",
        extent=[e1[0], e1[-1], e2[0], e2[-1]],
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"counts / deg$^2$")

    good = df_map["success"].astype(bool).to_numpy() & (~df_map["cluster_bin"].astype(bool).to_numpy())
    ax.plot(df_map.loc[good, "phi1_center"], df_map.loc[good, "mu"], color="C0", lw=2.5, label="baseline track")
    ax.plot(df_map.loc[good, "phi1_center"], df_map.loc[good, "mu"] + df_map.loc[good, "sigma"], "--", color="C1", lw=1.5, label=r"$\mu \pm \sigma$")
    ax.plot(df_map.loc[good, "phi1_center"], df_map.loc[good, "mu"] - df_map.loc[good, "sigma"], "--", color="C2", lw=1.5)

    cmask = df_map["cluster_bin"].astype(bool).to_numpy()
    if np.any(cmask):
        ax.plot(df_map.loc[cmask, "phi1_center"], df_map.loc[cmask, "mu"], "s", color="0.5", ms=7, label="cluster bins")

    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title("Step 3c: strict-sample local Pal 5 density + baseline track")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_alt_comparison_figure(
    outpath: Path,
    df_map: pd.DataFrame,
    df_alt: pd.DataFrame,
    label_map: str,
    label_alt: str,
) -> None:
    good = (
        df_map["success"].astype(bool).to_numpy()
        & df_alt["success"].astype(bool).to_numpy()
        & (~df_map["cluster_bin"].astype(bool).to_numpy())
        & (~df_alt["cluster_bin"].astype(bool).to_numpy())
    )
    phi1 = df_map.loc[good, "phi1_center"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].errorbar(phi1, df_map.loc[good, "mu"], yerr=df_map.loc[good, "mu_err"], fmt="o", ms=4, capsize=3, color="C0", label=label_map)
    axes[0].errorbar(phi1, df_alt.loc[good, "mu"], yerr=df_alt.loc[good, "mu_err"], fmt="o", ms=4, capsize=3, color="C3", alpha=0.7, label=label_alt)
    axes[0].set_ylabel(r"track $\mu$ [deg]")
    axes[0].legend(loc="best")

    axes[1].errorbar(phi1, df_map.loc[good, "sigma"], yerr=df_map.loc[good, "sigma_err"], fmt="o", ms=4, capsize=3, color="C0")
    axes[1].errorbar(phi1, df_alt.loc[good, "sigma"], yerr=df_alt.loc[good, "sigma_err"], fmt="o", ms=4, capsize=3, color="C3", alpha=0.7)
    axes[1].set_ylabel(r"width $\sigma$ [deg]")
    axes[1].set_ylim(bottom=0)

    axes[2].errorbar(phi1, df_map.loc[good, "linear_density"], yerr=df_map.loc[good, "linear_density_err"], fmt="o", ms=4, capsize=3, color="C0")
    axes[2].errorbar(phi1, df_alt.loc[good, "linear_density"], yerr=df_alt.loc[good, "linear_density_err"], fmt="o", ms=4, capsize=3, color="C3", alpha=0.7)
    axes[2].set_ylabel("stars / deg")
    axes[2].set_xlabel(r"$\phi_1$ [deg]")

    fig.suptitle("Step 3c: baseline vs alternate run")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_asymmetry_figure(
    outpath: Path,
    metrics_map: ProfileMetrics,
    metrics_alt: Optional[ProfileMetrics],
    label_map: str,
    label_alt: Optional[str],
    inner_limit: float,
    compare_limit: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    limits = [inner_limit, compare_limit]
    for ax, lim in zip(axes, limits):
        if math.isclose(lim, inner_limit):
            vals_map = [metrics_map.integrated_trailing_abs5, metrics_map.integrated_leading_abs5]
            vals_alt = ([metrics_alt.integrated_trailing_abs5, metrics_alt.integrated_leading_abs5] if metrics_alt else None)
            ref_ratio = BONACA_REF["trailing_to_leading_ratio_within_5deg"]
        else:
            vals_map = [metrics_map.integrated_trailing_abs8, metrics_map.integrated_leading_abs8]
            vals_alt = ([metrics_alt.integrated_trailing_abs8, metrics_alt.integrated_leading_abs8] if metrics_alt else None)
            ref_ratio = 1.0
        x = np.arange(2)
        ax.bar(x - 0.17, vals_map, width=0.34, label=label_map)
        if vals_alt is not None:
            ax.bar(x + 0.17, vals_alt, width=0.34, alpha=0.75, label=label_alt)
        ax.set_xticks(x)
        ax.set_xticklabels(["trailing", "leading"])
        ax.set_title(rf"Integrated stars, $|\phi_1|<{lim:.0f}^\circ$")
        ax.text(0.03, 0.95, f"Bonaca ref ratio ≈ {ref_ratio:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9)
    axes[0].set_ylabel("integrated stars")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    profiles_map_path = Path(args.profiles_map)
    summary_map_path = Path(args.summary_map)
    if not profiles_map_path.exists():
        raise FileNotFoundError(f"Baseline profiles not found: {profiles_map_path}")
    if not summary_map_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {summary_map_path}")

    df_map = load_profiles(profiles_map_path)
    summary_map = load_json(summary_map_path)

    df_alt = None
    summary_alt = None
    if not args.no_alt:
        profiles_alt_path = Path(args.profiles_alt)
        summary_alt_path = Path(args.summary_alt)
        if profiles_alt_path.exists() and summary_alt_path.exists():
            df_alt = load_profiles(profiles_alt_path)
            summary_alt = load_json(summary_alt_path)

    metrics_map = summarize_metrics(
        df=df_map,
        summary=summary_map,
        label=args.label_map,
        compare_limit=args.compare_limit,
        inner_limit=args.inner_limit,
        cluster_width_window=args.cluster_width_window,
        leading_width_min=args.leading_width_min,
        leading_width_max=args.leading_width_max,
        trailing_width_min=args.trailing_width_min,
        trailing_width_max=args.trailing_width_max,
    )
    metrics_alt = None
    if df_alt is not None and summary_alt is not None:
        metrics_alt = summarize_metrics(
            df=df_alt,
            summary=summary_alt,
            label=args.label_alt,
            compare_limit=args.compare_limit,
            inner_limit=args.inner_limit,
            cluster_width_window=args.cluster_width_window,
            leading_width_min=args.leading_width_min,
            leading_width_max=args.leading_width_max,
            trailing_width_min=args.trailing_width_min,
            trailing_width_max=args.trailing_width_max,
        )

    merged = merge_runs(df_map, df_alt)
    merged.to_csv(outdir / "pal5_step3c_profile_table.csv", index=False)

    report_json = {
        "baseline": metrics_map.to_dict(),
        "alternate": metrics_alt.to_dict() if metrics_alt is not None else None,
        "bonaca_reference": BONACA_REF,
        "input_files": {
            "profiles_map": str(profiles_map_path),
            "summary_map": str(summary_map_path),
            "profiles_alt": str(Path(args.profiles_alt)) if df_alt is not None else None,
            "summary_alt": str(Path(args.summary_alt)) if summary_alt is not None else None,
            "strict_fits": str(Path(args.strict_fits)) if (not args.no_map) else None,
        },
        "plot_config": {
            "compare_limit": args.compare_limit,
            "inner_limit": args.inner_limit,
            "cluster_width_window": args.cluster_width_window,
            "leading_width_window": [args.leading_width_min, args.leading_width_max],
            "trailing_width_window": [args.trailing_width_min, args.trailing_width_max],
        },
    }
    write_json(outdir / "pal5_step3c_summary.json", report_json)
    pd.DataFrame([
        {"run": metrics_map.label, **metrics_map.to_dict()},
        *([{"run": metrics_alt.label, **metrics_alt.to_dict()}] if metrics_alt is not None else []),
    ]).to_csv(outdir / "pal5_step3c_metrics.csv", index=False)

    write_markdown_report(
        outdir / "pal5_step3c_report.md",
        metrics_map=metrics_map,
        metrics_alt=metrics_alt,
        summary_map=summary_map,
        summary_alt=summary_alt,
        merged=merged,
        compare_limit=args.compare_limit,
        inner_limit=args.inner_limit,
    )

    make_profile_stack_figure(
        outdir / "fig_step3c_bonaca_profiles.png",
        df_map=df_map,
        label_map=args.label_map,
        df_alt=df_alt,
        label_alt=(args.label_alt if df_alt is not None else None),
    )

    make_asymmetry_figure(
        outdir / "fig_step3c_asymmetry.png",
        metrics_map=metrics_map,
        metrics_alt=metrics_alt,
        label_map=args.label_map,
        label_alt=(args.label_alt if metrics_alt is not None else None),
        inner_limit=args.inner_limit,
        compare_limit=args.compare_limit,
    )

    if df_alt is not None:
        make_alt_comparison_figure(
            outdir / "fig_step3c_baseline_vs_alternate.png",
            df_map=df_map,
            df_alt=df_alt,
            label_map=args.label_map,
            label_alt=args.label_alt,
        )

    if (not args.no_map) and Path(args.strict_fits).exists():
        make_local_map_figure(
            outdir / "fig_step3c_local_map.png",
            strict_fits=Path(args.strict_fits),
            df_map=df_map,
            bin_phi1=args.bin_phi1,
            bin_phi2=args.bin_phi2,
            phi2_min=args.phi2_min,
            phi2_max=args.phi2_max,
        )

    print("[done] wrote step 3c outputs to", outdir)
    print("[done] baseline success bins:", metrics_map.n_success, "excluding cluster:", metrics_map.n_success_excluding_cluster)
    if metrics_alt is not None:
        print("[done] alternate success bins:", metrics_alt.n_success, "excluding cluster:", metrics_alt.n_success_excluding_cluster)


if __name__ == "__main__":
    main()
