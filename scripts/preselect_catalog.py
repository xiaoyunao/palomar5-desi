from __future__ import annotations

import argparse
import glob
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from dustmaps.sfd import SFDQuery
from gala.coordinates import Pal5PriceWhelan18


KEEP_COLS = [
    "ID",
    "RA",
    "DEC",
    "MAG_G",
    "MAG_R",
    "MAG_Z",
    "MAG_W1",
    "MAG_W2",
    "MAGERR_G",
    "MAGERR_R",
    "MAGERR_Z",
    "MAGERR_W1",
    "MAGERR_W2",
    "TYPE",
    "MASKBITS",
    "FITBITS",
    "PSFDEPTH_G",
    "PSFDEPTH_R",
    "PSFDEPTH_Z",
    "SHAPE_R",
    "SHAPE_R_IVAR",
    "SERSIC",
    "SERSIC_IVAR",
    "GAIA_PHOT_G_MEAN_MAG",
    "GAIA_PHOT_BP_MEAN_MAG",
    "GAIA_PHOT_RP_MEAN_MAG",
    "PARALLAX",
    "PARALLAX_IVAR",
    "PMRA",
    "PMRA_IVAR",
    "PMDEC",
    "PMDEC_IVAR",
    "MJD_MIN",
    "MJD_MAX",
    "EBV_SFD",
    "A_G",
    "A_R",
    "A_Z",
    "G0",
    "R0",
    "Z0",
    "GR0",
    "RZ0",
    "PHI1",
    "PHI2",
]


def type_to_str(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in ("S", "O"):
        return np.char.decode(arr.astype("S"), "ascii", errors="ignore")
    return arr.astype(str)


def merge_fits_list(file_list: list[str], output_path: Path, batch_size: int = 20) -> None:
    file_list = sorted(file_list)
    if not file_list:
        raise RuntimeError(f"No files to merge for {output_path}")
    merged = None
    for start in range(0, len(file_list), batch_size):
        group = file_list[start : start + batch_size]
        tables = [Table.read(path) for path in group]
        chunk = vstack(tables, metadata_conflicts="silent")
        merged = chunk if merged is None else vstack([merged, chunk], metadata_conflicts="silent")
        print(f"merged {start + len(group)}/{len(file_list)} temporary files -> {len(merged):,} rows")
    merged.write(output_path, overwrite=True)
    print(f"wrote {output_path} with {len(merged):,} rows")


def process_chunk(
    sub: np.ndarray,
    chunk_id: int,
    temp_dir: Path,
    sfd: SFDQuery,
    pal5_frame: Pal5PriceWhelan18,
    chunk_size: int,
) -> tuple[int, int]:
    ra = np.asarray(sub["RA"], dtype=np.float64)
    dec = np.asarray(sub["DEC"], dtype=np.float64)
    mask = (ra > 210.0) & (ra < 260.0) & (dec > -20.0) & (dec < 20.0)
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]

    ra0 = np.deg2rad(229.64)
    dec0 = np.deg2rad(2.08)
    v0 = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)], dtype=np.float64)
    cos_rad = np.cos(np.deg2rad(0.3))

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    mask = (x * v0[0] + y * v0[1] + z * v0[2]) <= cos_rad
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]

    types = type_to_str(sub["TYPE"])
    mask = np.isin(types, ["PSF", "REX"])
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]

    g = np.asarray(sub["MAG_G"], dtype=np.float32)
    r = np.asarray(sub["MAG_R"], dtype=np.float32)
    zmag = np.asarray(sub["MAG_Z"], dtype=np.float32)
    psfdepth_g = np.asarray(sub["PSFDEPTH_G"], dtype=np.float32)
    mask = np.isfinite(g) & np.isfinite(r) & np.isfinite(zmag) & np.isfinite(psfdepth_g)
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]
    g = g[mask]
    r = r[mask]
    zmag = zmag[mask]
    psfdepth_g = psfdepth_g[mask]

    shape_r = np.asarray(sub["SHAPE_R"], dtype=np.float32)
    shape_r_ivar = np.asarray(sub["SHAPE_R_IVAR"], dtype=np.float32)
    sersic = np.asarray(sub["SERSIC"], dtype=np.float32)
    sersic_ivar = np.asarray(sub["SERSIC_IVAR"], dtype=np.float32)

    shape_good = np.isfinite(shape_r) & np.isfinite(shape_r_ivar) & (shape_r_ivar > 0.0)
    sersic_good = np.isfinite(sersic) & np.isfinite(sersic_ivar) & (sersic_ivar > 0.0)
    shape_snr = np.zeros(len(sub), dtype=np.float32)
    sersic_snr = np.zeros(len(sub), dtype=np.float32)
    shape_snr[shape_good] = shape_r[shape_good] * np.sqrt(shape_r_ivar[shape_good])
    sersic_snr[sersic_good] = sersic[sersic_good] * np.sqrt(sersic_ivar[sersic_good])
    is_extended = ((shape_r > 1.5) & (shape_snr > 5.0)) | ((sersic > 2.0) & (sersic_snr > 5.0))
    mask = ~is_extended
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ra = ra[mask]
    dec = dec[mask]
    g = g[mask]
    r = r[mask]
    zmag = zmag[mask]
    psfdepth_g = psfdepth_g[mask]

    coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    ebv = np.asarray(sfd(coords), dtype=np.float32)
    a_g = ebv * 3.30
    a_r = ebv * 2.31
    a_z = ebv * 1.28

    g0 = (g - a_g).astype(np.float32)
    r0 = (r - a_r).astype(np.float32)
    z0 = (zmag - a_z).astype(np.float32)
    gr0 = (g0 - r0).astype(np.float32)
    rz0 = (r0 - z0).astype(np.float32)

    pal5 = coords.transform_to(pal5_frame)
    phi1 = ((pal5.phi1.to_value(u.deg) + 180.0) % 360.0 - 180.0).astype(np.float32)
    phi2 = pal5.phi2.to_value(u.deg).astype(np.float32)

    mask = (phi1 > -25.0) & (phi1 < 20.0) & (phi2 > -5.0) & (phi2 < 10.0)
    if not np.any(mask):
        return 0, 0
    sub = sub[mask]
    ebv = ebv[mask]
    a_g = a_g[mask]
    a_r = a_r[mask]
    a_z = a_z[mask]
    g0 = g0[mask]
    r0 = r0[mask]
    z0 = z0[mask]
    gr0 = gr0[mask]
    rz0 = rz0[mask]
    phi1 = phi1[mask]
    phi2 = phi2[mask]
    psfdepth_g = psfdepth_g[mask]

    mask_g24 = (g0 > 12.0) & (g0 < 24.0)
    mask_g25 = ((g0 > 12.0) & (g0 <= 24.0)) | ((g0 > 24.0) & (g0 < 25.0) & (psfdepth_g >= 25.0))

    count_g24 = int(np.sum(mask_g24))
    count_g25 = int(np.sum(mask_g25))

    if count_g24:
        table = Table(sub[mask_g24])
        table["EBV_SFD"] = ebv[mask_g24]
        table["A_G"] = a_g[mask_g24]
        table["A_R"] = a_r[mask_g24]
        table["A_Z"] = a_z[mask_g24]
        table["G0"] = g0[mask_g24]
        table["R0"] = r0[mask_g24]
        table["Z0"] = z0[mask_g24]
        table["GR0"] = gr0[mask_g24]
        table["RZ0"] = rz0[mask_g24]
        table["PHI1"] = phi1[mask_g24]
        table["PHI2"] = phi2[mask_g24]
        table = table[[name for name in KEEP_COLS if name in table.colnames]]
        table.write(temp_dir / f"tmp_glt24_chunk{chunk_id:04d}.fits", overwrite=True)

    if count_g25:
        table = Table(sub[mask_g25])
        table["EBV_SFD"] = ebv[mask_g25]
        table["A_G"] = a_g[mask_g25]
        table["A_R"] = a_r[mask_g25]
        table["A_Z"] = a_z[mask_g25]
        table["G0"] = g0[mask_g25]
        table["R0"] = r0[mask_g25]
        table["Z0"] = z0[mask_g25]
        table["GR0"] = gr0[mask_g25]
        table["RZ0"] = rz0[mask_g25]
        table["PHI1"] = phi1[mask_g25]
        table["PHI2"] = phi2[mask_g25]
        table = table[[name for name in KEEP_COLS if name in table.colnames]]
        table.write(temp_dir / f"tmp_glt25_chunk{chunk_id:04d}.fits", overwrite=True)

    return count_g24, count_g25


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-select Pal 5 field stars from the raw DESI catalog.")
    parser.add_argument("--input", required=True, help="Raw DESI FITS catalog")
    parser.add_argument("--out-g24", required=True, help="Output FITS for G<24 selection")
    parser.add_argument("--out-g25", required=True, help="Output FITS for depth-aware G<25 selection")
    parser.add_argument("--chunk-size", type=int, default=10_000_000)
    parser.add_argument("--temp-dir", default="tmp/preselect_chunks")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(temp_dir / "tmp_glt24_chunk*.fits")) + glob.glob(str(temp_dir / "tmp_glt25_chunk*.fits")):
        Path(path).unlink()

    sfd = SFDQuery()
    pal5_frame = Pal5PriceWhelan18()

    total_g24 = 0
    total_g25 = 0
    with fits.open(args.input, memmap=True) as hdul:
        data = hdul[1].data
        print(f"total rows: {len(data):,}")
        chunk_id = 0
        for start in range(0, len(data), args.chunk_size):
            stop = min(start + args.chunk_size, len(data))
            kept_g24, kept_g25 = process_chunk(data[start:stop], chunk_id, temp_dir, sfd, pal5_frame, args.chunk_size)
            total_g24 += kept_g24
            total_g25 += kept_g25
            print(
                f"chunk {chunk_id:04d} rows {start:,}-{stop:,}: "
                f"keep g24={kept_g24:,}, g25={kept_g25:,} | totals g24={total_g24:,}, g25={total_g25:,}"
            )
            chunk_id += 1

    merge_fits_list(glob.glob(str(temp_dir / "tmp_glt24_chunk*.fits")), Path(args.out_g24))
    merge_fits_list(glob.glob(str(temp_dir / "tmp_glt25_chunk*.fits")), Path(args.out_g25))


if __name__ == "__main__":
    main()
