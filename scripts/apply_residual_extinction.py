from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table


def apply_extinction_residual(input_path: Path, output_path: Path, frac: float) -> None:
    table = Table.read(input_path)
    for mag_col, ext_col in [("G0", "A_G"), ("R0", "A_R"), ("Z0", "A_Z")]:
        if mag_col not in table.colnames or ext_col not in table.colnames:
            raise KeyError(f"missing required columns {mag_col} and/or {ext_col} in {input_path}")
        table[mag_col] = table[mag_col] + frac * table[ext_col]
    table["GR0"] = table["G0"] - table["R0"]
    table["RZ0"] = table["R0"] - table["Z0"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, overwrite=True)
    print(f"wrote {output_path} with residual extinction fraction {frac:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the residual extinction correction used in the Pal 5 notebooks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frac", type=float, default=0.14)
    args = parser.parse_args()
    apply_extinction_residual(Path(args.input), Path(args.output), args.frac)


if __name__ == "__main__":
    main()
