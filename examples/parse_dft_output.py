"""Parse an ORCA TD-DFT output and print the n-pi* and pi-pi* absorption rows.

Uses the bundled sample output in data/sample/sample_orca.out.
"""

# Run with: uv run examples/parse_dft_output.py

from __future__ import annotations

from pathlib import Path

from photoswitch.dft import extract_wavelength_from_calculation

# Column order returned by extract_wavelength_from_calculation.
COLUMNS = ["State", "Energy", "Wavelength", "fosc", "T2", "TX", "TY", "TZ"]


def _print_row(label: str, row: list[str]) -> None:
    record = dict(zip(COLUMNS, row, strict=False))
    print(f"{label:8s}  Wavelength (nm) = {record['Wavelength']:>8s}  fosc = {record['fosc']:>8s}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    orca_dir = str(repo_root / "data" / "sample") + "/"
    smiles = "sample_orca"

    pi_pi = extract_wavelength_from_calculation(orca_dir, smiles, task="pi-pi*")
    n_pi = extract_wavelength_from_calculation(orca_dir, smiles, task="n-pi*")

    print(f"Parsed ORCA output: {orca_dir}{smiles}.out\n")
    print("Strongest absorption transitions:")
    _print_row("pi-pi*", pi_pi)
    _print_row("n-pi*", n_pi)


if __name__ == "__main__":
    main()
