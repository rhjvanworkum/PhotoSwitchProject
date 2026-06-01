"""Generate a small combinatorial photoswitch library with photoswitch.library.generate_smiles.

Builds [Terminal]-[Linker]-[Azobenzene core]-[Linker]-[Terminal] molecules from a tiny
building-block set, writes them to a temp file, then reads them back and reports the results.
"""

# Run with: uv run examples/generate_library.py

from pathlib import Path

from photoswitch.library import generate_smiles


def main():
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "library.smi"

    parent_mols = ["[U](C1=CC=C(C=C1)N=NC2=CC=C([U])C=C2)"]  # azobenzene core
    linkers = ["([H])", "(c2ccc([Y])cc2)"]  # H-terminus, benzene
    terminals = ["([H])", "[F]"]  # hydrogen, fluoro

    generate_smiles(str(out_file), parent_mols, linkers, terminals)

    smiles = [line.strip() for line in out_file.read_text().splitlines() if line.strip()]
    unique = sorted(set(smiles))

    print(f"\nLibrary written to: {out_file}")
    print(f"Unique SMILES generated: {len(unique)}")
    print("First 5 SMILES:")
    for s in unique[:5]:
        print(f"  {s}")


if __name__ == "__main__":
    main()
