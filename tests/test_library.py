"""Characterization tests for photoswitch.library.generate_smiles."""

from rdkit import Chem

from photoswitch.library import generate_smiles


def test_generates_file_of_valid_canonical_smiles(tmp_path):
    out = tmp_path / "library.txt"
    parent_mols = ["[U](C1=CC=C(C=C1)N=NC2=CC=C([U])C=C2)"]  # azobenzene core
    linkers = ["([H])", "(c2ccc([Y])cc2)"]  # H-terminus, benzene
    terminals = ["([H])", "[F]"]  # hydrogen, fluoro

    generate_smiles(str(out), parent_mols, linkers, terminals)

    assert out.exists()
    lines = [line.strip() for line in out.read_text().splitlines() if line.strip()]
    assert lines, "expected at least one generated SMILES"
    # Every emitted line must parse as a molecule.
    for smiles in lines:
        assert Chem.MolFromSmiles(smiles) is not None


def test_emitted_smiles_are_unique(tmp_path):
    out = tmp_path / "library.txt"
    generate_smiles(
        str(out),
        ["[U](C1=CC=C(C=C1)N=NC2=CC=C([U])C=C2)"],
        ["([H])"],
        ["([H])", "[F]"],
    )
    lines = [line.strip() for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == len(set(lines))
