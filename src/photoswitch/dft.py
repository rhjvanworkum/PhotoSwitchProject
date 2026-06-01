"""Generate ORCA TD-DFT input files and parse absorption wavelengths back out.

``run_orca_calculation`` needs the external ORCA program and the optional
``chemml`` extra; ``extract_wavelength_from_calculation`` is pure Python and
only needs pandas.
"""

from __future__ import annotations

import os

import pandas as pd

orca_input_file_string = """
! BP86 def2-SVP def2/J TightSCF

%maxcore 1000 # Memory settings often need to be modified when running TDDFT. Check batching info in the TDDFT output.
%tddft
nroots 150   # Setting the number of roots (transitions) to be calculated.
maxdim 5 # Davidson expansion space = MaxDim * nroots. Use MaxDim 5-10 for favorable convergence. Note that the larger MaxDim is, the more disk space is required
end

%scf MaxIter 150
end

* xyz   0   1
"""

# n-pi* / pi-pi* task aliases, so both the underscore form used internally and
# the hyphen/star form written in the notebooks resolve to the same result.
_N_PI_TASKS = {"n_pi", "n-pi*"}
_PI_PI_TASKS = {"pi_pi", "pi-pi*"}


def _orca_basename(orca_dir: str, smiles: str) -> str:
    """Build a filesystem-safe base path (no extension) for a molecule.

    SMILES routinely contain ``/`` and ``\\`` (cis/trans bonds), which are path
    separators; replace them so the resulting file lives inside ``orca_dir``.
    """
    safe = smiles.replace("/", "_").replace("\\", "_")
    return os.path.join(orca_dir, safe)


def run_orca_calculation(orca_dir: str, smiles: str) -> None:
    """Write an ORCA TD-DFT input file for ``smiles`` and run ORCA on it.

    Args:
        orca_dir: Output directory for the ORCA calculation files.
        smiles: The input molecule.
    """
    # chemml is an optional extra (`pip install photoswitch[dft]`); import it
    # lazily so the pure output parser below stays importable without it.
    from chemml.chem import Molecule

    if not os.path.isdir(orca_dir):
        os.makedirs(orca_dir)

    filename = _orca_basename(orca_dir, smiles)

    mol = Molecule(smiles, "smiles")
    mol.hydrogens("add")
    mol.to_xyz(optimizer="MMFF", mmffVariant="MMFF94s", maxIters=300)
    symb = mol.xyz.atomic_symbols
    geom = mol.xyz.geometry

    with open(filename + ".inp", "w") as f:
        f.write(orca_input_file_string)
        for idx in range(len(symb)):
            symbol = symb[idx]
            geometry = geom[idx]
            f.write(" " + str(symbol[0]) + "  " + str(geometry)[1:-1])
            f.write("\n")
        f.write("\n")
        f.write("end")

    os.system("orca " + filename + ".inp" + " > " + filename + ".out")


def extract_wavelength_from_calculation(orca_dir: str, smiles: str, task: str):
    """Extract the n-pi* or pi-pi* absorption row from an ORCA output file.

    The two strongest transitions (by the ``fosc`` column) are returned as the
    n-pi* and pi-pi* rows respectively.

    Note:
        ``fosc`` is compared as text, matching the original implementation; this
        is reliable for the usual ``0.x`` oscillator strengths but can misorder
        values with differing magnitudes. Kept as-is to preserve behavior.

    Args:
        orca_dir: Directory containing the ORCA output file.
        smiles: The molecule whose output to read.
        task: ``"n_pi"``/``"n-pi*"`` or ``"pi_pi"``/``"pi-pi*"``.

    Returns:
        The matching spectrum row as a list, or a message string if the
        calculation has not been run, or ``None`` for an unknown task.
    """
    filename = _orca_basename(orca_dir, smiles) + ".out"

    if not os.path.exists(filename):
        return "Calculation has not been run yet"

    counter = 0
    temp_list = []
    with open(filename) as read:
        for line in read:
            if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                for line in read:
                    if line != "\n":
                        counter += 1
                        if counter >= 5:
                            new_line = " ".join(line.split())
                            tddft_vals = new_line.split()
                            temp_list.append(tddft_vals)
                    else:
                        break

    df = pd.DataFrame(
        temp_list,
        columns=["State", "Energy", "Wavelength", "fosc", "T2", "TX", "TY", "TZ"],
    )
    df = df.sort_values(by=["fosc"], ascending=False)

    n_pi = list(df.iloc[0])
    pi_pi = list(df.iloc[1])

    if task in _N_PI_TASKS:
        return n_pi
    elif task in _PI_PI_TASKS:
        return pi_pi
    return None
