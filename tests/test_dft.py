"""Characterization tests for photoswitch.dft output parsing.

Only the pure parser (extract_wavelength_from_calculation) is exercised here;
run_orca_calculation needs the external ORCA binary and the chemml extra.
"""

from photoswitch.dft import extract_wavelength_from_calculation


def test_returns_message_when_output_missing(tmp_path):
    result = extract_wavelength_from_calculation(str(tmp_path) + "/", "does_not_exist", "n_pi")
    assert result == "Calculation has not been run yet"


def test_n_pi_returns_highest_oscillator_strength_row(sample_orca_out):
    orca_dir = str(sample_orca_out.parent) + "/"
    smiles = sample_orca_out.stem  # filename is built as orca_dir + smiles + '.out'
    n_pi = extract_wavelength_from_calculation(orca_dir, smiles, "n_pi")
    # State 2 has the largest fosc (0.500): wavelength 333.3
    assert n_pi[0] == "2"
    assert n_pi[2] == "333.3"
    assert n_pi[3] == "0.500"


def test_pi_pi_returns_second_highest_oscillator_strength_row(sample_orca_out):
    orca_dir = str(sample_orca_out.parent) + "/"
    smiles = sample_orca_out.stem
    pi_pi = extract_wavelength_from_calculation(orca_dir, smiles, "pi_pi")
    # State 1 has the second-largest fosc (0.300): wavelength 454.5
    assert pi_pi[0] == "1"
    assert pi_pi[2] == "454.5"
    assert pi_pi[3] == "0.300"
