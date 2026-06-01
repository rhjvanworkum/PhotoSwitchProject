from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from itertools import chain
rdBase.DisableLog('rdApp.warning')

linker_reaction_template = '[*:1]([U])>>[*:1]'
terminal_reaction_template = '[*:1]([Y])>>[*:1]'


def generate_smiles(filename, parent_mols, linkers, terminals):
    """Function used to generate a library of molecules following the format of a
    [Terminal] - [Linker] - [Parent Mol] - [Linker] - [Terminal], where parent mol is
    some kind of photoswitch molecule (like azobenzene)

    Args:
        filename (string): path to the file to which the library of smiles strings will be writen to
        parent_mols (string): a smart string describing the parent molecule and the linker binding sites
        linkers (list): a smart string describing the linker and the terminal binding site
        terminals (list): a smart string describing the terminal
    """
    all_smiles = []
    
    for parent_mol in parent_mols:
        parent_mol = Chem.MolFromSmarts(parent_mol)

        # placeholder for linker addition
        linker_place_holder = '[*:1]([U])'
        linker_place_holder_mol = Chem.MolFromSmarts(linker_place_holder)

        # append linkers to parent molecule to generate unsubstituted cores
        unsubstituted_cores = []
        place_holder_count = len(parent_mol.GetSubstructMatches(linker_place_holder_mol))

        for linker in linkers:
            rxn = AllChem.ReactionFromSmarts(linker_reaction_template + linker)
            core = parent_mol
            for i in range(place_holder_count):
                new_mols = list(chain.from_iterable(rxn.RunReactants((core,))))
                if len(new_mols) > 0:
                    core = new_mols[0]
                    Chem.SanitizeMol(core)
            unsubstituted_cores.append(core)

        substituent_place_holder = '[*:1]([Y])'
        substituent_place_holder_mol = Chem.MolFromSmarts(substituent_place_holder)

        # append terminal groups
        all_mols = []
        for core in unsubstituted_cores:
            place_holder_count = len(
                core.GetSubstructMatches(substituent_place_holder_mol))
            if place_holder_count == 0:
                all_mols.append(core)
                continue
            for terminal in terminals:
                new_mol = core
                rxn = AllChem.ReactionFromSmarts(terminal_reaction_template + terminal)
                for i in range(place_holder_count):
                    new_mols = list(chain.from_iterable(rxn.RunReactants((new_mol,))))
                    new_mol = new_mols[0]
                    Chem.Cleanup(new_mol)
                all_mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))

        # canonicalize smiles to remove duplicates
        all_mols = [Chem.MolFromSmiles(smiles) for smiles in [
            Chem.MolToSmiles(mol) for mol in all_mols]]
        print(len(all_mols))
        all_smiles += list(set([Chem.MolToSmiles(mol) for mol in all_mols]))

    # write list of SMILES to text file
    with open(filename, "w") as f:
        for smiles in all_smiles:
            f.write(smiles + "\n")