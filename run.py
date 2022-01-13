import warnings
warnings.filterwarnings("ignore")

from rdkit.Chem import MolFromSmiles
from mordred import Calculator, descriptors

smiles_list = []
with open('library_01.txt', 'r') as f:
  lines = f.readlines()
  for line in lines:
    smiles_list.append(line.rstrip())

rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]

calc = Calculator(descriptors)

mordred_descriptors = calc.pandas(rdkit_mols)

error_columns = []
for i, e in enumerate(mordred_descriptors.dtypes):
    if e=="object":
        error_columns += [i]
        
mordred_descriptors = mordred_descriptors.drop(mordred_descriptors.columns[error_columns], axis=1)
mordred_descriptors = mordred_descriptors.dropna(axis=1)
mordred_descriptors.insert(0, "SMILES", smiles_list)
mordred_descriptors.to_csv('./mordred_descriptors_library_01.csv')