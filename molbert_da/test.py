import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

path_to_checkpoint = './molbert_100epochs/checkpoints/last.ckpt'
f = MolBertFeaturizer(path_to_checkpoint, device='cpu')

smiles = pd.read_csv('../raw_data/photoswitches.csv').to_numpy()[:, 1]

# # for now just used to pretrained dataset, later on might fine tune a model
# df = pd.read_csv('../raw_data/photoswitches.csv')
# columns = df.columns

# name = 'E isomer pi-pi* wavelength in nm'
# property_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()
# invalid_indices = np.argwhere(np.isnan(property_vals))

# df = np.delete(df.to_numpy(), invalid_indices, axis=0)

# df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
# df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=1)

# pd.DataFrame(df_train, columns=columns).to_csv('df_train.csv')
# pd.DataFrame(df_test, columns=columns).to_csv('df_test.csv')
# pd.DataFrame(df_val, columns=columns).to_csv('df_val.csv')


features, masks = f.transform(smiles)
assert all(masks)

index = ['Row'+str(i) for i in range(1, len(features[0]) + 1)]
df = pd.DataFrame(features, columns=index)
df.insert(loc=0, column='Smiles', value=smiles)

df.to_csv('./test.csv')

# git clone https://github.com/BenevolentAI/MolBERT.git

# python ./molbert/apps/finetune.py --train_file ./df_train.csv --valid_file ./df_val.csv --test_file ./df_test.csv --mode regression --output_size 1 --pretrained_model_path ./molbert_100epochs/checkpoints/last.ckpt --label_column "E isomer pi-pi* wavelength in nm"