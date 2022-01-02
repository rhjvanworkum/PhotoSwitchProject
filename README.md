# PhotoSwitchProject

### Objective of the project: Construct a ML model for in order to predict Transition wavelengths of photoswitch molecules

### Measures of succes:
1. Achieve good performance on the prediction tasks by leveraging external data
2. Implement Graph-based methods that can account for solvent effects on the transition wavelength vai a FiLM mechanism or incorporating extra features
3. Constructing a model where prediction error is interpretable and may be attributed to characteristics of the
molecular representation


### Plan:
1. tackle point 1 through training a fine-tuned pretrained BERT model
2. tackle point 3 through constructing a GPR model where the uncertainty can be investigated



Final Project
For your final project, you will be asked to analyse data from the following paper: The Photoswitch Dataset: A Molecular Machine Learning Benchmark for the Advancement of Synthetic Chemistry.

In this data, you will find a variety of measurements done for a total of 405 molecules. However, the original paper's data only contains SMILES representations of the molecules, and in order to do anything useful with this we need to "featurise" them, which means to turn the character strings describing molecules into quantitative data that can then be used by the models to be learnt and fit on. Since this step can sometimes be a little involved (much research in cheminformatics is dedicated to this field!), I have prepared this for you. You will find the data featurised in three distinct ways:

Using Morgan fingerprints as implemented in RDKit. This is a popular method to turn molecules of any shape into fixed sized vectors of binary data. The authors of the paper also use this method.
Using all molecular descriptors available in RDKit. I'm not sure if a comprehensive list of the descriptors exists, but they can all be looked up in the rdkit documentation. This is partially also used in the paper.
Using the mordred package, containing some 1826 descriptors, 1427 of which returned non-erroneous features for the molecules in the photoswitch dataset. For a complete list of all descriptors, click here. This is not in the paper and may offer additional descriptors that could be important.
In the Ed workspace for this course, you will find a zip file entitled final_project_data, which contains the following:

Featurisation.ipynb (My finished notebook with all the commented (!!!) code necessary to obtain the data below, in case you want to change parameters, in particular for fingerprints. Since this may be a little advanced, this is absolutely not mandatory!)
photoswitches.csv (this is the original data, SMILES + outcome measurements, i.e. transition wavelengths)
morgan_fingerprints.csv (SMILES + Morgan fingerprints)
rdkit_descriptors.csv (SMILES + rdkit descriptors)
mordred_descriptors.csv (SMILES + mordred descriptors)
molecule_images (A folder containing Chemdraw style pictures of all 405 molecules)
Some ideas to get you started:

Compare & contrast methods of featurisation (fingerprints have no real-life equivalents as opposed to descriptors etc)
Attempt to predict the other measurements (e.g. Extinction coefficient)
Supervised learning:
Improve model (particularly RFs) by pruning parameters
Get most important predictors from RF, does that say anything about chemistry?
Compute scores from paper (RMSE, MAE, R^2) and compare, maybe improved?
Could even try neural networks?
Unsupervised learning:
Clustering 405 molecules should be somewhat of a tangible, tractable task, what groups of molecules are you seeing?
Paper does have advanced methods of dimensionality reduction, but maybe even PCA etc. can show stuff, too. Should be interesting to see if featurisation matters!

#### molbert
version   freeze level  finetune_
3         0               0
4         -1              2
5         -4              3
6         -2              4
7         -3              5