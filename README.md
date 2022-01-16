# PhotoSwitchProject - A synthetic chemist's guide through organic photoswitch chemical space

1. Featurization.ipynb - Notebook explaining how we generated different input features
2. Feature_selection.ipynb - Notebook explaining the EDA & feature selection process
3. Model_selection.ipynb - Notebook explaining how we tried out different models
4. Screening.ipynb - Notebook explaining how we generated a photoswitch molecular library and select the promising candidates
5. Optimization.ipynb - Notebook explaining how we optimized over chemical space using gradients & a continous representation

### INTRO
Photoswitches are molecules with high potential in various innovative areas such a solar energy storage, locally effective drugs and photoelectronics. since
the chemical space of all possible photomolecules is quite extensive, it would be very much resourcefull for researchers in this area to have an effective way
of selecting photoswitches that might be worth synthesizing and testing out. Therefore we aim to build a virtual screening tool in order to select the most promising photoswitches from a large library of possible molecules.
For now we specifically focussed on extracting photoswitches with a desired transition wavelength, according to a synthetical chemists wishes.

### RESULTS
- It was found that a set of mordred descriptors & deep learned MolBERT fingerprints performed very well when training a model to predict transition wavelenghts.
- A Gaussian Process Regression model with the Tanimoto Kernel function was found to fit the data the best, which might be due to the small amount of data worked with. Plus the GPR model also allows us to model uncertainty in our predictions.
- We performed virtual screening rounds on photoswitches for transition wavelengths of 450 nm and 650 nm, which we unfortunately haven't been able to check with DFT yet.
- We took the most promising molecule from our screening round for 650 nm and optimized it's transition wavelength using a Junction-Tree VAE, although we haven't been able to verify the photoswitch characted of the molecule this model produced.