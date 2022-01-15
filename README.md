# PhotoSwitchProject - A synthetic chemist's guide through organic photoswitch chemical space

1. Featurization.ipynb - Notebook explaining how we generated different input features
2. Feature_selection.ipynb - Notebook explaining the EDA & feature selection process
3. Model_selection.ipynb - Notebook explaining how we tried out different models
4. Screening.ipynb - Notebook explaining how we generated a photoswitch molecular library and select the promising candidates
5. Optimization.ipynb - Notebook explaining how we optimized over chemical space using gradients & a continous representation

## INTRO

Photoswitches are molecules with high potential in various innovative areas such a solar energy storage, locally effective drugs and photoelectronics. since
the chemical space of all possible photomolecules is quite extensive, it would be very much resourcefull for researchers in this area to have an effective way
of selecting photoswitches that might be worth synthesizing and testing out. Therefore we aim to build a virtual screening tool in order to select the most promising photoswitches from a large library of possible molecules.