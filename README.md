# ELL319 Term Paper

### Predictive Modeling for Identifying the Human Brain Developmental Stages
**Objectives**:
- Represent each time series by a single statistic i.e. approimate entropy.
- Determine the number of developmental stages by finding the optimal number of clusters k. (using elbow method)
- Perform k-means clustering to determine the optimal age boundaries.
- Develop an SVM model to predict developmental stage taking fMRI data as input.
- Improve model using stratified split, PCA and, grid search.

**Data**: Resting state fMRI data of 1096 healthy patients. The data is pre-processed to obtain the BOLD activation time series corresponding to each voxel region (160) of the brain. (160 * 190 data matrix for each patient)
