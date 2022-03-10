import csv
import numpy as np

# read age data from csv
# age = np.loadtxt()
with open('../input/ell319-data/ELL319_kaggle/age_final.csv', 'r') as f:
    csvreader = csv.reader(f)
    age = next(csvreader)

# sort the ages into bins of developmental categories (from paper: 9-18, 19-32, 33-51, 52-83)
for i in range(len(age)):
    n = float(age[i])
    if n < 17:
        age[i] = 1
    elif n < 33.5:
        age[i] = 2
    elif n < 54.5:
        age[i] = 3
    else:
        age[i] = 4
age = np.array(age)
# now age contains the correct labels for training the svm
# print(age)

# Read data
# how to combine all the different csv files as one input data file X
# x = file of features of different subjects (each value is a timeseries and not a single float)

# assuming x is only the ApEn data of subjects
x = np.loadtxt('../input/ell319-apen/apEn.csv', delimiter = ',')

# # split the data (80-20)
# from sklearn.model_selection import train_test_split
# train_mri, test_mri, train_lbl, test_lbl = train_test_split(x, age, test_size = 0.2, random_state=0)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# pre-process the data using PRINCIPAL COMPONENT ANALYSIS
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.decomposition import PCA

# train the SVM using pre-processed data
# https://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Performance Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

acc = 0
i = 0
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# kf = KFold(n_splits = 5)

for train_index, test_index in sss.split(x, age):
    train_mri, test_mri = x[train_index], x[test_index]
    train_lbl, test_lbl = age[train_index], age[test_index]
#     print(train_mri.shape)
#     print(test_mri.shape)
#     print(train_lbl.shape)
#     print(test_lbl.shape)
    
    # standardise data
    scaler = StandardScaler()
    scaler.fit(train_mri)
    train_mri = scaler.transform(train_mri)
    test_mri = scaler.transform(test_mri)
    
    # perform PCA
    pca = PCA(0.9)
    pca.fit(train_mri)
    train_mri = pca.transform(train_mri)
    test_mri = pca.transform(test_mri) 

    # Grid Search over C and gamma
    # param_grid = {'C': [0.3, 0.4, 0.5, 0.6], 'gamma': [0.9, 1, 1.1], 'kernel': ['rbf']} 
    # classifier = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
#     classifier = SVC(kernel='poly', degree = 3, gamma=1, C=0.1)
    classifier = SVC(kernel='rbf', gamma=10, C=1050)
    classifier.fit(train_mri, train_lbl)
    test_pred = classifier.predict(test_mri)
    
    # print(classifier.best_params_)
    t = accuracy_score(test_lbl, test_pred)
    acc += t; i += 1;
    print('Model Accuracy, iter', i, ':', t)
    print()
    print('Classification Report')
    # print(confusion_matrix(test_lbl, test_pred))
    print(classification_report(test_lbl, test_pred))

print('Final accuracy:', acc/5)
