#####################################3
# Length of Stay (LOS) in hospital
# Alex's Balanced Data Sets W/ Optimal Features
#####################################

from random import Random
#import sys

from imblearn.under_sampling import RandomUnderSampler

import LOS_functions
import pandas  # to read csv file (comma separated values)
import numpy as np
#sys.stdout = open('LOS_Output.txt', 'w')
#### read csv file with Pandas
# selected columns
selcols = ['APR DRG Code', 'APR Severity of Illness Code', 'APR Risk of Mortality', 'CCS Diagnosis Code',
           'CCS Procedure Code', 'APR MDC Code', 'Length of Stay']

sample_size = 200000
training_size = int(0.8 * sample_size)

df = pandas.read_csv("LOS.csv", usecols=selcols, nrows=sample_size)  # df -data frame

# target -- LOS
y = df['Length of Stay'].replace("120 +", 120)

print("original sample size:", df.shape[0])

# print(y.value_counts().shape)
y = np.array(y).astype("float32")

### drop samples in which LOS>bad
bad = 15
df.drop(df[(y > bad)].index, inplace=True)
df = df.reset_index(drop=True)
y = y[y<=bad]

# features
X = df[['APR DRG Code', 'APR Severity of Illness Code', 'APR Risk of Mortality',
        'CCS Diagnosis Code', 'CCS Procedure Code', 'APR MDC Code']]

X, y, X_train, y_train, X_test, y_test = LOS_functions.dataPreprocessing(
    189656, training_size, X, y)

print("\n-------Training and testing results-------")

print("\n--Classification--")
"""
>20: 7205
15-19: 4375 (11580
10-14: 10804 (22384)
5-9: 40449 (62833)
4: 21597 (84430 - 62833)
3: 38651 (123081 - 84430)
2: 48329 (171473 - 123081)
1: 28527 (200000 - 171473)

"""
grp = [4, 8]
#grp = [4]

LOS_functions.grouping(189656, y, grp)

from collections import Counter
from sklearn.datasets import make_classification
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


undersample = RandomUnderSampler(sampling_strategy='not minority')
X_over, y_over = undersample.fit_resample(X_train, y_train)
print("\nUndersampled Training Data")
print(Counter(y_over))

X_testOver, y_testOver = undersample.fit_resample(X_test, y_test)
print("\nUndersampled Testing Data")
print(Counter(y_testOver))

oversample = RandomOverSampler(sampling_strategy='not majority')
X_under, y_under = oversample.fit_resample(X_train, y_train)
print("Oversampled Training Data")
print(Counter(y_under))

X_testUnder, y_testUnder = oversample.fit_resample(X_test, y_test)
print("Oversampled Testing Data")
print(Counter)

LOS_functions.logisticRegression(X_train, y_train, X_test, y_test)
LOS_functions.decisionTree(X_train, y_train, X_test, y_test)
LOS_functions.randomForest(X_train, y_train, X_test, y_test)
LOS_functions.gradientBoosting(X_train, y_train, X_test, y_test)
LOS_functions.AdaBoost(X_train, y_train, X_test, y_test)
#LOS_functions.MLPClassifier(X,y)


###### Results
"""
# 1-3, 4 -
--Classification--
Counter({0.0: 115570, 1.0: 84430})

## Logistic Regression ##
Accuracy:  0.73615
[[19286  5568]
 [ 4986 10160]]

## Decision Tree ##
Accuracy:  0.751525
[[19026  5828]
 [ 4111 11035]]

## Random Forest ##
Accuracy:  0.756125
[[19193  5661]
 [ 4094 11052]]

## Gradient Boosting Classifier ##
Accuracy:  0.761475
[[20794  4060]
 [ 5481  9665]]

## AdaBoost Classifier ##
Accuracy:  0.75445
[[20816  4038]
 [ 5784  9362]]
"""
