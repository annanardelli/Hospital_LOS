#####################################3
# Length of Stay (LOS) in hospital
# One Hot Encoding Applied - 2 group RF ~76% accuracy
#####################################

import LOS_functions_OHE
import pandas as pd  # to read csv file (comma separated values)
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

#### read csv file with Pandas
# selected columns
selcols = ['APR DRG Code', 'APR Severity of Illness Code', 'APR Risk of Mortality', 'CCS Diagnosis Code',
           'CCS Procedure Code', 'APR MDC Code', 'Length of Stay']

sample_size = 200000
training_size = int(0.8 * sample_size)

df = pd.read_csv("LOS.csv", usecols=selcols, nrows=sample_size, delimiter=',')  # df -data frame

df = df.loc[df['APR Risk of Mortality'].notna(), selcols]  # Takes all columns, ignores null values of Risk of Mortality

# df = LOS_functions_OHE.replaceByFrequency(df, 'APR DRG Code', 300, 999)
# df = LOS_functions_OHE.replaceByFrequency(df, 'CCS Diagnosis Code', 200, 998)
# df = LOS_functions_OHE.replaceByFrequency(df, 'CCS Procedure Code', 150, 997)
# df = LOS_functions_OHE.dropByFrequency(df, 'CCS Diagnosis Code', 500)
# df = LOS_functions_OHE.dropByFrequency(df, 'CCS Procedure Code', 250)

X = df.drop('Length of Stay', axis='columns')  # Drops LOS from dataframe

# target -- LOS
y = df[['Length of Stay']].replace("120 +", 120)  # Assigns LOS to y dataframe

print("original sample size:", df.shape[0])

# print(y.value_counts().shape)
y = np.array(y).astype("float32")

### drop samples in which LOS>bad
bad = 15
df.drop(df[(y > bad)].index, inplace=True)
df = df.reset_index(drop=True)
y = y[y <= bad]

# frequency1 = df['APR DRG Code'].value_counts()
# print(frequency1)

# features
X = df[['APR DRG Code', 'APR Severity of Illness Code', 'APR Risk of Mortality',
        'CCS Diagnosis Code', 'CCS Procedure Code', 'APR MDC Code']]

# Change shape of y to 1d array
y = np.ravel(y, order="C")

# One Hot Encode X and Split X and y for training/testing
X_train, y_train, X_test, y_test = LOS_functions_OHE.dataPreprocessing(X, y)

# print("\n-------Training and testing results-------")

# print("\n--Classification--")


grp = [4, 8]
# grp = [4]

LOS_functions_OHE.grouping(int(df.shape[0] * .2), y_test, grp)
LOS_functions_OHE.grouping(int(df.shape[0] * .8), y_train, grp)
print("Training Shape:")
print(X_train.shape)

# for x in range(5):
LOS_functions_OHE.logisticRegression(X_train, y_train, X_test, y_test)
LOS_functions_OHE.decisionTree(X_train, y_train, X_test, y_test)

LOS_functions_OHE.randomForest(X_train, y_train, X_test, y_test)

LOS_functions_OHE.gradientBoosting(X_train, y_train, X_test, y_test)

LOS_functions_OHE.AdaBoost(X_train, y_train, X_test, y_test)
LOS_functions_OHE.MLPClassifier(X,y)

'''
# SMOTE START
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("SMOTE Training Data")
print(Counter(y_smote))

X_testSmote, y_testSmote = smote.fit_resample(X_test, y_test)
print("SMOTE Testing Data")
print(Counter(y_testSmote))

print("Training and Testing Data")

LOS_functions_OHE.logisticRegression(X_smote, y_smote, X_testSmote, y_testSmote)
# for x in range(5):
LOS_functions_OHE.decisionTree(X_smote, y_smote, X_testSmote, y_testSmote)
# for x in range(5):
#LOS_functions_OHE.randomForest(X_smote, y_smote, X_testSmote, y_testSmote)
LOS_functions_OHE.gradientBoosting(X_smote, y_smote, X_testSmote, y_testSmote)
LOS_functions_OHE.AdaBoost(X_smote, y_smote, X_testSmote, y_testSmote)
# SMOTE END


print("End")
'''
'''
## Logistic Regression ##
Accuracy:  0.7729
[[4388 1321]
 [ 950 3341]]

## Decision Tree ##
Accuracy:  0.7517
[[4366 1343]
 [1140 3151]]

## Random Forest ##
Accuracy:  0.7629
[[4430 1279]
 [1092 3199]]

## Gradient Boosting Classifier ##
Accuracy:  0.7722
[[4699 1010]
 [1268 3023]]

## AdaBoost Classifier ##
Accuracy:  0.7668
[[4641 1068]
 [1264 3027]]
 '''