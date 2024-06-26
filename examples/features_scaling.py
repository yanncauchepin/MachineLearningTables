"""
Feature scaling is a crucial step in our preprocessing pipeline that
can easily be forgotten. Decision trees and random forests are two of
the very few machine learning algorithms where we don’t need to worry
about feature scaling. Those algorithms are scale-invariant. However,
the majority of machine learning and optimization algorithms behave
much better if features are on the same scale.
"""

import pandas as pd
import numpy as np

"""DATASET WINE"""

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None)
"""Or"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_PopularLearning/Wine/wine.data"
df_wine = pd.read_csv(dataset_path, header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

"""Partition Training Test"""
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

"""FEATURE SCALING"""

"""
There are two common approaches to bringing different features onto
the same scale: normalization and standardization.

Most often, normalization refers to the rescaling of the features to
a range of [0, 1], which is a special case of min-max scaling.
"""

"""Normalization with MinMaxScaler"""
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

"""
Although normalization via min-max scaling is a commonly used
technique that is useful when we need values in a bounded interval,
standardization can be more practical for many machine learning
algorithms, especially for optimization algorithms such as gradient
descent. The reason is that many linear models, such as the logistic
regression and SVM, initialize the weights to 0 or small random values
close to 0. Using standardization, we center the feature columns at
mean 0 with standard deviation 1 so that the feature columns have
the same parameters as a standard normal distribution (zero mean and
unit variance), which makes it easier to learn the weights. However,
we shall emphasize that standardization does not change the shape of
the distribution, and it does not transform non-normally distributed
data into normally distributed data. In addition to scaling data
such that it has zero mean and unit variance, standardization maintains
useful information about outliers and makes the algorithm less sensitive
to them in contrast to min-max scaling, which scales the data to a
limited range of values.
"""

"""Comparing Normalization and standardization"""
ex = np.array([0, 1, 2, 3, 4, 5])
standardized = (ex - ex.mean()) / ex.std()
normalized = (ex - ex.min()) / (ex.max() - ex.min())
table = np.column_stack((ex, standardized, normalized))
print("ex\tStandardized\tNormalized")
for row in table:
    print("\t".join(str(value) for value in row))

"""Standardization with StandardScaler"""
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

"""RobustScaler"""
"""
Other, more advanced methods for feature scaling are available
from scikit-learn, such as RobustScaler. RobustScaler is especially
helpful and recommended if we are working with small datasets that
contain many outliers. Similarly, if the machine learning algorithm
applied to this dataset is prone to overfitting, RobustScaler can be
a good choice. Operating on each feature column independently,
RobustScaler removes the median value and scales the dataset according
to the 1st and 3rd quartile of the dataset.
"""

from sklearn.preprocessing import RobustScaler
rbtsc = RobustScaler()
X_train_rbt = rbtsc.fit_transform(X_train)
X_test_rbt = rbtsc.transform(X_test)
