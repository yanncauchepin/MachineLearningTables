"""
Common solutions to reduce the generalization of overfitting are
as follows:
-   Collect more training data
-   Introduce a penalty for complexity via regularization
-   Choose a simpler model with fewer parameters
-   Reduce the dimensionality of the data

we will look at common ways to reduce overfitting by regularization
and dimensionality reduction via feature selection, which leads to
simpler models by requiring fewer parameters to be fitted to the data.

L1 regularization replaced the square of the weights with the sum
of the absolute values of the weights. In contrast to L2 regularization,
L1 regularization usually yields sparse feature vectors, and most
feature weights will be zero. Sparsity can be useful in practice if
we have a high-dimensional dataset with many features that are irrelevant,
especially in cases where we have more irrelevant dimensions than
training examples. In this sense, L1 regularization can be understood
as a technique for feature selection. L2 regularization adds a penalty
term to the loss function that effectively results in less extreme
weight values compared to a model trained with an unregularized loss
function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

"""Standardization with StandardScaler"""
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


"""FEATURE SELECTION"""

"""FEATURE ASSESSING WITH RANDOM FOREST"""

"""
Another useful approach for selecting relevant features from a
dataset is using a random forest, an ensemble technique. Using
a random forest, we can measure the feature importance as the
averaged impurity decrease computed from all decision trees in
the forest, without making any assumptions about whether our data
is linearly separable or not. Conveniently, the random forest
implementation in scikit-learn already collects the feature
importance values for us so that we can access them via the
feature_importances_ attribute after fitting a
RandomForestClassifier.

We train a forest of 500 trees on the Wine dataset and rank the
13 features by their respective importance measures. Remember that
we don’t need to use standardized or normalized features in
tree-based models.

We can create a plot that ranks the different features in the
Wine dataset by their relative importance ; Note that the feature
importance values are normalized so that they sum up to 1.0.
"""

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

"""
We can conclude that the proline and flavonoid levels, the color
intensity, the OD280/OD315 diffraction, and the alcohol concentration
of wine are the most discriminative features in the dataset based
on the average impurity decrease in the 500 decision trees.
Interestingly, two of the top-ranked features in the plot are also
in the three-feature subset selection from the SBS algorithm that we
implemented in the previous section (alcohol concentration and
OD280/OD315 of diluted wines).

However, as far as interpretability is concerned, the random forest
technique comes with an important gotcha that is worth mentioning.
If two or more features are highly correlated, one feature may be
ranked very highly while the information on the other feature(s)
may not be fully captured. On the other hand, we don’t need to be
concerned about this problem if we are merely interested in the
predictive performance of a model rather than the interpretation
of feature importance values.
"""

"""FEATURE ASSESSING WITH SCIKITLEARN SELECTFROMMODEL"""

"""
To conclude this section about feature importance values and random
forests, it is worth mentioning that scikit-learn also implements a
SelectFromModel object that selects features based on a user-specified
threshold after model fitting, which is useful if we want to use
the RandomForestClassifier as a feature selector and intermediate
step in a scikit-learn Pipeline object, which allows us to connect
different preprocessing steps with an estimator. For example, we
could set the threshold to 0.1 to reduce the dataset to the five
most important features using the following code.
"""

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold',
      'criterion:', X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
