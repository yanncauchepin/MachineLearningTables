import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""GRADIENT BOOSTING"""
"""
Gradient boosting is another variant of the boosting concept, that is, successively
training weak learners to create a strong ensemble. Gradient boosting is an extremely
important topic because it forms the basis of popular machine learning algorithms
such as XGBoost.
"""
"""
Fundamentally, gradient boosting is very similar to AdaBoost. AdaBoost trains
decision tree stumps based on errors of the previous decision tree stump. In
particular, the errors are used to compute sample weights in each round as well
as for computing a classifier weight for each decision tree stump when combining
the individual stumps into an ensemble. We stop training once a maximum number of
iterations (decision tree stumps) is reached. Like AdaBoost, gradient boosting
fits decision trees in an iterative fashion using prediction errors. However,
gradient boosting trees are usually deeper than decision tree stumps and have
typically a maximum depth of 3 to 6 (or a maximum number of 8 to 64 leaf nodes).
Also, in contrast to AdaBoost, gradient boosting does not use the prediction
errors for assigning sample weights; they are used directly to form the target
variable for fitting the next tree. Moreover, instead of having an individual
weighting term for each tree, like in AdaBoost, gradient boosting uses a global
learning rate that is the same for each tree.
In essence, gradient boosting builds a series of trees, where each tree is fit
on the error the difference between the label and the predicted value of the
previous tree. In each round, the tree ensemble improves as we are nudging each
tree more in the right direction via small updates. These updates are based on
a loss gradient, which is how gradient boosting got its name.
The following steps will introduce the general algorithm behind gradient boosting.
After illustrating the main steps, we will dive into some of its parts in more
detail and walk through a hands-on example in the next subsections.
"""

"""DATASET WINE SCIKITLEARN"""

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None)
"""Or"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_PopularLearning/Wine/wine.data"
df_wine = pd.read_csv(dataset_path, header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash',
                   'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
# drop 1 class
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =\
           train_test_split(X, y,
                            test_size=0.2,
                            random_state=1,
                            stratify=y)

"""XGBOOST"""

from sklearn.metrics import accuracy_score
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01,
                          max_depth=4, random_state=1,
                          use_label_encoder=False)
gbm = model.fit(X_train, y_train)
y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)
gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)
print(f'XGboost train/test accuracies '
      f'{gbm_train:.3f}/{gbm_test:.3f}')
