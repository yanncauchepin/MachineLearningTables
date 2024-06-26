"""
In boosting, the ensemble consists of very simple base classifiers, also often
referred to as weak learners, which often only have a slight performance advantage
over random guessing—a typical example of a weak learner is a decision tree stump.
The key concept behind boosting is to focus on training examples that are hard
to classify, that is, to let the weak learners subsequently learn from misclassified
training examples to improve the performance of the ensemble.

In contrast to bagging, the initial formulation of the boosting algorithm uses
random subsets of training examples drawn from the training dataset without
replacement ; the original boosting procedure can be summarized in the following
four key steps:

1.  Draw a random subset (sample) of training examples, d1, without replacement
    from the training dataset, D, to train a weak learner, C1.
2.  Draw a second random training subset, d2, without replacement from the training
    dataset and add 50 percent of the examples that were previously misclassified
    to train a weak learner, C2.
3.  Find the training examples, d3, in the training dataset, D, which C1 and C2
    disagree upon, to train a third weak learner, C3.
4.  Combine the weak learners C1, C2, and C3 via majority voting.

Boosting can lead to a decrease in bias as well as variance compared to bagging
models. In practice, however, boosting algorithms such as AdaBoost are also known
for their high variance, that is, the tendency to overfit the training data.

In contrast to the original boosting procedure described here, AdaBoost uses the
complete training dataset to train the weak learners, where the training examples
are reweighted in each iteration to build a strong classifier that learns from
the mistakes of the previous weak learners in the ensemble.
"""
"""
To walk through the AdaBoost illustration step by step, we will start with
subfigure 1, which represents a training dataset for binary classification where
all training examples are assigned equal weights. Based on this training dataset,
we train a decision stump (shown as a dashed line) that tries to classify the
examples of the two classes (triangles and circles), as well as possibly minimizing
the loss function (or the impurity score in the special case of decision tree
ensembles).
For the next round (subfigure 2), we assign a larger weight to the two previously
misclassified examples (circles). Furthermore, we lower the weight of the correctly
classified examples. The next decision stump will now be more focused on the training
examples that have the largest weights—the training examples that are supposedly
hard to classify.
The weak learner shown in subfigure 2 misclassifies three different examples from
the circle class, which are then assigned a larger weight, as shown in subfigure 3.
Assuming that our AdaBoost ensemble only consists of three rounds of boosting, we
then combine the three weak learners trained on different reweighted training
subsets by a weighted majority vote, as shown in subfigure 4.
Now that we have a better understanding of the basic concept of AdaBoost, let’s
take a more detailed look at the algorithm using pseudo code :
1.  Set the weight vector, w, to uniform weights, where somme(w_i) = 1
2.  For j in m boosting rounds, do the following :
    a)  Train a weighted weak learner : Cj = train(X, y, w).
    b)  Predict class labels : y^ = predict(c_j, X).
    c)  Compute the weighted error rate : error = w . (y^ # y).
    d)  Compute the coefficient : a_j = 0,5 x log ((1 - error) / error).
    e)  Update the weights : error = error x exp(-a_j x y^ x y).
    f)  Normalize the weights to sum to 1 : w = w / somme(w_i).
3.  Compute the final prediction: y^ = (somme (a_j x predict (c_j, X)) > 0).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""ADABOOST ENSEMBLE CLASSIFIER"""

"""
Although the AdaBoost algorithm seems to be pretty straightforward, let’s walk
through a more concrete example using a training dataset consisting of 10 training
examples :
"""

y = np.array([1, 1, 1, -1, -1, -1,  1,  1,  1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

correct = (y == yhat)
weights = np.full(10, 0.1)
print(weights)

epsilon = np.mean(~correct)
print(epsilon)

"""
Note that correct is a Boolean array consisting of True and False values where
True indicates that a prediction is correct. Via ~correct, we invert the array
such that np.mean(~correct) computes the proportion of incorrect predictions
(True counts as the value 1 and False as 0), that is, the classification error.
"""

alpha_j = 0.5 * np.log((1-epsilon) / epsilon)
print(alpha_j)

update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
print(update_if_correct)

update_if_wrong = 0.1 * np.exp(-alpha_j * 1 * -1)
print(update_if_wrong)

weights = np.where(correct == 1, update_if_correct, update_if_wrong)
print(weights)

normalized_weights = weights / np.sum(weights)
print(normalized_weights)

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

"""
we will encode the class labels into binary format and split the dataset into
80 percent training and 20 percent test datasets.
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test =\
           train_test_split(X, y,
                            test_size=0.2,
                            random_state=1,
                            stratify=y)

"""ADABOOST ENSEMBLE CLASSIFIER SCIKITLEARN"""

"""
Via the base_estimator attribute, we will train the AdaBoostClassifier on 500
decision tree stumps.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f'AdaBoost train/test accuracies '
      f'{ada_train:.3f}/{ada_test:.3f}')

"""
Here, you can see that the AdaBoost model predicts all class labels of the training
dataset correctly and also shows a slightly improved test dataset performance
compared to the decision tree stump.
Let’s check what the decision regions look like :
"""

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Decision tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue',
                       marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='green',
                       marker='o')
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,
         s='Alcohol',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

"""
By looking at the decision regions, you can see that the decision boundary of the
AdaBoost model is substantially more complex than the decision boundary of the
decision stump.
It is worth noting that ensemble learning increases the computational complexity
compared to individual classifiers. In practice, we need to think carefully about
whether we want to pay the price of increased computational costs for an often
relatively modest improvement in predictive performance.
"""
