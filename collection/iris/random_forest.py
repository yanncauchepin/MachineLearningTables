"""
Ensemble methods have gained huge popularity in applications of
machine learning during the last decade due to their good
classification performance and robustness toward overfitting.
Combining Different Models for Ensemble Learning, let’s
discuss the decision tree-based random forest algorithm,
which is known for its good scalability and ease of use.
A random forest can be considered as an ensemble of decision
trees. The idea behind a random forest is to average multiple
(deep) decision trees that individually suffer from high variance
to build a more robust model that has a better generalization
performance and is less susceptible to overfitting.
The random forest algorithm can be summarized in four simple
steps:
1. Draw a random bootstrap sample of size n (randomly choose
   n examples from the training dataset with replacement).
2. Grow a decision tree from the bootstrap sample. At each node:
   2.a. Randomly select d features without replacement.
   2.b. Split the node using the feature that provides the best
        split according to the objective function, for instance,
        maximizing the information gain.
3. Repeat steps 1-2 k times.
4. Aggregate the prediction by each tree to assign the class
   label by majority vote.

We should note one slight modification in step 2 when we are
training the individual decision trees :
Instead of evaluating all features to determine the best split
at each node, we only consider a random subset of those.
"""
"""
Although random forests don’t offer the same level of interpretability
as decision trees, a big advantage of random forests is that we
don’t have to worry so much about choosing good hyperparameter
values. We typically don’t need to prune the random forest since
the ensemble model is quite robust to noise from averaging the
predictions among the individual decision trees. The only parameter
that we need to care about in practice is the number of trees, k,
(step 3) that we choose for the random forest. Typically, the larger
the number of trees, the better the performance of the random forest
classifier at the expense of an increased computational cost.

Although it is less common in practice, other hyperparameters of the
random forest classifier that can be optimized—using techniques are
the size, n, of the bootstrap sample (step 1) and the number of
features, d, that are randomly chosen for each split (step 2a).
Via the sample size, n, of the bootstrap sample, we control the
bias-variance tradeoff of the random forest.
Decreasing the size of the bootstrap sample increases the diversity
among the individual trees since the probability that a particular
training example is included in the bootstrap sample is lower. Thus,
shrinking the size of the bootstrap samples may increase the randomness
of the random forest, and it can help to reduce the effect of
overfitting. However, smaller bootstrap samples typically result in a
lower overall performance of the random forest and a small gap between
training and test performance, but a low test performance overall.
Conversely, increasing the size of the bootstrap sample may increase
the degree of overfitting. Because the bootstrap samples, and
consequently the individual decision trees, become more similar to
one another, they learn to fit the original training dataset more
closely.

In most implementations, including the RandomForestClassifier
implementation in scikit-learn, the size of the bootstrap sample is
chosen to be equal to the number of training examples in the original
training dataset, which usually provides a good bias-variance tradeoff.
For the number of features, d, at each split, we want to choose a value
that is smaller than the total number of features in the training dataset.
A reasonable default that is used in scikit-learn and other implementations
is sqrt(m), where m is the number of features in the training dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

"""
EXAMPLE OF RANDOM FOREST SCIKITLEARN
"""

"""DATASET IRIS"""

"""Loading Iris dataset from scikitlearn"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

"""Split the dataset into separate training and test datasets"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

"""No Standardization of X"""

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

"""BUILD MODEL"""

"""Train a random forest ensemble"""
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

"""
Using the preceding code, we trained a random forest from 25
decision trees via the n_estimators parameter. By default, it
uses the Gini impurity measure as a criterion to split the nodes.
Although we are growing a very small random forest from a very
small training dataset, we used the n_jobs parameter for
demonstration purposes, which allows us to parallelize the model
training using multiple cores of our computer (here, two cores).
"""

"""Plot Mutliclass decision regions"""
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
