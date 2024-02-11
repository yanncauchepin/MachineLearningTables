"""
Support vector machine (SVM) can be considered an extension of the perceptron.
Using the perceptron algorithm, we minimized misclassification errors.
However, in SVMs, our optimization objective is to maximize the margin. The margin
is defined as the distance between the separating hyperplane (decision boundary)
and the training examples that are closest to this hyperplane, which are the
so-called support vectors.

The rationale behind having decision boundaries with large margins is that they
tend to have a lower generalization error, whereas models with small margins are
more prone to overfitting.

In concepts behind the maximum-margin classification, the so-called slack variable,
was introduced by Vladimir Vapnik in 1995 and led to the so-called soft-margin
classification. The motivation for introducing the slack variable was that the
linear constraints in the SVM optimization objective need to be relaxed for
nonlinearly separable data to allow the convergence of the optimization in the
presence of misclassifications, under appropriate loss penalization.

The use of the slack variable, in turn, introduces the variable, which is commonly
referred to as C in SVM contexts. We can consider C as a hyperparameter for
controlling the penalty for misclassification. Large values of C correspond to
large error penalties, whereas we are less strict about misclassification errors
if we choose smaller values for C. We can then use the C parameter to control
the width of the margin and therefore tune the bias-variance tradeoff.
This concept is related to regularization, where decreasing the value of C
increases the bias (underfitting) and lowers the variance (overfitting) of the model.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
EXAMPLE OF SCIKITLEARN SUPPORT VECTOR MACHINE ON A LINEAR CLASSIFICATION
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

"""Standardization of X"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""BUILD MODEL"""

"""Train a support vector machine model"""
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

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

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


"""Logistic regression versus SVMs"""
"""
In practical classification tasks, linear logistic regression and linear SVMs
often yield very similar results. Logistic regression tries to maximize the
conditional likelihoods of the training data, which makes it more prone to outliers
than SVMs, which mostly care about the points that are closest to the decision
boundary (support vectors). On the other hand, logistic regression has the advantage
of being a simpler model and can be implemented more easily, and is mathematically
easier to explain. Furthermore, logistic regression models can be easily updated,
which is attractive when working with streaming data.
"""
