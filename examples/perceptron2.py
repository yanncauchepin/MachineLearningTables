import numpy as np
import matplotlib.pyplot as plt

"""
EXAMPLE OF SCIKITLEARN PERCEPTRON ON A LINEAR CLASSIFICATION
"""

"""DATASET IRIS"""

"""Loading Iris dataset from scikitlearn"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))
"""
np.unique(y). This function returned the three unique class labels stored in
iris.target, and as we can see, the Iris flower class names, Iris-setosa,
Iris-versicolor, and Iris-virginica, are already stored as integers (here: 0, 1, 2).
"""

"""Split the dataset into separate training and test datasets"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
"""
stratify = y. In this context, stratification means that the
train_test_split method returns training and test subsets that have the
same proportions of class labels as the input dataset.
random_state = 1. Also, we used the random_state parameter to ensure the
reproducibility of the initial shuffling of the training dataset after each epoch.
"""
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

"""Standardization of X"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
"""
We loaded the StandardScaler class from the preprocessing module and initialized
a new StandardScaler object that we assigned to the sc variable. Using the fit
method, StandardScaler estimated the parameters, sample mean and standard deviation,
for each feature dimension from the training data. By calling the transform method,
we then standardized the training data using those estimated parameters.
Note that we used the same scaling parameters to standardize the test dataset so
that both the values in the training and test dataset are comparable with one another.
"""

"""BUILD MODEL"""

"""Train a perceptron model"""
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

"""Prediction"""
y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())
"""
Classification error versus accuracy. Instead of the misclassification error,
many machine learning practitioners report the classification accuracy of a model,
which is simply calculated as follows:
1–error = accuracy. Here error is 1/45 = 2.2 %.
"""

"""Performance metrics"""
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
"""
Alternatively, each classifier in scikit-learn has a score method, which computes
a classifier’s prediction accuracy by combining the predict call with accuracy_score.
"""
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))


"""Plot One Versus All decision regions"""
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""
Remember that the perceptron algorithm never converges on datasets that
aren’t perfectly linearly separable, which is why the use of the perceptron
algorithm is typically not recommended in practice.
"""
