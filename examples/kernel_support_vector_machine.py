"""
Another reason why SVMs enjoy high popularity among machine learning
practitioners is that they can be easily kernelized to solve
nonlinear classification problems.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
EXAMPLE OF SCIKITLEARN KERNEL SUPPORT VECTOR MACHINE ON A NONLINEAR CLASSIFICATION
"""

"""
Obviously, we would not be able to separate the examples from the
positive and negative class very well using a linear hyperplane
as a decision boundary via the linear logistic regression or
linear SVM model that we discussed in earlier sections.

The basic idea behind kernel methods for dealing with such linearly
inseparable data is to create nonlinear combinations of the
original features to project them onto a higher-dimensional space
via a mapping function, where the data becomes linearly separable.
We can transform a two-dimensional dataset into a new
three-dimensional feature space, where the classes become
separable via a projection.

This allows us to separate the two classes shown in the plot via
a linear hyperplane that becomes a nonlinear decision boundary
if we project it back onto the original feature space.
"""

"""NONLINEAR CREATED XOR DATASET"""

"""Building XOR Dataset"""
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

"""¨Plot XOR Dataset"""
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue', marker='s',
            label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato', marker='o',
            label='Class 0')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.title('Overview of XOR Dataset', fontsize=12, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

"""
To solve a nonlinear problem using an SVM, we would transform the
training data into a higher-dimensional feature space via a
mapping function, and train a linear SVM model to classify the
data in this new feature space. Then, we could use the same
mapping function, to transform new, unseen data to classify it
using the linear SVM model.

However, one problem with this mapping approach is that the
construction of the new features is computationally very expensive,
especially if we are dealing with high-dimensional data. This is
where the so-called kernel trick comes into play.

To save the expensive step of calculating the dot product between
two points explicitly, we define a so-called kernel function.
One of the most widely used kernels is the radial basis function
(RBF) kernel, which can simply be called the Gaussian kernel.
"""

"""BUILD MODEL"""

"""
We can train a kernel SVM that is able to draw a nonlinear decision
boundary that separates the XOR data well. Here, we simply use the
SVC class from scikit-learn that we imported earlier and replace
the kernel='linear' parameter with kernel='rbf'.
"""

"""Train a support vector machine model"""
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

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

plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""
The parameter gamma which we set to gamma=0.1, can be
understood as a cut-off parameter for the Gaussian sphere.
If we increase the value for gamma, we increase the influence
or reach of the training examples, which leads to a tighter
and bumpier decision boundary. To get a better understanding
of gamma, let’s apply an RBF kernel SVM to our Iris flower
dataset.
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


"""BUILD MODEL"""

"""Train a support vector machine model with a gamma of 0.2"""
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

"""Plot Mutliclass decision regions"""
plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105, 150))
plt.title('The decision regions boundaries with a gamma of 0.2', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""Train a support vector machine model with a gamma of 100"""
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

"""Plot Mutliclass decision regions"""
plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.title('The decision regions boundaries with a gamma of 100', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""
Although the model fits the training dataset very well, such a
classifier will likely have a high generalization error on unseen
data. This illustrates that the gamma parameter also plays an
important role in controlling overfitting or variance when the
algorithm is too sensitive to fluctuations in the training dataset.
"""
