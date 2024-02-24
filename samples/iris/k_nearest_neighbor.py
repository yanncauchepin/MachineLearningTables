"""
KNN is a typical example of a lazy learner. It is called “lazy”
not because of its apparent simplicity, but because it doesn’t learn
a discriminative function from the training data but memorizes the
training dataset instead.

The KNN algorithm itself is fairly straightforward and can be
summarized by the following steps:
1.   Choose the number of k and a distance
2.   Find the k-nearest neighbors of the data record that we want to
     classify
3.   Assign the class label by majority vote

Based on the chosen distance metric, the KNN algorithm finds the
k examples in the training dataset that are closest (most similar)
to the point that we want to classify. The class label of the data
point is then determined by a majority vote among its k nearest
neighbors.
"""
"""
In the case of a tie, the scikit-learn implementation of the KNN
algorithm will prefer the neighbors with a closer distance to
the data record to be classified. If the neighbors have similar
distances, the algorithm will choose the class label that comes
first in the training dataset.

The right choice of k is crucial to finding a good balance between
overfitting and underfitting. We also have to make sure that we
choose a distance metric that is appropriate for the features in
the dataset. Often, a simple Euclidean distance measure is used
for real-value examples, for example, the flowers in our Iris
dataset, which have features measured in centimeters. However,
if we are using a Euclidean distance measure, it is also
important to standardize the data so that each feature
contributes equally to the distance.
"""
"""
Lastly, it is important to mention that KNN is very susceptible
to overfitting due to the curse of dimensionality. The curse of
dimensionality describes the phenomenon where the feature space
becomes increasingly sparse for an increasing number of dimensions
of a fixed-size training dataset. We can think of even the closest
neighbors as being too far away in a high-dimensional space to
give a good estimate.

We discussed the concept of regularization in the section about
logistic regression as one way to avoid overfitting. However, in
models where regularization is not applicable, such as decision
trees and KNN, we can use feature selection and dimensionality
reduction techniques to help us to avoid the curse of dimensionality.
"""

"""
EXAMPLE OF K NEAREST NEIGHBOR SCIKITLEARN
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

"""Train a K nearest neighbor model"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

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

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
