import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
CLASS PERCEPTRON
"""

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of
          examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        """
        Technically, we could initialize the weights to zero :
        self.w_ = np.zeros(X.shape[1])
        But if all the weights are initialized to zero, the learning rate parameter,
        eta, affects only the scale of the weight vector, not the direction.
        """
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                """Add 1 to error if update is not null"""
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

"""
EXAMPLE OF PERCEPTRON CLASSIFICATION ON A LINEAR CLASSIFICATION
"""


"""DATASET IRIS"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_Popular/Iris/iris.data"
df = pd.read_csv(dataset_path,  header=None, encoding='utf-8')
df.tail()

X = df.iloc[0:150, [0, 2]].values

"""Plot Full data with the two first feature axes"""
plt.scatter(X[:50, 0], X[:50, 1],
             color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='o', label='Versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1],
            color='green', marker='o', label='Virginica')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.title('Iris datasets with the two first feature axes', fontsize=12, fontweight='bold')
plt.show()

"""Select Iris-setosa over the others"""
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)


"""Plot One Versus All data with the two first feature axes"""
plt.scatter(X[:50, 0], X[:50, 1],
             color='red', marker='o', label='Setosa')
plt.scatter(X[50:150, 0], X[50:150, 1],
            color='blue', marker='s', label='All others')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.title('One Versus All Iris datasets with the two first feature axes', fontsize=12, fontweight='bold')
plt.show()
"""
we can see that a linear decision boundary should be sufficient
to separate setosa from others flowers. Thus, a linear classifier
such as the perceptron should be able to classify the flowers in
this dataset perfectly.
"""

"""BUILD MODEL"""

"""Plot updates over the number of epoch of learning"""
from matplotlib.ticker import MaxNLocator
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
"""Put the scale axe into integers"""
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Misclassification errors through the number of epochs of learning', fontsize=12, fontweight='bold')
plt.show()
"""
Our perceptron converged after the sixth epoch and should now
be able to classify the training examples perfectly.
"""

"""Plot One Versus All decision regions with the two first feature axes"""
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
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

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.show()

"""
Although the perceptron classified the two Iris flower classes perfectly,
convergence is one of the biggest problems of the perceptron. Rosenblatt
proved mathematically that the perceptron learning rule converges
if the two classes can be separated by a linear hyperplane.
However, if the classes cannot be separated perfectly by such a linear
decision boundary, the weights will never stop updating unless we set a
maximum number of epochs.
"""
