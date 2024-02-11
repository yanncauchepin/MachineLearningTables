"""
In Adaline Descent Gradient, we learned how to minimize a loss function by
taking a step in the opposite direction of the loss gradient that is
calculated from the whole training dataset ; this is why this approach is
sometimes also referred to as full batch gradient descent. Now imagine that
we have a very large dataset with millions of data points, which is not uncommon
in many machine learning applications. Running full batch gradient descent can
be computationally quite costly in such scenarios, since we need to reevaluate
the whole training dataset each time we take one step toward the global minimum.

A popular alternative to the batch gradient descent algorithm is stochastic
gradient descent (SGD), which is sometimes also called iterative or online
gradient descent. Instead of updating the weights based on the sum of the
accumulated errors over all training examples, we update the parameters
incrementally for each training example.

Although SGD can be considered as an approximation of gradient descent,
it typically reaches convergence much faster because of the more frequent
weight updates. Since each gradient is calculated based on a single training
example, the error surface is noisier than in gradient descent, which can also
have the advantage that SGD can escape shallow local minima more readily if
we are working with nonlinear loss functions.

To obtain satisfying results via SGD, it is important to present training data
in a random order ; also, we want to shuffle the training dataset for every
epoch to prevent cycles.

Another advantage of SGD is that we can use it for online learning. In online
learning, our model is trained on the fly as new training data arrives. This
is especially useful if we are accumulating large amounts of data, for example,
customer data in web applications. Using online learning, the system can
immediately adapt to changes, and the training data can be discarded after
updating the model if storage space is an issue.

A compromise between full batch gradient descent and SGD is so-called mini-batch
gradient descent. Mini-batch gradient descent can be understood as applying full
batch gradient descent to smaller subsets of the training data, for example,
32 training examples at a time. The advantage over full batch gradient descent
is that convergence is reached faster via mini-batches because of the more frequent
weight updates. Furthermore, mini-batch learning allows us to replace the for loop
over the training examples in SGD with vectorized operations leveraging concepts
from linear algebra (for example, implementing a weighted sum via a dot product),
which can further improve the computational efficiency of our learning algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
CLASS ADALINE STOCHASTIC GRADIENT DESCENT
"""

class AdalineSGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent
        cycles.
    random_state : int
        Random number generator seed for random weight
        initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

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

        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.5, 1, 0)



"""
EXAMPLE OF ADALINESGD CLASSIFICATION ON A LINEAR CLASSIFICATION
"""


"""DATASET IRIS"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_Popular/Iris"
df = pd.read_csv(dataset_path,  header=None, encoding='utf-8')
df.tail()

"""Select Iris-setosa over the others"""
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:150, [0, 2]].values

"""Standardization of X"""
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

"""BUILD MODEL"""

"""Application of AdalineSGD algorithm with 15 epochs and a learning rate of 0.01"""
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

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

plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""Plot the gradient descent of the loss function over the epochs"""
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.title('AdalineSGD with standardization - Learning rate 0.01', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

"""
he average loss goes down pretty quickly, and the final decision boundary after
15 epochs looks similar to the batch gradient descent AdalineGD in this example.
If we want to update our model, for example, in an online learning scenario
with streaming data, we could simply call the partial_fit method on individual
training examples—for instance :
ada_sgd.partial_fit(X_std[0, :], y[0]).
"""
