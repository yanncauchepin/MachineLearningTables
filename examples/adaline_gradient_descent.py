"""
The ADAptive LInear NEuron (Adaline) algorithm is particularly interesting because it illustrates
the key concepts of defining and minimizing continuous loss functions.
This lays the groundwork for understanding other machine learning algorithms
for classification, such as logistic regression, support vector machines,
and multilayer neural networks, as well as linear regression models.

The key difference between the Adaline rule (also known as the Widrow-Hoff rule)
and Rosenblatt’s perceptron is that the weights are updated based on a
linear activation function rather than a unit step function like in the perceptron.

we can define the loss function as the mean squared error (MSE) ;
it becomes differentiable and convex. Thus, we can use a very simple yet powerful
optimization algorithm called gradient descent to find the weights that minimize
our loss function. The main idea behind gradient descent as climbing down a hill
until a local or global loss minimum is reached. In each iteration, we take a step
in the opposite direction of the gradient, where the step size is determined
by the value of the learning rate, as well as the slope of the gradient.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
CLASS ADALINE GRADIENT DESCENT
"""

class AdalineGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
      Mean squared error loss function values in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

"""
EXAMPLE OF ADALINEGD CLASSIFICATION ON A LINEAR CLASSIFICATION
"""


"""DATASET IRIS"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_Popular/Iris"
df = pd.read_csv(dataset_path,  header=None, encoding='utf-8')
df.tail()

"""Select Iris-setosa over the others"""
y = df.iloc[0:150, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:150, [0, 2]].values

"""Comparing two differents learning rate in the adaline algorithm learning"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - (log10) Learning rate 0.1', fontsize=12, fontweight='bold')
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001', fontsize=12, fontweight='bold')
plt.show()

"""
As we can see in the resulting loss function plots, we encountered two different
types of problems. The left chart shows what could happen if we choose a learning
rate that is too large. Instead of minimizing the loss function, the MSE becomes
larger in every epoch, because we overshoot the global minimum. On the other hand,
we can see that the loss decreases on the right plot, but the chosen learning rate,
0.0001, is so small that the algorithm would require a very large number of epochs
to converge to the global loss minimum.

Gradient descent is one of the many algorithms that benefit from feature scaling.
In this section, we will use a feature scaling method called standardization.
This normalization procedure helps gradient descent learning to converge more quickly;
however, it does not make the original dataset normally distributed. Standardization
shifts the mean of each feature so that it is centered at zero and each feature has
 a standard deviation of 1 (unit variance). This standardization technique is applied
 to each feature in our dataset.

 One of the reasons why standardization helps with gradient descent learning is
 that it is easier to find a learning rate that works well for all weights
 (and the bias). If the features are on vastly different scales, a learning rate
 that works well for updating one weight might be too large or too small to update
 the other weight equally well. Overall, using standardized features can stabilize
 the training such that the optimizer has to go through fewer steps to find a good
 or optimal solution (the global loss minimum).
"""

"""Standardization of X"""
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

"""BUILD MODEL"""

"""Application of AdalineGD algorithm with 20 epochs and a learning rate of 0.5"""
ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

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

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""Plot the gradient descent of the loss function over the epochs"""
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.title('AdalineGD with standardization - Learning rate 0.5', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

"""
Adaline has now converged after training on the standardized features.
However, note that the MSE remains non-zero even though all flower examples
were classified correctly.
"""
