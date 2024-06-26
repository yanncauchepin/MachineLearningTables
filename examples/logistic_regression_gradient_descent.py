"""
Logistic regression is a classification model that is very easy to implement and
performs very well on linearly separable classes. It is a linear model for
binary classification. Logistic regression can be readily generalized to
multiclass settings, which is known as multinomial logistic regression,
or softmax regression. Another way to use logistic regression in multiclass
settings is via the OvA, also named One Versus Rest, technique.

To explain the main mechanics behind logistic regression as a probabilistic
model for binary classification, let’s first introduce the odds. The odds can
be written as p/(1-p), where p stands for the probability of the positive event.
The term “positive event” does not necessarily mean “good,” but refers to the
event that we want to predict.

We can then further define the logit function, which is simply the logarithm
of the odds (log-odds). The logit function takes input values in the range 0
to 1 and transforms them into values over the entire real-number range.

We assume that there is a linear relationship between the log-odds and the net
inputs. logit(p) = w1*x1 + w2*x2 + ... wn*xn + b

While the logit function maps the probability to a real-number range, we can
consider the inverse of this function to map the real-number range back to a
[0, 1] range for the probability p, with an intercept at sigmoid(0) = 0.5.
This inverse of the logit function is typically called the logistic sigmoid
function, which is sometimes simply abbreviated to sigmoid function due to
its characteristic S-shape :
sigmoid(z) = 1/(1+exp(-z)).
Here, z is the net input, the linear combination of weights, and the inputs.
"""

import matplotlib.pyplot as plt
import numpy as np

"""Plot the sigmoid function"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)
plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')
# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
plt.title('Overview of the sigmoid function', fontsize=12, fontweight='bold')
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

"""
The only difference between Adaline and logistic regression is the activation
function. In logistic regression, this activation function simply becomes
the sigmoid function.
The output of the sigmoid function is then interpreted as the probability of a
particular example belonging to class 1 given its features, x, and parameterized
by the weights and bias, w and b : sigmoid(z) = p(y=1|x,w,b)

The predicted probability can then simply be converted into a binary outcome
via a threshold function :
y_predict = 1 if sigmoid(z) > 0.5 or z > 0
y_predict = 0 otherwise

Now, let’s briefly talk about how we fit the parameters of the model, for
instance, the weights and bias unit, w and b.

We could use an optimization algorithm such as gradient ascent to maximize a
log-likelihood function. Gradient ascent works exactly the same way as gradient
descent, except that gradient ascent maximizes a function instead of minimizing it.
Alternatively, we can rewrite the log-likelihood as a loss function, L, that can
be minimized using gradient descent.
"""

"""
Plot the loss of classifying a single training example for different values
of sigmoid(z)
"""
def loss_1(z):
    return - np.log(sigmoid(z))
def loss_0(z):
    return - np.log(1 - sigmoid(z))
z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)
c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='Loss(w, b) if y=1')
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle='--', label='Loss(w, b) if y=0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('Loss(w, b)')
plt.legend(loc='best')
plt.title('The loss function used in logistic regression', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""
The resulting plot shows the sigmoid activation on the x axis in the range 0 to 1
(the inputs to the sigmoid function were z values in the range –10 to 10) and
the associated logistic loss on the y axis.

We can see that the loss approaches 0 if we correctly predict that an example
belongs to right class. However, if the prediction is wrong, the loss goes toward
infinity. The main point is that we penalize wrong predictions with an
increasingly larger loss.
"""

"""
If we were to implement logistic regression ourselves, we could simply substitute
the loss function, L, in Adaline implementation, with the new loss function. We
use this to compute the loss of classifying all training examples per epoch.
Also, we need to swap the linear activation function with the sigmoid. If we
make those changes to the Adaline code, we will end up with a working logistic
regression implementation. The following is an implementation for full-batch
gradient descent (but note that the same changes could be made to the stochastic
gradient descent version as well).
"""

"""
CLASS LOGISTIC REGRESSION GRADIENT DESCENT
"""

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.

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
      Weights after training.
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
          Training vectors, where n_examples is the
          number of examples and n_features is the
          number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : Instance of LogisticRegressionGD
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
            loss = (-y.dot(np.log(output))
                   - ((1 - y).dot(np.log(1 - output)))
                    / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

"""
EXAMPLE OF LOGISTIC REGRESSION CLASSIFICATION ON A LINEAR CLASSIFICATION
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

"""Select Iris-setosa over the others"""
y_train = np.where(y_train == 2, 1, y_train)

"""BUILD MODEL"""

"""
Application of LogisticRegressionGD algorithm with 1000 epochs and a
learning rate of 0.3
"""
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train, y_train)

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

plot_decision_regions(X=X_train, y=y_train, classifier=lrgd)
plt.title('The decision regions boundaries for this two-dimensional analysis', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
