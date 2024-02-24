import numpy as np
import matplotlib.pyplot as plt

"""
EXAMPLE OF SCIKITLEARN LOGISTIC REGRESSION ON A LINEAR CLASSIFICATION
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

"""Train a logistic regression model"""
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr2 = LogisticRegression(C=100.0, solver='lbfgs', multi_class='multinomial')
"""
The technique used for multiclass classification, multinomial, or OvR, is
chosen automatically. In the following code example, we will use the
sklearn.linear_model.LogisticRegression class as well as the familiar fit method
to train the model on all three classes in the standardized flower training dataset.
Also, we set multi_class='ovr' for illustration purposes. We may want to compare
the results with multi_class='multinomial'. Note that the multinomial setting is
now the default choice in scikit-learn’s LogisticRegression class and recommended
in practice for mutually exclusive classes, such as those found in the Iris dataset.
Here, “mutually exclusive” means that each training example can only belong to a
single class (in contrast to multilabel classification, where a training example
can be a member of multiple classes).
"""
lr1.fit(X_train_std, y_train)
lr2.fit(X_train_std, y_train)

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

plot_decision_regions(X_combined_std, y_combined, classifier=lr1, test_idx=range(105, 150))
plt.title('The decision regions boundaries for this two-dimensional analysis using OneVersusRest technique', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_combined_std, y_combined, classifier=lr2, test_idx=range(105, 150))
plt.title('The decision regions boundaries for this two-dimensional analysis using Multinomial technique', fontsize=12, fontweight='bold')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""
Note that there exist many different algorithms for solving optimization problems.
For minimizing convex loss functions, such as the logistic regression loss, it is
recommended to use more advanced approaches than regular stochastic gradient
descent (SGD). In fact, scikit-learn implements a whole range of such optimization
algorithms, which can be specified via the solver parameter, namely, 'newton-cg',
'lbfgs', 'liblinear', 'sag', and 'saga'.
While the logistic regression loss is convex, most optimization algorithms should
converge to the global loss minimum with ease. However, there are certain advantages
of using one algorithm over the other. For example, in previous versions (for
instance, v 0.21), scikit-learn used 'liblinear' as a default, which cannot handle
the multinomial loss and is limited to the OvR scheme for multiclass classification.
However, in scikit-learn v 0.22, the default solver was changed to 'lbfgs', which
stands for the limited-memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm.
"""


"""Predictions"""
"""
The probability that training examples belong to a certain class can be computed
using the predict_proba method. For example, we can predict the probabilities of
the first three examples in the test dataset.
"""
lr1.predict_proba(X_test_std[:3, :])
"""
The highest value in the first row is approximately 0.85, which means that the first
example belongs to class 3 (Iris-virginica) with a predicted probability of 85
percent. So, as you may have already noticed, we can get the predicted class labels
by identifying the largest column in each row, for example, using NumPy’s argmax
function.
"""
lr1.predict_proba(X_test_std[:3, :]).argmax(axis=1)
"""
If you want to predict the class label of a single flower example: scikit-learn
expects a two-dimensional array as data input ;Thus, we have to convert a single
row slice into such a format first. One way to convert a single row entry into a
two-dimensional data array is to use NumPy’s reshape method to add a new dimension.
"""
lr1.predict(X_test_std[0, :].reshape(1, -1))


"""Underfitting (high-biais) vs Overfitting (high-variance)"""
"""
Overfitting is a common problem in machine learning, where a model performs well
on training data but does not generalize well to unseen data (test data). If a
model suffers from overfitting, we also say that the model has a high variance,
which can be caused by having too many parameters, leading to a model that is too
complex given the underlying data. Similarly, our model can also suffer from
underfitting (high bias), which means that our model is not complex enough to
capture the pattern in the training data well and therefore also suffers from
low performance on unseen data.
"""
"""
One way of finding a good bias-variance tradeoff is to tune the complexity of
the model via regularization. Regularization is a very useful method for handling
collinearity (high correlation among features), filtering out noise from data, and
eventually preventing overfitting.

The concept behind regularization is to introduce additional information to penalize
extreme parameter (weight) values. The most common form of regularization is
so-called L2 regularization (sometimes also called L2 shrinkage or weight decay).

The loss function for logistic regression can be regularized by adding a simple
regularization term, which will shrink the weights during model training.

Via the regularization parameter, we can then control how closely we fit the
training data, while keeping the weights small. Please note that the bias unit,
which is essentially an intercept term or negative threshold, is usually not
regularized.

The parameter C, implemented for the LogisticRegression class in scikit-learn,
comes from a convention in support vector machines. The term C is inversely
proportional to the regularization parameter. Consequently, decreasing the value
of the inverse regularization parameter, C, means that we are increasing the
regularization strength, which we can visualize by plotting the L2 regularization
path for the two weight coefficients.
"""
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
"""
Increasing the regularization strength can reduce overfitting, so we might ask
why we don’t strongly regularize all models by default. The reason is that we
have to be careful when adjusting the regularization strength. For instance,
if the regularization strength is too high and the weights coefficients approach
zero, the model can perform very poorly due to underfitting.
"""
