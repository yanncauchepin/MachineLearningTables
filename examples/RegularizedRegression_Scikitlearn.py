"""
Regularization is one approach to tackling the problem of overfitting by adding
additional information and thereby shrinking the parameter values of the model
to induce a penalty against complexity. The most popular approaches to regularized
linear regression are the so-called ridge regression, least absolute shrinkage
and selection operator (LASSO), and elastic net.
"""

"""RIDGE REGRESSION"""
"""
Ridge regression is an L2 penalized model where we simply add the squared sum of
the weights to the MSE loss function. By increasing the value of hyperparameter,
we increase the regularization strength and thereby shrink the weights of our model.
"""

"""LASSO REGRESSION"""
"""
An alternative approach that can lead to sparse models is LASSO. Depending on the
regularization strength, certain weights can become zero, which also makes LASSO
useful as a supervised feature selection technique. Here, the L1 penalty for LASSO
is defined as the sum of the absolute magnitudes of the model weights.
However, a limitation of LASSO is that it selects at most n features if m > n,
where n is the number of training examples. This may be undesirable in certain
applications of feature selection. In practice, however, this property of LASSO
is often an advantage because it avoids saturated models. The saturation of a
model occurs if the number of training examples is equal to the number of features,
which is a form of overparameterization. As a consequence, a saturated model can
always fit the training data perfectly but is merely a form of interpolation and
thus is not expected to generalize well.
"""

"""ELASTIC NET"""
"""
A compromise between ridge regression and LASSO is elastic net, which has an L1
penalty to generate sparsity and an L2 penalty such that it can be used for selecting
more than n features if m > n.
"""

"""REGULARIZED REGRESSION SCIKITLEARN"""
"""
Those regularized regression models are all available via scikit-learn, and their
usage is similar to the regular regression model except that we have to specify the
regularization strength via the parameter , for example, optimized via k-fold
cross-validation.
"""

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)

from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
"""
For example, if we set l1_ratio to 1.0, the ElasticNet regressor would be equal
to LASSO regression.
"""
