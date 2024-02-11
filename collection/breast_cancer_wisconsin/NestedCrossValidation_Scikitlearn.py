"""
Using k-fold cross-validation in combination with grid search or randomized search
is a useful approach for fine-tuning the performance of a machine learning model by
varying its hyperparameter values, as we saw in the previous subsections. If we want
to select among different machine learning algorithms, though, another recommended
approach is nested cross-validation.

In nested cross-validation, we have an outer k-fold cross-validation loop to split
the data into training and test folds, and an inner loop is used to select the model
using k-fold cross-validation on the training fold. After model selection, the test
fold is then used to evaluate the model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""DATASET BREAST CANCER WISCONSIN"""

df_wdbc = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases'
    '/breast-cancer-wisconsin/wdbc.data',
    header=None)
"""Or"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_PopularLearning/BreastCancerWisconsin/wdbc.data"
df_wdbc = pd.read_csv(dataset_path, header=None)

"""PREPROCESSING DATA"""

"""LABEL ENCODING"""

from sklearn.preprocessing import LabelEncoder
X = df_wdbc.loc[:, 2:].values
y = df_wdbc.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

"""PARTITION TRAINING TEST"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

"""GRID SEARCH"""

"""PIPELINE FOR STANDARDIZATION AND LOGISTIC REGRESSION"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid,
                  scoring = 'accuracy',
                  cv = 2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} 'f'+/- {np.std(scores):.3f}')
"""
The returned average cross-validation accuracy gives us a good estimate of what to
expect if we tune the hyperparameters of a model and use it on unseen data.

For example, we can use the nested cross-validation approach to compare an SVM model
to a simple decision tree classifier; for simplicity, we will only tune its depth
parameter.
"""

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                scoring='accuracy',
                cv=2)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f} 'f'+/- {np.std(scores):.3f}')

"""
As we can see, the nested cross-validation performance of the SVM model is notably
better than the performance of the decision tree, and thus, we would expect that it
might be the better choice to classify new data that comes from the same population
as this particular dataset.
"""



gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
"""
We set the param_grid parameter of GridSearchCV to a list of dictionaries to
specify the parameters that we’d want to tune. For the linear SVM, we only
evaluated the inverse regularization parameter, C ; For the radial basis function
(RBF) kernel SVM, we tuned both the svc__C and svc__gamma parameters. Note that
the svc__gamma parameter is specific to kernel SVMs.

GridSearchCV uses k-fold cross-validation for comparing models trained with
different hyperparameter settings. Via the cv = 10 setting, it will carry out
10-fold cross-validation and compute the average accuracy (via scoring='accuracy')
across these 10-folds to assess the model performance. We set n_jobs = -1 so that
GridSearchCV can use all our processing cores to speed up the grid search by
fitting models to the different folds in parallel, but if your machine has problems
with this setting, you may change this setting to n_jobs = None for single processing.

After we used the training data to perform the grid search, we obtained the score
of the best-performing model via the best_score_ attribute and looked at its
parameters, which can be accessed via the best_params_ attribute.
"""

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')


"""
Since grid search is an exhaustive search, it is guaranteed to find the optimal
hyperparameter configuration if it is contained in the user-specified parameter
grid. However, specifying large hyperparameter grids makes grid search very
expensive in practice. An alternative approach for sampling different parameter
combinations is randomized search. In randomized search, we draw hyperparameter
configurations randomly from distributions (or discrete sets). In contrast to
grid search, randomized search does not do an exhaustive search over the
hyperparameter space. Still, it allows us to explore a wider range of hyperparameter
value settings in a more cost- and time-effective manner.

The main takeaway is that while grid search only explores discrete, user-specified
choices, it may miss good hyperparameter configurations if the search space is
too scarce.

We can use randomized search for tuning an SVM. Scikit-learn implements a
RandomizedSearchCV class, which is analogous to the GridSearchCV we used in the
previous subsection. The main difference is that we can specify distributions
as part of our parameter grid and specify the total number of hyperparameter
configurations to be evaluated.
"""

"""REGULARIZED SEARCH"""

import scipy.stats
param_range = scipy.stats.loguniform(0.0001, 1000.0)
"""
For instance, using a loguniform distribution instead of a regular uniform
distribution will ensure that in a sufficiently large number of trials, the same
number of samples will be drawn from the [0.0001, 0.001] range as, for example,
the [10.0, 100.0] range. To check its behavior, we can draw 10 random samples
from this distribution via the rvs(10) method, as shown here.
"""
np.random.seed(1)
param_range.rvs(10)

"""PIPELINE FOR STANDARDIZATION AND LOGISTIC REGRESSION"""

from sklearn.model_selection import RandomizedSearchCV
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
rs = RandomizedSearchCV(estimator=pipe_svc,
                        param_distributions=param_grid,
                        scoring='accuracy',
                        refit=True,
                        n_iter=20,
                        cv=10,
                        random_state=1,
                        n_jobs=-1)
rs = rs.fit(X_train, y_train)
print(rs.best_score_)
print(rs.best_params_)


"""HALVING RANDOM SEARCH"""
"""
Taking the idea of randomized search one step further, scikit-learn implements a
successive halving variant, HalvingRandomSearchCV, that makes finding suitable
hyperparameter configurations more efficient. Successive halving, given a large
set of candidate configurations, successively throws out unpromising hyperparameter
configurations until only one configuration remains. We can summarize the procedure
via the following steps :
1.  Draw a large set of candidate configurations via random sampling
2.  Train the models with limited resources, for example, a small subset of the
    training data (as opposed to using the entire training set)
3.  Discard the bottom 50 percent based on predictive performance
4.  Go back to step 2 with an increased amount of available resources

The steps are repeated until only one hyperparameter configuration remains. Note
that there is also a successive halving implementation for the grid search variant
called HalvingGridSearchCV, where all specified hyperparameter configurations are
used in step 1 instead of random samples.
"""
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
hs = HalvingRandomSearchCV(pipe_svc,
                           param_distributions=param_grid,
                           n_candidates='exhaust',
                           resource='n_samples',
                           factor=1.5,
                           random_state=1,
                           n_jobs=-1)

"""
The resource='n_samples' (default) setting specifies that we consider the training
set size as the resource we vary between the rounds. Via the factor parameter, we
can determine how many candidates are eliminated in each round. For example, setting
factor=2 eliminates half of the candidates, and setting factor=1.5 means that only
100%/1.5 ≈ 66% of the candidates make it into the next round. Instead of choosing
a fixed number of iterations as in RandomizedSearchCV, we set n_candidates='exhaust'
(default), which will sample the number of hyperparameter configurations such that
the maximum number of resources (here : training examples) are used in the last round.
"""

hs = hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)
clf = hs.best_estimator_
print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')


"""OTHERS HYPERPARAMETER OPTIMIZATION"""
"""
Another popular library for hyperparameter optimization is hyperopt, which implements
several different methods for hyperparameter optimization, including randomized search
and the Tree-structured Parzen Estimators (TPE) method. TPE is a Bayesian optimization
method based on a probabilistic model that is continuously updated based on past
hyperparameter evaluations and the associated performance scores instead of regarding
these evaluations as independent events.

While hyperopt provides a general-purpose interface for hyperparameter optimization,
there is also a scikit-learn-specific package called hyperopt-sklearn for additional
convenience.
"""
