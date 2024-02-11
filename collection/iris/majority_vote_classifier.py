import numpy as np
import matplotlib.pyplot as plt

"""CLASS MAJORITYVOTECLASSIFIER"""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel'"
                             f"; got (vote={self.vote})")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and'
                             f' weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,
                               self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else: # 'classlabel' vote

            # Collect results from clf.predict calls
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T

            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self.weights)
                ),
                axis=1, arr=predictions
            )
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0,
                               weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(
                        deep=True).items():
                    out[f'{name}__{key}'] = value
            return out

"""
We used the BaseEstimator and ClassifierMixin parent classes to get some base
functionality for free, including the get_params and set_params methods to set
and return the classifier’s parameters, as well as the score method to calculate
the prediction accuracy.

Next, we will add the predict method to predict the class label via a majority
vote based on the class labels if we initialize a new MajorityVoteClassifier
object with vote='classlabel'. Alternatively, we will be able to initialize the
ensemble classifier with vote='probability' to predict the class label based on
the class membership probabilities. Furthermore, we will also add a predict_proba
method to return the averaged probabilities, which is useful when computing the
receiver operating characteristic area under the curve (ROC AUC).

Also, note that we defined our own modified version of the get_params method to
use the _name_estimators function to access the parameters of individual classifiers
in the ensemble; this may look a little bit complicated at first, but it will
make perfect sense when we use grid search for hyperparameter tuning in later
sections.
"""

"""CLASS MAJORITYVOTECLASSIFIER SCIKITLEARN"""
"""
Although the MajorityVoteClassifier implementation is very useful for demonstration
purposes, we implemented a more sophisticated version of this majority vote
classifier in scikit-learn based on the implementation in the first edition of
this book. The ensemble classifier is available as sklearn.ensemble.VotingClassifier.
"""

"""DATASET IRIS SCIKITLEARN"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

"""Partition Training Test"""
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

"""EXAMPLE OF APPLICATION MAJORITYVOTECLASSIFIER"""

"""
Using the training dataset, we now will train three different classifiers.
We will then evaluate the model performance of each classifier via 10-fold
cross-validation on the training dataset before we combine them into an ensemble
classifier.
"""

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')

"""
Why we trained the logistic regression and k-nearest neighbors classifier as part
of a pipeline ? The reason behind it is that, as discussed in Chapter 3, both the
logistic regression and k-nearest neighbors algorithms (using the Euclidean
distance metric) are not scale-invariant, in contrast to decision trees. Although
the Iris features are all measured on the same scale (cm), it is a good habit
to work with standardized features.
"""

mv_clf = MajorityVoteClassifier(
    classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')

"""Evaluation"""
"""
In this section, we are going to compute the ROC curves from the test dataset to
check that MajorityVoteClassifier generalizes well with unseen data. We must
remember that the test dataset is not to be used for model selection ; its purpose
is merely to report an unbiased estimate of the generalization performance of a
classifier system.
"""
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
    in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,
                     y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label=f'{label} (auc = {roc_auc:.2f})')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()
"""
The ensemble classifier also performs well on the test dataset. However, you can
see that the logistic regression classifier performs similarly well on the same
dataset, which is probably due to the high variance (in this case, the sensitivity
of how we split the dataset) given the small size of the dataset.

Since we only selected two features for the classification examples, it would be
interesting to see what the decision region of the ensemble classifier actually
looks like.
Although it is not necessary to standardize the training features prior to model
fitting, because our logistic regression and k-nearest neighbors pipelines will
automatically take care of it, we will standardize the training dataset so that
the decision regions of the decision tree will be on the same scale for visual
purposes.
"""

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                  X_train_std[y_train==0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                  X_train_std[y_train==1, 1],
                                  c='green',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -5.,
         s='Sepal width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5,
         s='Petal length [standardized]',
         ha='center', va='center',
         fontsize=12, rotation=90)
plt.show()
"""
Interestingly, but also as expected, the decision regions of the ensemble classifier
seem to be a hybrid of the decision regions from the individual classifiers.
At first glance, the majority vote decision boundary looks a lot like the decision
of the decision tree stump, which is orthogonal to the y axis for sepal width ≥ 1.
"""

"""
Before we tune the individual classifier’s parameters for ensemble classification,
let’s call the get_params method to get a basic idea of how we can access the
individual parameters inside a GridSearchCV object.
"""
mv_clf.get_params()

"""
Let’s now tune the inverse regularization parameter, C, of the logistic regression
classifier and the decision tree depth via a grid search for demonstration purposes.
"""
from sklearn.model_selection import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)
"""
After the grid search has completed, we can print the different hyperparameter
value combinations and the average ROC AUC scores computed via 10-fold cross-validation
as follows.
"""
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean_score = grid.cv_results_['mean_test_score'][r]
    std_dev = grid.cv_results_['std_test_score'][r]
    params = grid.cv_results_['params'][r]
    print(f'{mean_score:.3f} +/- {std_dev:.2f} {params}')
print(f'Best parameters: {grid.best_params_}')
print(f'ROC AUC : {grid.best_score_:.2f}')
"""
As you can see, we get the best cross-validation results when we choose a lower
regularization strength (C=0.001), whereas the tree depth does not seem to affect
the performance at all, suggesting that a decision stump is sufficient to separate
the data. To remind ourselves that it is a bad practice to use the test dataset
more than once for model evaluation, we are not going to estimate the generalization
performance of the tuned hyperparameters in this section. We will move on swiftly
to an alternative approach for ensemble learning: bagging.
"""

"""
The majority vote approach we implemented in this section is not to be confused
with stacking. The stacking algorithm can be understood as a two-level ensemble,
where the first level consists of individual classifiers that feed their predictions
to the second level, where another classifier (typically logistic regression) is
fit to the level-one classifier predictions to make the final predictions.
"""
