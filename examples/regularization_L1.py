"""
Common solutions to reduce the generalization of overfitting are
as follows:
-   Collect more training data
-   Introduce a penalty for complexity via regularization
-   Choose a simpler model with fewer parameters
-   Reduce the dimensionality of the data

we will look at common ways to reduce overfitting by regularization
and dimensionality reduction via feature selection, which leads to
simpler models by requiring fewer parameters to be fitted to the data.

L1 regularization replaced the square of the weights with the sum
of the absolute values of the weights. In contrast to L2 regularization,
L1 regularization usually yields sparse feature vectors, and most
feature weights will be zero. Sparsity can be useful in practice if
we have a high-dimensional dataset with many features that are irrelevant,
especially in cases where we have more irrelevant dimensions than
training examples. In this sense, L1 regularization can be understood
as a technique for feature selection. L2 regularization adds a penalty
term to the loss function that effectively results in less extreme
weight values compared to a model trained with an unregularized loss
function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""DATASET WINE"""

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None)
"""Or"""
dataset_path = "/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/Table/Table_PopularLearning/Wine/wine.data"
df_wine = pd.read_csv(dataset_path, header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

"""Partition Training Test"""
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

"""FEATURE SCALING"""

"""Standardization with StandardScaler"""
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


"""FEATURE SELECTION"""


"""REGULARIZATION L1"""
"""
we can simply set the penalty parameter to 'l1' to obtain a sparse
solution. Note that we also need to select a different optimization
algorithm (for example, solver='liblinear'), since 'lbfgs' currently
does not support L1-regularized loss optimization. Note that C=1.0 is
the default. You can increase or decrease it to make the regularization
effect stronger or weaker, respectively.
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
"""
When we access the intercept terms via the lr.intercept_ attribute,
we can see that the array returns three values. Since we fit the
LogisticRegression object on a multiclass dataset via the
one-versus-rest (OvR) approach, the first intercept belongs to the
model that fits class 1 versus classes 2 and 3, the second value is
the intercept of the model that fits class 2 versus classes 1 and 3,
and the third value is the intercept of the model that fits class 3
versus classes 1 and 2.
ntercept_ corresponds to the bias unit
"""
lr.intercept_

"""
The weight array that we accessed via the lr.coef_ attribute contains
three rows of weight coefficients, one weight vector for each class.
Each row consists of 13 weights, where each weight is multiplied by
the respective feature in the 13-dimensional Wine dataset to calculate
the net input.
"""
lr.coef_

"""
In scikit-learn, intercept_ corresponds to the bias unit and coef_
corresponds to the values the different weights.
"""

"""
As a result of L1 regularization, which, as mentioned, serves as a
method for feature selection, we just trained a model that is robust
to the potentially irrelevant features in this dataset. Strictly
speaking, though, the weight vectors from the previous example are
not necessarily sparse because they contain more non-zero than zero
entries. However, we could enforce sparsity (more zero entries) by
further increasing the regularization strength—that is, choosing lower
values for the C parameter.

We vary the regularization strength and plot the regularization
path—the weight coefficients of the different features for different
regularization strengths.
"""

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c,
                            solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.title('The impact of the value of the regularization strength hyperparameter C', fontsize=12, fontweight='bold')
plt.ylabel('Weight coefficient')
plt.xlabel('C (Inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

"""
An alternative way to reduce the complexity of the model and avoid
overfitting is dimensionality reduction via feature selection,
which is especially useful for unregularized models. There are two
main categories of dimensionality reduction techniques : feature
selection and feature extraction. Via feature selection, we select
a subset of the original features, whereas in feature extraction,
we derive information from the feature set to construct a new
feature subspace.

Sequential feature selection algorithms are a family of greedy
search algorithms that are used to reduce an initial d-dimensional
feature space to a k-dimensional feature subspace where k<d. The
motivation behind feature selection algorithms is to automatically
select a subset of features that are most relevant to the problem,
to improve computational efficiency, or to reduce the generalization
error of the model by removing irrelevant features or noise, which
can be useful for algorithms that don’t support regularization.
"""

"""
CLASS SEQUENTIAL BACKWARD SELECTION
"""
"""
A classic sequential feature selection algorithm is sequential
backward selection (SBS), which aims to reduce the dimensionality
of the initial feature subspace with a minimum decay in the
performance of the classifier to improve upon computational
efficiency. In certain cases, SBS can even improve the predictive
power of the model if a model suffers from overfitting.
The idea behind the SBS algorithm is quite simple : SBS sequentially
removes features from the full feature subset until the new feature
subspace contains the desired number of features. To determine which
feature is to be removed at each stage, we need to define the
criterion function, J, that we want to minimize.

The criterion calculated by the criterion function can simply be
the difference in the performance of the classifier before and
after the removal of a particular feature. Then, the feature to be
removed at each stage can simply be defined as the feature that
maximizes this criterion ; Or in more simple terms, at each stage
we eliminate the feature that causes the least performance loss
after removal. Based on the preceding definition of SBS, we can
outline the algorithm in four simple steps :
1.  Initialize the algorithm with k = d, where d is the dimensionality of the full feature space, Xd.
2.  Determine the feature, x–, that maximizes the criterion: x– = argmax J(Xk – x).
3.  Remove the feature, x–, from the feature set: Xk–1 = Xk – x–; k = k – 1.
4.  Terminate if k equals the number of desired features; otherwise, go to step 2.
"""


from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:

    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

"""
In the preceding implementation, we defined the k_features parameter
to specify the desired number of features we want to return. By
default, we use accuracy_score from scikit-learn to evaluate the
performance of a model (an estimator for classification) on the
feature subsets.

Inside the while loop of the fit method, the feature subsets
created by the itertools.combination function are evaluated and
reduced until the feature subset has the desired dimensionality.
In each iteration, the accuracy score of the best subset is collected
in a list, self.scores_, based on the internally created test
dataset, X_test. We will use those scores later to evaluate the
results. The column indices of the final feature subset are assigned
to self.indices_, which we can use via the transform method to return
a new data array with the selected feature columns. Note that,
instead of calculating the criterion explicitly inside the fit
method, we simply removed the feature that is not contained in
the best performing feature subset.
"""

"""
EXAMPLE OF SEQUENTIAL BACKWARD SELECTION ON A KNN ALGORITHM
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
"""
Although our SBS implementation already splits the dataset into
a test and training dataset inside the fit function, we still
fed the training dataset, X_train_std, to the algorithm. The SBS
fit method will then create new training subsets for testing
(validation) and training, which is why this test set is also
called the validation dataset. This approach is necessary to
prevent our original test set from becoming part of the training
data.
"""

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.title('Impact of number of features on model accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
"""
The accuracy of the KNN classifier improved on the validation
dataset as we reduced the number of features, which is likely due
to a decrease in the curse of dimensionality. We can see the
classifier achieved 100 percent accuracy
for k = {3, 7, 8, 9, 10, 11, 12}.
"""

"""
the smallest feature subset (k=3), which yielded such a good
performance on the validation dataset.
WWe obtained the column indices of the three-feature subset from
the 11th position in the sbs.subsets_ attribute and returned the
corresponding feature names from the column index of the pandas
Wine DataFrame.
"""
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

"""
Let’s evaluate the performance of the KNN classifier on the
original test dataset.
"""
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))
"""
We used the complete feature set and obtained approximately 97
percent accuracy on the training dataset and approximately 96
percent accuracy on the test dataset, which indicates that our
model already generalizes well to new data.
"""

"""
Let’s use the selected three-feature subset and see how well
KNN performs.
"""
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))

"""
When using less than a quarter of the original features in the Wine
dataset, the prediction accuracy on the test dataset declined slightly.
This may indicate that those three features do not provide less
discriminatory information than the original dataset. However, we
also have to keep in mind that the Wine dataset is a small dataset
and is very susceptible to randomness—that is, the way we split the
dataset into training and test subsets, and how we split the training
dataset further into a training and validation subset.

While we did not increase the performance of the KNN model by
reducing the number of features, we shrank the size of the dataset,
which can be useful in real-world applications that may involve
expensive data collection steps. Also, by substantially reducing
the number of features, we obtain simpler models, which are easier
to interpret.

There are many more feature selection algorithms available via
scikit-learn. These include recursive backward elimination based
on feature weights, tree-based methods to select features by
importance, and univariate statistical tests.
"""

"""FEATURE ASSESSING WITH RANDOM FOREST"""

"""
Another useful approach for selecting relevant features from a
dataset is using a random forest, an ensemble technique. Using
a random forest, we can measure the feature importance as the
averaged impurity decrease computed from all decision trees in
the forest, without making any assumptions about whether our data
is linearly separable or not. Conveniently, the random forest
implementation in scikit-learn already collects the feature
importance values for us so that we can access them via the
feature_importances_ attribute after fitting a
RandomForestClassifier.

We train a forest of 500 trees on the Wine dataset and rank the
13 features by their respective importance measures. Remember that
we don’t need to use standardized or normalized features in
tree-based models.

We can create a plot that ranks the different features in the
Wine dataset by their relative importance ; Note that the feature
importance values are normalized so that they sum up to 1.0.
"""

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

"""
We can conclude that the proline and flavonoid levels, the color
intensity, the OD280/OD315 diffraction, and the alcohol concentration
of wine are the most discriminative features in the dataset based
on the average impurity decrease in the 500 decision trees.
Interestingly, two of the top-ranked features in the plot are also
in the three-feature subset selection from the SBS algorithm that we
implemented in the previous section (alcohol concentration and
OD280/OD315 of diluted wines).

However, as far as interpretability is concerned, the random forest
technique comes with an important gotcha that is worth mentioning.
If two or more features are highly correlated, one feature may be
ranked very highly while the information on the other feature(s)
may not be fully captured. On the other hand, we don’t need to be
concerned about this problem if we are merely interested in the
predictive performance of a model rather than the interpretation
of feature importance values.
"""

"""FEATURE ASSESSING WITH SCIKITLEARN SELECTFROMMODEL"""

"""
To conclude this section about feature importance values and random
forests, it is worth mentioning that scikit-learn also implements a
SelectFromModel object that selects features based on a user-specified
threshold after model fitting, which is useful if we want to use
the RandomForestClassifier as a feature selector and intermediate
step in a scikit-learn Pipeline object, which allows us to connect
different preprocessing steps with an estimator. For example, we
could set the threshold to 0.1 to reduce the dataset to the five
most important features using the following code.
"""

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold',
      'criterion:', X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
