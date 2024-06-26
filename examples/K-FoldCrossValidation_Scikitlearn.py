"""
A classic and popular approach for estimating the generalization
performance of machine learning models is the holdout method. Using
the holdout method, we split our initial dataset into separate
training and test datasets—the former is used for model training,
and the latter is used to estimate its generalization performance.
However, in typical machine learning applications, we are also
interested in tuning and comparing different parameter settings to
further improve the performance for making predictions on unseen
data. This process is called model selection, with the name
referring to a given classification problem for which we want to
select the optimal values of tuning parameters (also called
hyperparameters). However, if we reuse the same test dataset over
and over again during model selection, it will become part of our
training data and thus the model will be more likely to overfit.
Despite this issue, many people still use the test dataset for
model selection, which is not a good machine learning practice.

A better way of using the holdout method for model selection is to
separate the data into three parts: a training dataset, a validation
dataset, and a test dataset. The training dataset is used to fit
the different models, and the performance on the validation dataset
is then used for model selection. The advantage of having a test
dataset that the model hasn’t seen before during the training and
model selection steps is that we can obtain a less biased estimate
of its ability to generalize to new data. Once we are satisfied with
the tuning of hyperparameter values, we estimate the model’s
generalization performance on the test dataset.

A disadvantage of the holdout method is that the performance estimate
may be very sensitive to how we partition the training dataset into
the training and validation subsets ; The estimate will vary for
different examples of the data. In the next subsection, we will
take a look at a more robust technique for performance estimation,
k-fold cross-validation, where we repeat the holdout method k times
on k subsets of the training data.
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

"""PIPELINE FOR STANDARDIZATION, DIMENSIONAL REDUCTION AND LOGISTIC REGRESSION"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression())
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

"""K-FOLD CROSS-VALIDATION"""

"""
In k-fold cross-validation, we randomly split the training dataset
into k folds without replacement. Here, k – 1 folds, the so-called
training folds, are used for the model training, and one fold, the
so-called test fold, is used for performance evaluation. This
procedure is repeated k times so that we obtain k models and
performance estimates.

We then calculate the average performance of the models based on
the different, independent test folds to obtain a performance
estimate that is less sensitive to the sub-partitioning of the
training data compared to the holdout method. Typically, we use
k-fold cross-validation for model tuning, that is, finding the
optimal hyperparameter values that yield a satisfying generalization
performance, which is estimated from evaluating the model
performance on the test folds.

Once we have found satisfactory hyperparameter values, we can
retrain the model on the complete training dataset and obtain a
final performance estimate using the independent test dataset. The
rationale behind fitting a model to the whole training dataset
after k-fold cross-validation is that first, we are typically
interested in a single, final model (versus k individual models),
and second, providing more training examples to a learning algorithm
usually results in a more accurate and robust model.

Since k-fold cross-validation is a resampling technique without
replacement, the advantage of this approach is that in each iteration,
each example will be used exactly once, and the training and test
folds are disjoint. Furthermore, all test folds are disjoint ; That
is, there is no overlap between the test folds. For example, the
training dataset is divided into 10 folds, and during the 10
iterations, 9 folds are used for training, and 1 fold will be used
as the test dataset for model evaluation.
"""

"""
In summary, k-fold cross-validation makes better use of the dataset
than the holdout method with a validation set, since in k-fold
cross-validation all data points are being used for evaluation. A
good standard value for k in k-fold cross-validation is 10, as
empirical evidence shows.

However, if we are working with relatively small training sets, it
can be useful to increase the number of folds. If we increase the
value of k, more training data will be used in each iteration,
which results in a lower pessimistic bias toward estimating the
generalization performance by averaging the individual model
estimates. However, large values of k will also increase the
runtime of the cross-validation algorithm and yield estimates
with higher variance, since the training folds will be more similar
to each other. On the other hand, if we are working with large
datasets, we can choose a smaller value for k, for example,
k = 5, and still obtain an accurate estimate of the average
performance of the model while reducing the computational cost
of refitting and evaluating the model on the different folds.

A special case of k-fold cross-validation is the leave-one-out
cross-validation (LOOCV) method. In LOOCV, we set the number of
folds equal to the number of training examples (k = n) so that
only one training example is used for testing during each
iteration, which is a recommended approach for working with very
small datasets.

A slight improvement over the standard k-fold cross-validation
approach is stratified k-fold cross-validation, which can yield
better bias and variance estimates, especially in cases of unequal
class proportions. In stratified cross-validation, the class label
proportions are preserved in each fold to ensure that each fold
is representative of the class proportions in the training dataset.
"""

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'Fold: {k+1:02d}, '
          f'Class distr.: {np.bincount(y_train[train])}, '
          f'Acc.: {score:.3f}')
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')
"""
First, we initialized the StratifiedKFold iterator from the
sklearn.model_selection module with the y_train class labels in
the training dataset, and we specified the number of folds via
the n_splits parameter. When we used the kfold iterator to loop
through the k folds, we used the returned indices in train to fit
the logistic regression pipeline that we set up at the beginning
of this chapter. Using the pipe_lr pipeline, we ensured that the
examples were scaled properly (for instance, standardized) in
each iteration. We then used the test indices to calculate the
accuracy score of the model, which we collected in the scores
list to calculate the average accuracy and the standard deviation
of the estimate.

Although the previous code example was useful to illustrate how
k-fold cross-validation works, scikit-learn also implements a
k-fold cross-validation scorer, which allows us to evaluate our
model using stratified k-fold cross-validation less verbosely.
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print(f'CV accuracy scores: {scores}')
print(f'CV accuracy: {np.mean(scores):.3f} '
      f'+/- {np.std(scores):.3f}')
"""
An extremely useful feature of the cross_val_score approach is
that we can distribute the evaluation of the different folds
across multiple central processing units (CPUs) on our machine.
If we set the n_jobs parameter to 1, only one CPU will be used
to evaluate the performances, just like in our StratifiedKFold
example previously. However, by setting n_jobs=2, we could
distribute the 10 rounds of cross-validation to two CPUs (if
available on our machine), and by setting n_jobs=-1, we can use
all available CPUs on our machine to do the computation in parallel.
"""
