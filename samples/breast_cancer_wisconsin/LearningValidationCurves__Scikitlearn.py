"""
If a model is too complex for a given training dataset—for example,
think of a very deep decision tree—the model tends to overfit the
training data and does not generalize well to unseen data. Often,
it can help to collect more training examples to reduce the degree
of overfitting.

However, in practice, it can often be very expensive or simply not
feasible to collect more data. By plotting the model training and
validation accuracies as functions of the training dataset size,
we can easily detect whether the model suffers from high variance
or high bias, and whether the collection of more data could help
to address this problem.

A model with a high bias has both low training and cross-validation
accuracy, which indicates that it underfits the training data.
Common ways to address this issue are to increase the number of
model parameters, for example, by collecting or constructing
additional features, or by decreasing the degree of regularization,
for example, in support vector machine (SVM) or logistic regression
classifiers.

A model that suffers from high variance, which is indicated by the
large gap between the training and cross-validation accuracy. To
address this problem of overfitting, we can collect more training
data, reduce the complexity of the model, or increase the
regularization parameter, for example.
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

"""PIPELINE FOR STANDARDIZATION AND LOGISTIC REGRESSION"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2',
                                           max_iter=10000))
train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.title('Learning curve showing training and validation dataset accuracy by the number of training examples', fontsize=12, fontweight='bold')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()

"""
Note that we passed max_iter=10000 as an additional argument when
instantiating the LogisticRegression object (which uses 1,000
iterations as a default) to avoid convergence issues for the
smaller dataset sizes or extreme regularization parameter values
(covered in the next section).

Via the train_sizes parameter in the learning_curve function, we
can control the absolute or relative number of training examples
that are used to generate the learning curves. Here, we set
train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly spaced,
relative intervals for the training dataset sizes. By default,
the learning_curve function uses stratified k-fold cross-validation
to calculate the cross-validation accuracy of a classifier, and we
set k = 10 via the cv parameter for 10-fold stratified cross-validation.

Then, we simply calculated the average accuracies from the returned
cross-validated training and test scores for the different sizes
of the training dataset. Furthermore, we added the standard
deviation of the average accuracy to the plot using the
fill_between function to indicate the variance of the estimate.
"""

"""
Validation curves are a useful tool for improving the performance
of a model by addressing issues such as overfitting or underfitting.
Validation curves are related to learning curves, but instead of
plotting the training and test accuracies as functions of the sample
size, we vary the values of the model parameters, for example, the
inverse regularization parameter, C, in logistic regression.
"""

from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                            estimator=pipe_lr,
                            X=X_train,
                            y=y_train,
                            param_name='logisticregression__C',
                            param_range=param_range,
                            cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.title('A validation curve plot for the SVM hyperparameter C', fontsize=12, fontweight='bold')
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()

"""
Similar to the learning_curve function, the validation_curve
function uses stratified k-fold cross-validation by default to
estimate the performance of the classifier. Inside the
validation_curve function, we specified the parameter that we
wanted to evaluate. In this case, it is C, the inverse
regularization parameter of the LogisticRegression classifier,
which we wrote as 'logisticregression__C' to access the
LogisticRegression object inside the scikit-learn pipeline for a
specified value range that we set via the param_range parameter.
Similar to the learning curve example in the previous section,
we plotted the average training and cross-validation accuracies
and the corresponding standard deviations.

Although the differences in the accuracy for varying values of C
are subtle, we can see that the model slightly underfits the data
when we increase the regularization strength (small values of C).
However, for large values of C, it means lowering the strength of
regularization, so the model tends to slightly overfit the data.
In this case, the sweet spot appears to be between 0.1 and 1.0
of the C value.
"""
