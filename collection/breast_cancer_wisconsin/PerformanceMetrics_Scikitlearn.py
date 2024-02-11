"""
We can evaluate different machine learning models using prediction accuracy, which is a
useful metric with which to quantify the performance of a model in general. However, there
are several other performance metrics that can be used to measure a model’s relevance, such
as precision, recall, the F1 score, and Matthews correlation coefficient (MCC).
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
from sklearn.svm import SVC
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

"""CONFUSION MATRIX"""
"""
Before we get into the details of different scoring metrics, let’s take a look at a confusion
matrix, a matrix that lays out the performance of a learning algorithm.
A confusion matrix is simply a square matrix that reports the counts of the true positive (TP),
true negative (TN), false positive (FP), and false negative (FN) predictions of a classifier.

Although these metrics can be easily computed manually by comparing the actual and predicted
class labels, scikit-learn provides a convenient confusion_matrix function.
"""

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
"""
The array that was returned after executing the code provides us with information about the
different types of error the classifier made on the test dataset. We can map this
information onto the confusion matrix illustration using Matplotlib’s matshow function.
"""
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

"""PERFORMANCE METRICS"""

"""ERROR AND PREDICTION"""
"""
Both the prediction error and accuracy provide general information about how many
examples are misclassified. The error can be understood as the sum of all false
predictions divided by the number of total predictions, and the accuracy is
calculated as the sum of correct predictions divided by the total number of predictions.
Accuracy = 1 - Error.
"""
"""TRUE POSITIVE RATE AND FALSE POSITIVE RATE"""
"""
The true positive rate (TPR) and false positive rate (FPR) are performance metrics that
are especially useful for imbalanced class problems.
True Positive Rate = True Positive / (False Negative + True Positive (= Positive)).
False Positive Rate = False Positive / (False Positive + True Negative (= Negative)).
"""
"""PRECISION AND RECALL"""
"""
Recall quantifies how many of the relevant records the positives are captured as such
the true positives. Precision quantifies how many of the records predicted as relevant
the sum of true and false positives are actually relevant true positives.
Recall = True Positive Rate.
Precision = True Positive / (True Positive + False Positive).
"""
"""F1 SCORE"""
"""
To balance the up- and downsides of optimizing Precision and Recall, the harmonic mean of
Precision and Recall is used, the so-called F1 score.
F1 = 2 x Precision x Recall / (Precision + Recall).
"""
"""MCC"""
"""
A measure that summarizes a confusion matrix is the MCC, which is especially popular
in biological research contexts.
In contrast to Precision, Rrcall, and the F1 score, the MCC ranges between –1 and 1,
and it takes all elements of a confusion matrix into account. For instance, the F1
score does not involve the TN. While the MCC values are harder to interpret than the
F1 score, it is regarded as a superior metric.
MCC = (TP x TN - FP x FN) / SQRT((TP + FP) x (TP + FN) x (TN + FP) x (TN + FN)).
"""

"""PERFORMANCE METRICS SCIKITLEARN"""
"""
Those scoring metrics are all implemented in scikit-learn and can be imported from
the sklearn.metrics module.
"""
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')
rec_val = recall_score(y_true=y_test, y_pred=y_pred)
print(f'Recall: {rec_val:.3f}')
f1_val = f1_score(y_true=y_test, y_pred=y_pred)
print(f'F1: {f1_val:.3f}')
mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f'MCC: {mcc_val:.3f}')

"""
Furthermore, we can use a different scoring metric than accuracy in the GridSearchCV
via the scoring parameter.

Remember that the positive class in scikit-learn is the class that is labeled as
class 1. If we want to specify a different positive label, we can construct our own
scorer via the make_scorer function, which we can then directly provide as an argument
to the scoring parameter in GridSearchCV.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

"""RECEIVER OPERATING CHARACTERISTIC (ROC)"""
"""
Receiver operating characteristic (ROC) graphs are useful tools to select models for
classification based on their performance with respect to the FPR and TPR, which are
computed by shifting the decision threshold of the classifier. The diagonal of a ROC
graph can be interpreted as random guessing, and classification models that fall below
the diagonal are considered as worse than random guessing. A perfect classifier would
fall into the top-left corner of the graph with a TPR of 1 and an FPR of 0. Based on the
ROC curve, we can then compute the so-called ROC area under the curve (ROC AUC) to
characterize the performance of a classification model.

Similar to ROC curves, we can compute precision-recall curves for different probability
thresholds of a classifier. A function for plotting those precision-recall curves is also
implemented in scikit-learn.

we will plot a ROC curve of a classifier that only uses two features from the Breast
Cancer Wisconsin dataset to predict whether a tumor is benign or malignant. Although we
are going to use the same logistic regression pipeline that we defined previously, we are
only using two features this time. This is to make the classification task more
challenging for the classifier, by withholding useful information contained in the other
features, so that the resulting ROC curve becomes visually more interesting. For similar
reasons, we are also reducing the number of folds in the StratifiedKFold validator to
three.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from numpy import interp
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1,
                       solver='lbfgs', C=100.0))
X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(
        X_train2[train],
        y_train[train]
    ).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label=f'ROC fold {i+1} (area = {roc_auc:.2f})')
plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing (area=0.5)')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance (area=1.0)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.show()
"""
We used the already familiar StratifiedKFold class from scikit-learn and calculated the
ROC performance of the LogisticRegression classifier in our pipe_lr pipeline using the
roc_curve function from the sklearn.metrics module separately for each iteration.
Furthermore, we interpolated the average ROC curve from the three folds via the interp
function that we imported from NumPy and calculated the area under the curve via the
auc function. The resulting ROC curve indicates that there is a certain degree of
variance between the different folds, and the average ROC AUC falls between a perfect
score (1.0) and random guessing (0.5).

Note that if we are just interested in the ROC AUC score, we could also directly import
the roc_auc_score function from the sklearn.metrics submodule, which can be used similarly
to the other scoring functions (for example, precision_score) that were introduced in the
previous sections.
"""

"""MULTICLASS CLASSIFICATION"""
"""
scikit-learn also implements macro and micro averaging methods to extend those scoring
metrics to multiclass problems via one-vs.-all (OvA) classification. The micro-average
is calculated from the individual TPs, TNs, FPs, and FNs of the system. For example,
the micro-average of the precision score in a k-class system can be calculated as follows :
Precision_Micro = (TP_1 + ... + TP_k) / (TP_1 + ... + TP_k + FP_1 + ... + FP_k).
The macro-average is simply calculated as the average scores of the different systems.
Precision_Macro = (Precision_1 + ... + Precision_k) / k

Micro-averaging is useful if we want to weight each instance or prediction equally, whereas
macro-averaging weights all classes equally to evaluate the overall performance of a
classifier with regard to the most frequent class labels.

If we are using binary performance metrics to evaluate multiclass classification models
in scikit-learn, a normalized or weighted variant of the macro-average is used by default.
The weighted macro-average is calculated by weighting the score of each class label by
the number of true instances when calculating the average. The weighted macro-average is
useful if we are dealing with class imbalances, that is, different numbers of instances
for each label.

While the weighted macro-average is the default for multiclass problems in scikit-learn,
we can specify the averaging method via the average parameter inside the different scoring
functions that we import from the sklearn.metrics module, for example, the precision_score
or make_scorer functions.
"""
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')
