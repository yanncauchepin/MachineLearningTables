"""
Class imbalance is a quite common problem when working with real-world data—examples from
one class or multiple classes are over-represented in a dataset. We can think of several
domains where this may occur, such as spam filtering, fraud detection, or screening for
diseases.

Imagine that the Breast Cancer Wisconsin dataset that we’ve been working with in this
chapter consisted of 90 percent healthy patients. In this case, we could achieve 90
percent accuracy on the test dataset by just predicting the majority class (benign tumor)
for all examples, without the help of a supervised machine learning algorithm. Thus,
training a model on such a dataset that achieves approximately 90 percent test accuracy
would mean our model hasn’t learned anything useful from the features provided in this
dataset.
"""

"""CLASS IMBALANCE"""

import numpy as np
import pandas as pd

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

"""
We create an imbalanced dataset from our dataset, which originally consisted of
357 benign tumors (class 0) and 212 malignant tumors (class 1).
we took all 357 benign tumor examples and stacked them with the first 40 malignant
examples to create a stark class imbalance. If we were to compute the accuracy of
a model that always predicts the majority class (benign, class 0), we would achieve
a prediction accuracy of approximately 90 percent.
"""

X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

"""
Thus, when we fit classifiers on such datasets, it would make sense to focus on other
metrics than accuracy when comparing different models, such as precision, recall, the
ROC curve—whatever we care most about in our application. For instance, our priority
might be to identify the majority of patients with malignant cancer to recommend an
additional screening, so recall should be our metric of choice. In spam filtering,
where we don’t want to label emails as spam if the system is not very certain, precision
might be a more appropriate metric.

Aside from evaluating machine learning models, class imbalance influences a learning
algorithm during model fitting itself. Since machine learning algorithms typically
optimize a reward or loss function that is computed as a sum over the training examples
that it sees during fitting, the decision rule is likely going to be biased toward the
majority class.

In other words, the algorithm implicitly learns a model that optimizes the predictions
based on the most abundant class in the dataset to minimize the loss or maximize the
reward during training.

One way to deal with imbalanced class proportions during model fitting is to assign
a larger penalty to wrong predictions on the minority class. Via scikit-learn, adjusting
such a penalty is as convenient as setting the class_weight parameter to
class_weight='balanced', which is implemented for most classifiers.

Other popular strategies for dealing with class imbalance include upsampling the minority
class, downsampling the majority class, and the generation of synthetic training examples.
Unfortunately, there’s no universally best solution or technique that works best across
different problem domains. Thus, in practice, it is recommended to try out different
strategies on a given problem, evaluate the results, and choose the technique that seems
most appropriate.

The scikit-learn library implements a simple resample function that can help with the
upsampling of the minority class by drawing new samples from the dataset with replacement.
The following code will take the minority class from our imbalanced Breast Cancer
Wisconsin dataset (here, class 1) and repeatedly draw new samples from it until it
contains the same number of examples as class label 0.
"""

from sklearn.utils import resample
print('Number of class 1 examples before:',
      X_imb[y_imb == 1].shape[0])
X_upsampled, y_upsampled = resample(
        X_imb[y_imb == 1],
        y_imb[y_imb == 1],
        replace=True,
        n_samples=X_imb[y_imb == 0].shape[0],
        random_state=123)
print('Number of class 1 examples after:',
      X_upsampled.shape[0])

"""
After resampling, we can then stack the original class 0 samples with the upsampled
class 1 subset to obtain a balanced dataset.
"""

X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

"""
Consequently, a majority vote prediction rule would only achieve 50 percent accuracy.
"""

y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
