import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""DATASET BREAST CANCER WISCONSIN"""
"""
We are working with the Breast Cancer Wisconsin dataset, which
contains 569 examples of malignant and benign tumor cells. The
first two columns in the dataset store the unique ID numbers of the
examples and the corresponding diagnoses (M = malignant, B = benign),
respectively. Columns 3-32 contain 30 real-valued features that have
been computed from digitized images of the cell nuclei, which can be
used to build a model to predict whether a tumor is benign or
malignant.
"""

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
After encoding the class labels (diagnosis) in an array, y, the
malignant tumors are now represented as class 1, and the benign
tumors are represented as class 0, respectively. We can
double-check this mapping by calling the transform method of the
fitted LabelEncoder on two dummy class labels.
"""
le.transform(['M', 'B'])

"""PARTITION TRAINING TEST"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

"""PIPELINE FOR STANDARDIZATION, DIMENSIONAL REDUCTION AND LOGISTIC REGRESSION"""
"""
Since the features in the Breast Cancer Wisconsin dataset are
measured on various different scales, we will standardize the columns
in the Breast Cancer Wisconsin dataset before we feed them to a linear
classifier, such as logistic regression. Furthermore, let’s assume
that we want to compress our data from the initial 30 dimensions
into a lower two-dimensional subspace via principal component
analysis (PCA), a feature extraction technique for dimensionality
reduction.
Instead of going through the model fitting and data transformation
steps for the training and test datasets separately, we can chain
the StandardScaler, PCA, and LogisticRegression objects in a pipeline.
"""

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

"""
The make_pipeline function takes an arbitrary number of scikit-learn
transformers (objects that support the fit and transform methods as
input), followed by a scikit-learn estimator that implements the
fit and predict methods. In our preceding code example, we provided
two scikit-learn transformers, StandardScaler and PCA, and a
LogisticRegression estimator as inputs to the make_pipeline function,
which constructs a scikit-learn Pipeline object from these objects.
We can think of a scikit-learn Pipeline as a meta-estimator or
wrapper around those individual transformers and estimators. If we
call the fit method of Pipeline, the data will be passed down a
series of transformers via fit and transform calls on these
intermediate steps until it arrives at the estimator object (the
final element in a pipeline). The estimator will then be fitted to
the transformed training data.
When we executed the fit method on the pipe_lr pipeline in the
preceding code example, StandardScaler first performed fit and
transform calls on the training data. Second, the transformed
training data was passed on to the next object in the pipeline,
PCA. Similar to the previous step, PCA also executed fit and
transform on the scaled input data and passed it to the final
element of the pipeline, the estimator.
Finally, the LogisticRegression estimator was fit to the training
data after it underwent transformations via StandardScaler and PCA.
Again, we should note that there is no limit to the number of
intermediate steps in a pipeline; however, if we want to use the
pipeline for prediction tasks, the last pipeline element has to be
an estimator.
Similar to calling fit on a pipeline, pipelines also implement a
predict method if the last step in the pipeline is an estimator.
If we feed a dataset to the predict call of a Pipeline object
instance, the data will pass through the intermediate steps via
transform calls. In the final step, the estimator object will then
return a prediction on the transformed data.
"""
