import pandas as pd
import numpy as np

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

print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()


"""PARTITION TRAINING TEST"""
"""
A convenient way to randomly partition this dataset into separate
test and training datasets is to use the train_test_split function
from scikit-learn’s model_selection submodule.
"""
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
"""
We assigned the NumPy array representation of the feature columns 1-13
to the variable X and we assigned the class labels from the first
column to the variable y. Then, we used the train_test_split function
to randomly split X and y into separate training and test datasets.

By setting test_size=0.3, we assigned 30 percent of the wine examples
to X_test and y_test, and the remaining 70 percent of the examples
were assigned to X_train and y_train, respectively. Providing the
class label array y as an argument to stratify ensures that both
training and test datasets have the same class proportions as the
original dataset.
"""

"""
If we are dividing a dataset into training and test datasets, we
have to keep in mind that we are withholding valuable information
that the learning algorithm could benefit from. Thus, we don’t want
to allocate too much information to the test set. However, the
smaller the test set, the more inaccurate the estimation of the
generalization error. Dividing a dataset into training and test
datasets is all about balancing this tradeoff. In practice, the
most commonly used splits are 60:40, 70:30, or 80:20, depending
on the size of the initial dataset. However, for large datasets,
90:10 or 99:1 splits are also common and appropriate. For example,
if the dataset contains more than 100,000 training examples, it
might be fine to withhold only 10,000 examples for testing in order
to get a good estimate of the generalization performance.

Moreover, instead of discarding the allocated test data after model
training and evaluation, it is a common practice to retrain a
classifier on the entire dataset, as it can improve the predictive
performance of the model. While this approach is generally
recommended, it could lead to worse generalization performance if
the dataset is small and the test dataset contains outliers, for
example. Also, after refitting the model on the whole dataset, we
don’t have any independent data left to evaluate its performance.
"""
