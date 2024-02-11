import pandas as pd
import numpy as np
from io import StringIO


"""CSV DATA"""

csv_data = \
 '''A,B,C,D
 1.0,2.0,3.0,4.0
 5.0,6.0,,8.0
 10.0,11.0,12.0,'''
"""
If you are using Python 2.7, you need to convert the string to
unicode :
csv_data = unicode(csv_data)
"""

"""Loading CSV Data"""
df = pd.read_csv(StringIO(csv_data))

"""Show Data"""
df
df.values

"""Count missing data"""
df.isnull().sum()


"""DELETING DATA"""

"""
One of the easiest ways to deal with missing data is simply to
remove the corresponding features (columns) or training examples
(rows) from the dataset entirely ; Rows with missing values can
easily be dropped via the dropna method:
"""
df.dropna(axis=0)

"""
Similarly, we can drop columns that have at least one NaN in any
row by setting the axis argument to 1.
"""
df.dropna(axis=1)

"""
The dropna method supports several additional parameters that
can come in handy.
"""

"""Only drop rows where all columns are NaN."""
df.dropna(how='all')
"""
(Returns the whole array here since we don't have a row with
all values NaN)
"""

"""Drop rows that have fewer than 4 real values"""
df.dropna(thresh=4)

"""Only drop rows where NaN appear in specific columns (here: 'C')"""
df.dropna(subset=['C'])


"""IMPUTING MISSING VALUES"""
"""
We can use different interpolation techniques to estimate the
missing values from the other training examples in our dataset.
"""

"""
One of the most common interpolation techniques is mean imputation,
where we simply replace the missing value with the mean value of
the entire feature column. A convenient way to achieve this is by
using the SimpleImputer class from scikit-learn.
"""
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

"""
Here, we replaced each NaN value with the corresponding mean,
which is separately calculated for each feature column. Other
options for the strategy parameter are median or most_frequent,
where the latter replaces the missing values with the most frequent
values. This is useful for imputing categorical feature values, for
example, a feature column that stores an encoding of color names,
such as red, green, and blue.

Alternatively, an even more convenient way to impute missing values
is by using pandas’ fillna method and providing an imputation method
as an argument. For example, using pandas, we could achieve the same
mean imputation directly in the DataFrame object.
"""
df.fillna(df.mean())

"""
For additional imputation techniques, including the KNNImputer based
on a k-nearest neighbors approach to impute missing features by
nearest neighbors, we recommend the scikit-learn imputation
documentation.

The SimpleImputer class is part of the so-called transformer API
in scikit-learn, which is used for implementing Python classes
related to data transformation. The two essential methods of those
estimators are fit and transform. The fit method is used to learn
the parameters from the training data, and the transform method
uses those parameters to transform the data. Any data array that
is to be transformed needs to have the same number of features as
the data array that was used to fit the model.
"""
