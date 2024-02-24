"""
When we are talking about categorical data, we have to further
distinguish between ordinal and nominal features. Ordinal features
can be understood as categorical values that can be sorted or ordered.
In contrast, nominal features don’t imply any order.
"""

import pandas as pd
import numpy as np


"""DATA"""

"""Dataset created"""
df = pd.DataFrame([
           ['green', 'M', 10.1, 'class2'],
           ['red', 'L', 13.5, 'class1'],
           ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

df_map = df

"""Show Data"""
df

"""
The newly created DataFrame contains a nominal feature (color),
an ordinal feature (size), and a numerical feature (price) column.

To make sure that the learning algorithm interprets the ordinal
features correctly, we need to convert the categorical string values
into integers. Unfortunately, there is no convenient function that
can automatically derive the correct order of the labels of our
size feature, so we have to define the mapping manually.
"""


"""MAPPING THE ORDINAL FEATURE"""

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size_map']=df['size'].map(size_mapping)

"""
If we want to transform the integer values back to the original
string representation at a later stage, we can simply define a
reverse-mapping dictionary, inv_size_mapping, which can then be
used via the pandas map method on the transformed feature column
and is similar to the size_mapping dictionary that we used previously.
"""
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size_map'].map(inv_size_mapping)

"""
We can use the apply method of pandas’ DataFrame to write custom
lambda expressions in order to encode these variables using the
value-threshold approach.
"""
df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

"""Delete mapping in df"""
del df['x > M']
del df['x > L']
del df['size_map']

"""MAPPING THE NOMINAL FEATURE"""

"""
Many machine learning libraries require that class labels are
encoded as integer values. Although most estimators for classification
in scikit-learn convert class labels to integers internally, it is
considered good practice to provide class labels as integer arrays
to avoid technical glitches. To encode the class labels, we can use
an approach similar to the mapping of ordinal features discussed
previously. We need to remember that class labels are not ordinal,
and it doesn’t matter which integer number we assign to a particular
string label. Thus, we can simply enumerate the class labels,
starting at 0.
"""
class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
class_mapping
df['classlabel_map'] = df['classlabel'].map(class_mapping)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel_map'].map(inv_class_mapping)

"""
Alternatively, there is a convenient LabelEncoder class directly
implemented in scikit-learn to achieve this
"""
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
df['classlabel_label'] = class_le.fit_transform(df['classlabel'].values)
class_le.inverse_transform(df['classlabel_label_scikitlearn'])

"""
Since scikit-learn’s estimators for classification treat class
labels as categorical data that does not imply any order (nominal),
we used the convenient LabelEncoder to encode the string labels
into integers.
"""
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X[:, 1] = color_le.fit_transform(X[:, 1])
X

"""Delete mapping in df"""
del df['classlabel_map']
del df['classlabel_label']

"""MAPPING THE NOMINAL FEATURE WITH ONE-HOT ENCODING"""

"""
Although this assumption is incorrect, a classifier could still
produce useful results. However, those results would not be optimal.
A common workaround for this problem is to use a technique called
one-hot encoding. The idea behind this approach is to create a new
dummy feature for each unique value in the nominal feature column.
Here, we would convert the color feature into three new features :
blue, green, and red. Binary values can then be used to indicate
the particular color of an example ; For example, a blue example
can be encoded as blue=1, green=0, red=0. To perform this
transformation, we can use the OneHotEncoder that is implemented
in scikit-learn’s preprocessing module
"""
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

"""
Note that we applied the OneHotEncoder to only a single column,
(X[:, 0].reshape(-1, 1)), to avoid modifying the other two columns
in the array as well. If we want to selectively transform columns
in a multi-feature array, we can use the ColumnTransformer, which
accepts a list of (name, transformer, column(s)) tuples. We
specified that we want to modify only the first column and leave
the other two columns untouched via the 'passthrough' argument.
"""

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size_map']=df['size'].map(size_mapping)

from sklearn.compose import ColumnTransformer
X = df[['color', 'size_map', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
c_transf.fit_transform(X).astype(float)

"""
An even more convenient way to create those dummy features via
one-hot encoding is to use the get_dummies method implemented
in pandas. Applied to a DataFrame, the get_dummies method will
only convert string columns and leave all other columns unchanged.
"""
pd.get_dummies(df[['price', 'color', 'size']])

"""
When we are using one-hot encoding datasets, we have to keep in
mind that this introduces multicollinearity, which can be an issue
for certain methods . If features are highly correlated, matrices are
computationally difficult to invert, which can lead to numerically
unstable estimates. To reduce the correlation among variables, we
can simply remove one feature column from the one-hot encoded array.
Note that we do not lose any important information by removing a
feature column, though; for example, if we remove the column color_blue,
the feature information is still preserved since if we observe
color_green=0 and color_red=0, it implies that the observation must
be blue.
If we use the get_dummies function, we can drop the first column
by passing a True argument to the drop_first parameter.
"""
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)

"""
In order to drop a redundant column via the OneHotEncoder, we need
to set drop='first' and set categories='auto'.
"""
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
           ('onehot', color_ohe, [0]),
           ('nothing', 'passthrough', [1, 2])
])
c_transf.fit_transform(X).astype(float)
