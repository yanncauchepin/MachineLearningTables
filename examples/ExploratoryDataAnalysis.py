"""
Exploratory data analysis (EDA) is an important and recommended first step prior
to the training of a machine learning model.
We create a scatterplot matrix that allows us to visualize the pair-wise correlations
between the different features in this dataset in one place. To plot the scatterplot
matrix, we will use the scatterplotmatrix function from the mlxtend library, which
is a Python library that contains various convenience functions for machine learning
and data science applications in Python.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""EXPLORATORY DATA ANALYSIS"""

"""DATASET AMES HOUSING"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)

"""DATA DISTRIBUTION"""
"""
We visualize the data distributions of the Ames Housing dataset variables in the
form of histograms and scatterplots.
"""
from mlxtend.plotting import scatterplotmatrix
scatterplotmatrix(df.values, figsize=(12, 10),
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()
"""
Using this scatterplot matrix, we can now quickly see how the data is distributed
and whether it contains outliers.
"""

"""CORRELATION MATRIX"""
"""
we create a correlation matrix to quantify and summarize linear relationships
between variables. A correlation matrix is closely related to the covariance matrix.
We can interpret the correlation matrix as being a rescaled version of the covariance
matrix. In fact, the correlation matrix is identical to a covariance matrix computed
from standardized features.
The correlation matrix is a square matrix that contains the Pearson product-moment
correlation coefficient (often abbreviated as Pearson’s r), which measures the linear
dependence between pairs of features. The correlation coefficients are in the
range –1 to 1. Two features have a perfect positive correlation if r = 1, no
correlation if r = 0, and a perfect negative correlation if r = –1. As mentioned
previously, Pearson’s correlation coefficient can simply be calculated as the
covariance between two features, x and y (numerator), divided by the product of
their standard deviations (denominator) :
r = sigma_xy / (sigma_x x sigma_y)
"""
"""
We use NumPy’s corrcoef function on the five feature columns that we previously
visualized in the scatterplot matrix, and we will use mlxtend’s heatmap function
to plot the correlation matrix array as a heat map.
"""

from mlxtend.plotting import heatmap
cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()
