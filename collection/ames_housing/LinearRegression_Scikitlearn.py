import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""UNIVARIATE LINEAR REGRESSION"""
"""
The goal of simple (univariate) linear regression is to model the relationship
between a single feature (explanatory variable, x) and a continuous-valued target
(response variable, y). The equation of a linear model with one explanatory variable
is defined as follows :
y = w x x + b
Here, the parameter (bias unit), b, represents the y axis intercept and w1 is the
weight coefficient of the explanatory variable. Our goal is to learn the weights
of the linear equation to describe the relationship between the explanatory variable
and the target variable, which can then be used to predict the responses of new
explanatory variables that were not part of the training dataset.
Based on the linear equation that we defined previously, linear regression can be
understood as finding the best-fitting straight line through the training examples.
This best-fitting line is also called the regression line, and the vertical lines
from the regression line to the training examples are the so-called offsets or
residuals—the errors of our prediction.
"""

"""MULTIVARIATE LINEAR REGRESSION"""
"""
We have no good means of visualizing hyperplanes with two dimensions in a scatterplot
(multiple linear regression models fit to datasets with three or more features).
y = somme(w_i x x) + b
"""

"""DATASET AMES HOUSING"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.dropna(axis=0)

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

"""LINEAR REGRESSION SCIKITLEARN"""
"""
Many of scikit-learn’s estimators for regression make use of the least squares
implementation in SciPy, which, in turn, uses highly optimized code optimizations
based on the Linear Algebra Package (LAPACK). The linear regression implementation
in scikit-learn also works (better) with unstandardized variables, since it does
not use (S)GD-based optimization, so we can skip the standardization step.
"""

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print(f'Slope: {slr.coef_[0]:.3f}')
print(f'Intercept: {slr.intercept_:.3f}')

"""
Scikit-learn’s LinearRegression model, fitted with the unstandardized Gr Liv Area
and SalePrice variables, yielded different model coefficients, since the features
have not been standardized.
"""
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
lin_regplot(X, y, slr)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.tight_layout()
plt.show()
