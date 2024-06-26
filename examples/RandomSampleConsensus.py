"""RANDOM SAMPLE CONSENSUS (RANSAC)"""
"""
As an alternative to throwing out outliers, we will look at a robust method of
regression using the RANdom SAmple Consensus (RANSAC) algorithm, which fits a
regression model to a subset of the data, the so-called inliers.
We can summarize the iterative RANSAC algorithm as follows :
1.  Select a random number of examples to be inliers and fit the model.
2.  Test all other data points against the fitted model and add those points that
    fall within a user-given tolerance to the inliers.
3.  Refit the model using all inliers.
4.  Estimate the error of the fitted model versus the inliers.
5.  Terminate the algorithm if the performance meets a certain user-defined
    threshold or if a fixed number of iterations was reached ; go back to step 1
    otherwise.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


"""RANDOM SAMPLE CONSENSUS SCIKITLEARN"""
"""
Let’s now use a linear model in combination with the RANSAC algorithm as implemented
in scikit-learn’s RANSACRegressor class.
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100, # default value
    min_samples=0.95,
    residual_threshold=None, # default value
    random_state=123)
ransac.fit(X, y)
"""
We set the maximum number of iterations of the RANSACRegressor to 100, and using
min_samples=0.95, we set the minimum number of the randomly chosen training examples
to be at least 95 percent of the dataset.
"""

"""MEDIAN ABSOLUTE DEVIATION (MAD)"""
"""
By default (via residual_threshold=None), scikit-learn uses the MAD estimate to
select the inlier threshold, where MAD stands for the median absolute deviation
of the target values, y. However, the choice of an appropriate value for the inlier
threshold is problem-specific, which is one disadvantage of RANSAC.
Once we have fitted the RANSAC model, let’s obtain the inliers and outliers from
the fitted RANSAC linear regression model and plot them together with the linear
fit.
"""
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f'Slope: {ransac.estimator_.coef_[0]:.3f}')
print(f'Intercept: {ransac.estimator_.intercept_:.3f}')

"""
Remember that we set the residual_threshold parameter to None, so RANSAC was using
the MAD to compute the threshold for flagging inliers and outliers. The MAD, for
this dataset, can be computed as follows :
"""
def median_absolute_deviation(data):
    return np.median(np.abs(data - np.median(data)))
median_absolute_deviation(y)
"""
So, if we want to identify fewer data points as outliers, we can choose a
residual_threshold value greater than the preceding MAD.
Using RANSAC, we reduced the potential effect of the outliers in this dataset,
but we don’t know whether this approach will have a positive effect on the
predictive performance for unseen data or not.
"""
