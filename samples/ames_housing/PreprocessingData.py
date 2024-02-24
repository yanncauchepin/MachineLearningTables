import pandas as pd
import numpy as np

"""DATASET AMES HOUSING"""
"""
The Ames Housing dataset consists of 2,930 examples and 80 features.
The features we will be working with, including the target variable, are as follows :
-   Overall Qual: Rating for the overall material and finish of the house on a scale
    from 1 (very poor) to 10 (excellent)
-   Overall Cond: Rating for the overall condition of the house on a scale from
    1 (very poor) to 10 (excellent)
-   Gr Liv Area: Above grade (ground) living area in square feet
-   Central Air: Central air conditioning (N=no, Y=yes)
-   Total Bsmt SF: Total square feet of the basement area
-   SalePrice: Sale price in U.S. dollars ($)
"""

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)
df.head()
df.shape

"""PREPROCESSING DATA"""

df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
df.isnull().sum()
df = df.dropna(axis=0)
df.isnull().sum()
