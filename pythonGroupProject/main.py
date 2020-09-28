import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from Utils import Utils
from Preprocessing.Preprocessor import Preprocessor

# END OF IMPORTS #

# Read in data
trainingData = pd.read_csv('train.csv')
testingData = pd.read_csv('test.csv')

# Preprocessing
# Takes in the read data and spits out processed data
preprocessor = Preprocessor(trainingData, testingData)
dataObject = preprocessor.process()
trainingData = dataObject.trainingData
testingData = dataObject.testingData
combinedData = dataObject.combinedData






# print(dataset['KitchenQual'].isnull().sum())
# trainingData.info()
# print('_'*40)
# testingData.info()

# print()
# print(trainingData.describe())
# print(testingData.describe())

# print()
# print(trainingData.describe(include=['O']))
# print(testingData.describe(include=['O']))

# print()
# print(trainingData[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], as_index=False).mean().sort_values(by='SalePrice', ascending=False))

# print()
# details = trainingData.corr()['SalePrice']
# print(details.sort_values(ascending=False))

# print()
# print(trainingData.describe().transpose())

# Utils.plotData(trainingData, 'MSZoning', 'SalePrice')
# Utils.plotData(testingData, 'MSZoning', 'SalePrice')

# g = sns.FacetGrid(testingData)
# g.map(plt.hist, 'MSZoning')
# plt.show()

