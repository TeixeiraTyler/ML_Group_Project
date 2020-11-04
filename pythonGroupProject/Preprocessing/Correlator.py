import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from DataObject import DataObject


class Correlator:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def correlateData(self):
        # correlatedData = self.performCorrelation(self.trainingData)
        # self.trainingData = filterByCorrelation(self.trainingData, correlatedData)
        # self.trainingData = self.filterByCorrelation(self.trainingData)
        return DataObject(self.trainingData, self.testingData, self.combinedData)

    def performCorrelation(self, dataset):
        correlatedData = dataset[dataset.SalePrice>1].corr()
        top_correlation_columns = correlatedData[abs((correlatedData.SalePrice)>=.26)].SalePrice.sort_values(ascending=False).keys()
        top_correlated_data = correlatedData.loc[top_correlation_columns, top_correlation_columns]
        dropSelf = np.zeros_like(top_correlated_data)
        dropSelf[np.triu_indices_from(dropSelf)] = True
        return top_correlated_data

    def filterByCorrelation(self, dataset):
        dataset.drop(['FireplaceQu', 'BsmtSFPoints', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OverallQual', 'GrLivArea',
                       'TotalBsmtSF_x_Bsm', '1stFlrSF', 'PoolArea', 'LotArea', 'SaleCondition_Partial',
                       'Exterior1st_VinylSd', 'GarageCond', 'HouseStyle_2Story', 'BsmtSFMultPoints', 'ScreenPorch',
                       'LowQualFinSF', 'BsmtFinSF2', 'TSsnPorch'], axis=1, inplace=True)
        return dataset