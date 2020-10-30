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

from DataObject import DataObject


class Converter:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def convertData(self):
        self.trainingData = self.mapCategoricalToOrdinal(self.trainingData)
        self.testingData = self.mapCategoricalToOrdinal(self.testingData)
        self.combinedData = [self.trainingData, self.testingData]
        return DataObject(self.trainingData, self.testingData, self.combinedData)


    def mapCategoricalToOrdinal(self, dataset):
        ordinal_label = ['MSZoning','LandSlope','LotConfig','Utilities','LandContour','LotShape','Alley','Street','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
        cat = []
        for i in ordinal_label:
            unique = dataset[i].unique()
            categ = dataset[i].astype("category").cat.categories
            cat.append({i:ix for ix,i in enumerate(categ)})
            dataset[i] = dataset[i].astype("category").cat.codes
        return dataset
