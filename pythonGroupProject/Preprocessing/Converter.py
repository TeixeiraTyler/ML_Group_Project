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
        self.trainingData = mapCategoricalToOrdinal(self.trainingData)
        self.testingData = mapCategoricalToOrdinal(self.testingData)
        self.combinedData = [self.trainingData, self.testingData]
        return DataObject(self.trainingData, self.testingData, self.combinedData)


def mapCategoricalToOrdinal(dataset):
    number = LabelEncoder()
    # objectTypeLabels = dataset.select_dtypes(include=['object']).columns
    # for label in objectTypeLabels:
    # dataset[label] = number.fit_transform(dataset[label].astype('str'))

    dataset['MSZoning'] = number.fit_transform(dataset['MSZoning'].astype('str'))
    dataset['Street'] = number.fit_transform(dataset['Street'].astype('str'))
    dataset['Alley'] = number.fit_transform(dataset['Alley'].astype('str'))
    dataset['LotShape'] = number.fit_transform(dataset['LotShape'].astype('str'))
    dataset['LandContour'] = number.fit_transform(dataset['LandContour'].astype('str'))
    dataset['PoolQC'] = number.fit_transform(dataset['PoolQC'].astype('str'))

    return dataset
