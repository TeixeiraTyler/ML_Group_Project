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


class Filterer:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def filterData(self):
        # Dropping ID column as it is not needed
        self.trainingData = self.trainingData.drop(['Id'], axis=1)
        self.testingData = self.testingData.drop(['Id'], axis=1)

        # Handle the removing of labels before we start the rest of the preprocessing steps
        labelsToRemove = ['Utilities']
        self.trainingData = self.trainingData.drop(labelsToRemove, axis=1)
        self.testingData = self.testingData.drop(labelsToRemove, axis=1)
        self.combinedData = [self.trainingData, self.testingData]

        isolateOutliers(self.trainingData)

        return DataObject(self.trainingData, self.testingData, self.combinedData)


def isolateOutliers(dataset):
    iso = IsolationForest(contamination=0.01)
    iso.fit(dataset)
    dataset["outlier"] = pd.Series(iso.predict(dataset))
    print(iso)
