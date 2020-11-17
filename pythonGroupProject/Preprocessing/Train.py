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


class Train:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def go(self):
        # Main script here, said to use all data here, all_data....[], error doesn't know cols???
        y_train = (self.combinedData.SalePrice[self.combinedData.SalePrice>0].reset_index(drop=True, inplace=False))
        # Data with Polynomials
        train = self.combinedData.loc[(self.combinedData.SalePrice>0), cols].reset_index(drop=True, inplace=False)
        test = self.combinedData.loc[(self.combineData.SalePrice==0), cols].reset_index(drop=True, inplace=False)
 