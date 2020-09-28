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


class Encoder:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def encodeData(self):
        return DataObject(self.trainingData, self.testingData, self.combinedData)
