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
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder

class CompressData:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData
    
    def go(self):
        df = self.trainingData
        scale = RobustScaler() 
        df = scale.fit_transform(train)
        print('With only 120 features: {:6.4%}'.format(sum(pca.explained_variance_ratio_[:120])),"%\n")
        print('After PCA, {:3} features only not explained {:6.4%} of variance ratio from the original {:3}'.format(120,
                                                                                    (sum(pca.explained_variance_ratio_[120:])),
                                                                                    df.shape[1]))
        #dataObject.trainingData = df
        return(dataObject)
        # Main script heres
