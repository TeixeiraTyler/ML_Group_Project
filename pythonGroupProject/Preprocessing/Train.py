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

from Preprocessing.DataObject import DataObject


class Train:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		all_data = pd.concat((self.trainingData, self.testingData)).reset_index(drop=True)
		# Main script here, said to use all data here, all_data....[], error doesn't know cols???
		y_train = (all_data.SalePrice[all_data.SalePrice>0].reset_index(drop=True, inplace=False))

		# Data with Polynomials
		train = all_data.loc[(all_data.SalePrice>0), cols].reset_index(drop=True, inplace=False)
		test = all_data.loc[(all_data.SalePrice==0), cols].reset_index(drop=True, inplace=False)
 