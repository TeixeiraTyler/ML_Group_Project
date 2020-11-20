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


class ExploratoryDataAnalysis:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		self.trainingData.drop("Id", axis = 1, inplace = True)
		self.testingData.drop("Id", axis = 1, inplace = True)

		self.trainingData.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
		self.testingData.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

		self.testingData['SalePrice'] = 0

		#Deleting outliers
		self.trainingData = self.trainingData.drop(self.trainingData[(self.trainingData.GrLivArea>4000) & (self.trainingData.SalePrice<300000)].index)

		self.trainingData = self.trainingData[self.trainingData.GrLivArea * self.trainingData.TotRmsAbvGrd < 45000]

		self.trainingData = self.trainingData[self.trainingData.GarageArea * self.trainingData.GarageCars < 3700]

		self.trainingData = self.trainingData[(self.trainingData.FullBath + (self.trainingData.HalfBath*0.5) + self.trainingData.BsmtFullBath + (self.trainingData.BsmtHalfBath*0.5))<5]

		self.trainingData = self.trainingData.loc[~(self.trainingData.SalePrice==392500.0)]
		self.trainingData = self.trainingData.loc[~((self.trainingData.SalePrice==275000.0) & (self.trainingData.Neighborhood=='Crawfor'))]
		self.trainingData.SalePrice = np.log1p(self.trainingData.SalePrice)
		
		self.combinedData = [self.trainingData, self.testingData]
		return DataObject(self.trainingData, self.testingData, self.combinedData)
