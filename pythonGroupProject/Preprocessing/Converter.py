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

from Preprocessing.DataObject import DataObject


class Converter:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def convertData(self):
		self.trainingData = self.mapCategoricalToOrdinal(self.trainingData)
		self.testingData = self.mapCategoricalToOrdinal(self.testingData)

		self.trainingData = self.changeYearsToAge(self.trainingData)
		self.testingData = self.changeYearsToAge(self.testingData)

		self.trainingData = self.addRemodAndConvertAge(self.trainingData)
		self.testingData = self.addRemodAndConvertAge(self.testingData)

		self.trainingData = self.defineUint8Types(self.trainingData)
		self.testingData = self.defineUint8Types(self.testingData)

		self.combinedData = [self.trainingData, self.testingData]
		return DataObject(self.trainingData, self.testingData, self.combinedData)

	def mapCategoricalToOrdinal(self, dataset):
		ordinal_label = ['LandSlope', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
						 'FireplaceQu', 'GarageCond', 'PavedDrive', 'LotShape', 'BsmtQual', 'BsmtCond', 'GarageQual',
						 'PoolQC', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'CentralAir', 'GarageFinish',
						 'Functional',
						 'Street', 'Fence']
		number = LabelEncoder()
		for i in ordinal_label:
			dataset[i] = number.fit_transform(dataset[i].astype('str'))
		return dataset

	def changeYearsToAge(self, dataset):
		dataset.YearBuilt = self.ageYears(dataset.YearBuilt)
		dataset.YearRemodAdd = self.ageYears(dataset.YearRemodAdd)
		dataset.GarageYrBlt = self.ageYears(dataset.GarageYrBlt)
		dataset.YrSold = self.ageYears(dataset.YrSold)
		return dataset

	def ageYears(self, feature):
		StartingYear = 2011
		return feature.apply(lambda x: 0 if x == 0 else (StartingYear - x))

	def addRemodAndConvertAge(self, dataset):
		dataset['Remod'] = -1
		dataset.loc[(dataset.YearBuilt == dataset.YearRemodAdd), ['Remod']] = 0
		dataset.loc[(dataset.YearBuilt != dataset.YearRemodAdd), ['Remod']] = 1

		dataset['Age'] = dataset.YearRemodAdd - dataset.YrSold

		dataset["IsNew"] = -1
		dataset.loc[(dataset.YearBuilt == dataset.YrSold), ['IsNew']] = 1
		dataset.loc[(dataset.YearBuilt != dataset.YrSold), ['IsNew']] = 0

		return dataset

	def defineUint8Types(self, dataset):
		dataset.CentralAir = dataset.CentralAir.astype('uint8')
		# dataset.Garage_Newest = dataset.Garage_Newest.astype('uint8')
		dataset.EnclosedPorch = dataset.EnclosedPorch.astype('uint8')
		dataset.FullBath = dataset.FullBath.astype('uint8')
		dataset.HalfBath = dataset.HalfBath.astype('uint8')
		dataset.BsmtFullBath = dataset.BsmtFullBath.astype('uint8')
		dataset.BsmtHalfBath = dataset.BsmtHalfBath.astype('uint8')
		dataset.Remod = dataset.Remod.astype('uint8')
		dataset.IsNew = dataset.IsNew.astype('uint8')
		dataset.Street = dataset.Street.astype('uint8')
		dataset.PavedDrive = dataset.PavedDrive.astype('uint8')
		dataset.Functional = dataset.Functional.astype('uint8')
		dataset.LandSlope = dataset.LandSlope.astype('uint8')

		return dataset
