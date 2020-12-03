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

from Utils import Utils
from Preprocessing.Correlator import Correlator
from Preprocessing.Filler import Filler
from Preprocessing.Converter import Converter
from Preprocessing.Encoder import Encoder
from Preprocessing.Filterer import Filterer
from Preprocessing.DataObject import DataObject
from Preprocessing.PreliminaryDataAdjuster import PreliminaryDataAdjuster
from Preprocessing.OrdinalToNumericalConverter import OrdinalToNumericalConverter
from Preprocessing.FeatureEngineer import FeatureEngineer
from Preprocessing.SelectFeatures import SelectFeatures
from Preprocessing.Modeling import Modeling

# main class for preprocessing data
class Preprocessor:
	def __init__(self, trainingData, testingData):
		self.trainingData = trainingData
		self.testingData = testingData
		self.combinedData = [trainingData, testingData]

	# main function that combines all preprocessing.
	def process(self, test_ID):
		dataObject = DataObject(self.trainingData, self.testingData, self.combinedData)

		prelim = PreliminaryDataAdjuster(dataObject)
		dataObject = prelim.go()

		converter = OrdinalToNumericalConverter(dataObject)
		dataObject = converter.go()

		creator = FeatureEngineer(dataObject)
		dataObject, dataObject.combinedData, y_train, cols, colsP = creator.go()



		step7 = SelectFeatures(dataObject)
		dataObject, totalCols, RFEcv, XGBestCols = step7.go(dataObject.combinedData, cols, colsP)
		
		step9 = Modeling(dataObject)
		ouput_ensembled = step9.go(all_data, totalCols, test_ID, colsP, RFEcv, XGBestCols)

		ouput_ensembled.to_csv('SalePrice_N_submission.csv', index=False)

		print(dataObject.trainingData)

		return dataObject
