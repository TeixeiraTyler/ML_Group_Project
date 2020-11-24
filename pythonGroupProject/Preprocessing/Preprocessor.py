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

from Preprocessing.Correlator import Correlator
from Preprocessing.Filler import Filler
from Preprocessing.Converter import Converter
from Preprocessing.Encoder import Encoder
from Preprocessing.Filterer import Filterer
from Preprocessing.DataObject import DataObject
from Preprocessing.ExploratoryDataAnalysis import ExploratoryDataAnalysis
from Preprocessing.CheckDataQuality import CheckDataQuality
from Preprocessing.MappingOrdinalFeatures import MappingOrdinalFeatures
from Preprocessing.FeatureEngineering import FeatureEngineering
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

		# Step 1 is preparing environment

		step2 = ExploratoryDataAnalysis(dataObject)
		dataObject = step2.go()

		step3 = CheckDataQuality(dataObject)
		dataObject = step3.go()

		step4 = MappingOrdinalFeatures(dataObject)
		dataObject = step4.go()

		step5 = FeatureEngineering(dataObject)
		dataObject, all_data, y_train, cols, colsP = step5.go()

		#step6 = Train(dataObject)
		#dataObject = step6.go()

		step7 = SelectFeatures(dataObject)
		dataObject, totalCols, RFEcv, XGBestCols = step7.go(all_data, cols, colsP)

		#step8 = CompressData(dataObject)
		#dataObject = step8.go(y_train)
		
		#input Dataobject
		#Dataobject.trainingData = train 
		#Dataobject.testingData = y_train
		
		step9 = Modeling(dataObject)
		ouput_ensembled = step9.go(all_data, totalCols, test_ID, colsP, RFEcv, XGBestCols)


		ouput_ensembled.to_csv('SalePrice_N_submission.csv', index = False)
		# filler = Filler(dataObject)
		# dataObject = filler.fillMissingData()
		#
		# converter = Converter(dataObject)
		# dataObject = converter.convertData()
		#
		# filterer = Filterer(dataObject)
		# dataObject = filterer.filterData()
		#
		# encoder = Encoder(dataObject)
		# dataObject = encoder.encode()
		#
		# correlator = Correlator(dataObject)
		# dataObject = correlator.correlateData()

		print(dataObject.trainingData)

		return dataObject
