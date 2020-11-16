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
from DataObject import DataObject

# main class for preprocessing data
class Preprocessor:
    def __init__(self, trainingData, testingData):
        self.trainingData = trainingData
        self.testingData = testingData
        self.combinedData = [trainingData, testingData]

    # main function that combines all preprocessing.
    def process(self):
        dataObject = DataObject(self.trainingData, self.testingData, self.combinedData)

        # Step 1 is preparing environment

        step2 = ExploratoryDataAnalysis(dataObject)
        dataObject = step2.go()

        step3 = CheckDataQuality(dataObject)
        dataObject = step3.go()
        Utils.printDatasetNulls(dataObject.trainingData)
        Utils.printDatasetNulls(dataObject.testingData)

        step4 = MappingOrdinalFeatures(dataObject)
        dataObject = step4.go()

        step5 = FeatureEngineering(dataObject)
        dataObject = step5.go()

        step6 = Train(dataObject)
        dataObject = step6.go()

        step7 = SelectFeatures(dataObject)
        dataObject = step7.go()

        step8 = CompressData(dataObject)
        dataObject = step8.go()
        
        #input Dataobject
        #Dataobject.trainingData = train 
		#Dataobject.testingData = y_train
        
        step9 = Modeling(dataObject)
        ouput_ensembled = step9.go()


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
