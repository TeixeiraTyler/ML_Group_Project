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


class MappingOrdinalFeatures:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		self.trainingData = self.map_ordinals(self.trainingData)
		self.testingData = self.map_ordinals(self.testingData)
		self.combinedData = [self.trainingData, self.testingData]

		return DataObject(self.trainingData, self.testingData, self.combinedData)

	def map_ordinals(self, data):
		# LandSlope: Slope of property
		LandSlope = {}
		LandSlope['Gtl'] = 3  # 'Gentle slope'
		LandSlope['Mod'] = 2  # 'Moderate Slope'
		LandSlope['Sev'] = 1  # 'Severe Slope'

		data.LandSlope = data.LandSlope.map(LandSlope)

		# ExterQual: Evaluates the quality of the material on the exterior
		ExterQual = {}
		ExterQual['Ex'] = 5  # 'Excellent'
		ExterQual['Gd'] = 4  # 'Good'
		ExterQual['TA'] = 3  # 'Average/Typical'
		ExterQual['Fa'] = 2  # 'Fair'
		ExterQual['Po'] = 1  # 'Poor'
		ExterQual['NA'] = 0  # 'NA'

		data.ExterQual = data.ExterQual.map(ExterQual)

		# ExterCond: Evaluates the present condition of the material on the exterior
		data.ExterCond = data.ExterCond.map(ExterQual)

		# HeatingQC: Heating quality and condition
		data.HeatingQC = data.HeatingQC.map(ExterQual)

		# KitchenQual: Kitchen quality
		data.KitchenQual = data.KitchenQual.map(ExterQual)

		# FireplaceQu: Fireplace quality
		data.FireplaceQu = data.FireplaceQu.map(ExterQual)

		# GarageCond: Garage Conditionals
		data.GarageCond = data.GarageCond.map(ExterQual)

		PavedDrive = {}
		PavedDrive['Y'] = 3  # 'Paved'
		PavedDrive['P'] = 2  # 'Partial Pavement'
		PavedDrive['N'] = 1  # 'Dirt/Gravel'

		data.PavedDrive = data.PavedDrive.map(PavedDrive)

		# LotShape: General shape of property
		LotShape = {}
		LotShape['Reg'] = 4  # 'Regular'
		LotShape['IR1'] = 3  # 'Slightly irregular'
		LotShape['IR2'] = 2  # 'Moderately Irregular'
		LotShape['IR3'] = 1  # 'Irregular'

		data.LotShape = data.LotShape.map(LotShape)

		# BsmtQual: Evaluates the height of the basement
		BsmtQual = {}
		BsmtQual['Ex'] = 5  # 'Excellent (100+ inches)'
		BsmtQual['Gd'] = 4  # 'Good (90-99 inches)'
		BsmtQual['TA'] = 3  # 'Typical (80-89 inches)'
		BsmtQual['Fa'] = 2  # 'Fair (70-79 inches)'
		BsmtQual['Po'] = 1  # 'Poor (<70 inches'
		BsmtQual['NA'] = 0  # 'No Basement'

		data.BsmtQual = data.BsmtQual.map(BsmtQual)

		# BsmtCond: Evaluates the general condition of the basement
		data.BsmtCond = data.BsmtCond.map(BsmtQual)

		# GarageQual: Garage quality
		data.GarageQual = data.GarageQual.map(BsmtQual)

		# PoolQC: Pool quality
		data.PoolQC = data.PoolQC.map(BsmtQual)

		# BsmtExposure: Refers to walkout or garden level walls
		BsmtExposure = {}
		BsmtExposure['Gd'] = 4  # 'Good Exposure'
		BsmtExposure['Av'] = 3  # 'Average Exposure (split levels or foyers typically score average or above)'
		BsmtExposure['Mn'] = 2  # 'Mimimum Exposure'
		BsmtExposure['No'] = 1  # 'No Exposure'
		BsmtExposure['NA'] = 0  # 'No Basement'

		data.BsmtExposure = data.BsmtExposure.map(BsmtExposure)

		# BsmtFinType1: Rating of basement finished area
		BsmtFinType1 = {}
		BsmtFinType1['GLQ'] = 6  # 'Good Living Quarters'
		BsmtFinType1['ALQ'] = 5  # 'Average Living Quarters'
		BsmtFinType1['BLQ'] = 4  # 'Below Average Living Quarters'
		BsmtFinType1['Rec'] = 3  # 'Average Rec Room'
		BsmtFinType1['LwQ'] = 2  # 'Low Quality'
		BsmtFinType1['Unf'] = 1  # 'Unfinshed'
		BsmtFinType1['NA'] = 0  # 'No Basement'

		data.BsmtFinType1 = data.BsmtFinType1.map(BsmtFinType1)

		# BsmtFinType2: Rating of basement finished area (if multiple types)
		data.BsmtFinType2 = data.BsmtFinType2.map(BsmtFinType1)

		# CentralAir: Central air conditioning
		# Since with this transformatio as the same as binarize this feature
		CentralAir = {}
		CentralAir['N'] = 0
		CentralAir['Y'] = 1

		data.CentralAir = data.CentralAir.map(CentralAir)

		# GarageFinish: Interior finish of the garage
		GarageFinish = {}
		GarageFinish['Fin'] = 3  # 'Finished'
		GarageFinish['RFn'] = 2  # 'Rough Finished'
		GarageFinish['Unf'] = 1  # 'Unfinished'
		GarageFinish['NA'] = 0  # 'No Garage'

		data.GarageFinish = data.GarageFinish.map(GarageFinish)

		# Functional: Home functionality
		Functional = {}
		Functional['Typ'] = 7  # Typical Functionality
		Functional['Min1'] = 6  # Minor Deductions 1
		Functional['Min2'] = 5  # Minor Deductions 2
		Functional['Mod'] = 4  # Moderate Deductions
		Functional['Maj1'] = 3  # Major Deductions 1
		Functional['Maj2'] = 2  # Major Deductions 2
		Functional['Sev'] = 1  # Severely Damaged
		Functional['Sal'] = 0  # Salvage only

		data.Functional = data.Functional.map(Functional)

		# Street: Type of road access to property
		# Since with this transformatio as the same as binarize this feature
		Street = {}
		Street['Grvl'] = 0  # Gravel
		Street['Pave'] = 1  # Paved

		data.Street = data.Street.map(Street)

		# Fence: Fence quality
		Fence = {}
		Fence['GdPrv'] = 5  # 'Good Privacy'
		Fence['MnPrv'] = 4  # 'Minimum Privacy'
		Fence['GdWo'] = 3  # 'Good Wood'
		Fence['MnWw'] = 2  # 'Minimum Wood/Wire'
		Fence['NA'] = 1  # 'No Fence'

		data.Fence = data.Fence.map(Fence)
		# But No Fence has the higest median Sales Price. So I try to use it as categorical
		return data