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
from sklearn.base import TransformerMixin
from Utils import Utils

from Preprocessing.DataObject import DataObject


# Removes outliers and unnecessary columns and fills n/a values.
class PreliminaryDataAdjuster:
	def __init__(self, combinedData):
		self.combinedData = combinedData

	def go(self):
		self.combinedData.drop("Id", axis = 1, inplace = True)

		self.combinedData.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

		# self.testingData['SalePrice'] = 0

		self.combinedData = self.combinedData.drop(self.combinedData[self.combinedData.SalePrice < 300000].index)
		self.combinedData = self.combinedData.drop(self.combinedData[(self.combinedData.GrLivArea > 4000)].index)
		self.combinedData = self.combinedData[self.combinedData.GarageArea * self.combinedData.GarageCars < 3700]
		self.combinedData = self.combinedData[self.combinedData.GrLivArea * self.combinedData.TotRmsAbvGrd < 45000]
		self.combinedData = self.combinedData[(self.combinedData.FullBath + (self.combinedData.HalfBath*0.5) + self.combinedData.BsmtFullBath + (self.combinedData.BsmtHalfBath*0.5)) < 5]

		self.combinedData = self.combinedData.loc[~(self.combinedData.SalePrice==392500.0)]
		self.combinedData = self.combinedData.loc[~((self.combinedData.SalePrice==275000.0) & (self.combinedData.Neighborhood=='Crawfor'))]
		self.combinedData.SalePrice = np.log1p(self.combinedData.SalePrice)

		self.combinedData = self.fillMissingData(self.combinedData)

		return self.combinedData

	def fillMissingData(self, dataset):
		labelsToFillWithNA = ['Alley', 'Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu']

		# Utils.printDatasetNulls(dataset)

		# Handle missing values
		dataset = self.fillMSZoningMissingValues(dataset)
		dataset = self.fillLotFrontageMissingValues(dataset)
		dataset = self.fillMasonryVeneerMissingValues(dataset)
		dataset = self.fillExteriorCoveringMissingValues(dataset)
		dataset = self.fillBasementFeaturesMissingValues(dataset)
		dataset = self.fillElectricalMissingValues(dataset)
		dataset = self.fillKitchenQualityMissingValues(dataset)
		dataset = self.fillGarageFeaturesMissingValues(dataset)
		dataset = self.fillPoolQualityMissingValues(dataset)
		dataset = self.fillSaleTypeMissingValues(dataset)

		# Handle NULL values
		dataset = Utils.fillNullLabels(dataset, labelsToFillWithNA, 'NA')
		dataset = Utils.fillNullLabels(dataset, ['Functional'],
									   'Typ')  # data_description.txt tells us to assume typical 'typ'

		# Utils.printDatasetNulls(dataset)

		return dataset

	# This function handles the MSZoning missing values
	# Since missing values are small for this feature we will just fill with the most
	#    common value in the dataset
	def fillMSZoningMissingValues(self, dataset):
		mostFrequentZoningValue = dataset.MSZoning.dropna().mode()[0]
		dataset['MSZoning'] = dataset['MSZoning'].fillna(mostFrequentZoningValue)

		return dataset

	# This function handles the Lot Frontage Features missing values
	# We are going to first group the Lot Frontage Features by neighborhood
	# Then we are going to fill the missing values with the mean Lot Frontage in the neighborhood
	#
	# We are grouping by neighborhood because houses in a neighborhood have similiar LotFrontage values
	def fillLotFrontageMissingValues(self, dataset):
		neighborhoodLotFrontageMeans = dataset.groupby('Neighborhood').LotFrontage.mean()
		lotFrontageValues = (dataset.loc[dataset.LotFrontage.isnull(), ['Neighborhood']]).transpose()

		lotFrontageFeature = dataset['LotFrontage'].copy()
		for i in lotFrontageValues:
			lotFrontageFeature[lotFrontageValues[i].name] = neighborhoodLotFrontageMeans[lotFrontageValues[i].values[0]]

		dataset['LotFrontage'] = lotFrontageFeature
		return dataset

	def fillMasonryVeneerMissingValues(self, dataset):
		masonryVeneerCase1NULL = (
			dataset.loc[(dataset.MasVnrType.isnull()) & (dataset.MasVnrArea > 0), ['MasVnrType']]).transpose()
		masonryVeneerCase1None = (
			dataset.loc[(dataset.MasVnrType == 'None') & (dataset.MasVnrArea > 0), ['MasVnrType']]).transpose()
		masonryVeneerCase1 = masonryVeneerCase1NULL.append(masonryVeneerCase1None)
		masonryVeneerCase2 = (
			dataset.loc[(dataset.MasVnrType != 'None') & (dataset.MasVnrArea == 0), ['MasVnrArea']]).transpose()
		medianOfMasonryVeneerCase2 = \
			dataset.loc[(dataset.MasVnrType != 'None') & (dataset.MasVnrArea > 0), ['MasVnrArea']].median()[0]

		mostCommon = dataset['MasVnrType'].value_counts().index[0]
		if (mostCommon == 'None'):
			mostCommon = dataset['MasVnrType'].value_counts().index[1]

		masVnrTypeFeature = dataset['MasVnrType'].copy()
		for i in masonryVeneerCase1:
			masVnrTypeFeature[masonryVeneerCase1[i].name] = mostCommon

		masVnrAreaFeature = dataset['MasVnrArea'].copy()
		for i in masonryVeneerCase2:
			masVnrAreaFeature[masonryVeneerCase2[i].name] = medianOfMasonryVeneerCase2

		dataset['MasVnrType'] = masVnrTypeFeature
		dataset['MasVnrArea'] = masVnrAreaFeature

		dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
		dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
		return dataset

	# This function handles the Garage Features missing values
	# Need to handle the following cases:
	#    1. Fill GarageType NULL values with 'NA'
	#    2. Handle case where GarageType is Detchd but the rest of the row is NULL
	def fillGarageFeaturesMissingValues(self, dataset):
		dataset = Utils.fillNullLabels(dataset, ['GarageType'], 'NA')

		dataset = self.fillGarageFeatureValue(dataset, 'GarageYrBlt', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageFinish', 'mode')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageCars', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageArea', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageQual', 'mode')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageCond', 'mode')

		return dataset

	def fillGarageFeatureValue(self, dataset, feature, fillType):
		if (fillType == 'median'):
			fillValue = dataset[dataset.GarageType == 'Detchd'][feature].median()
			fillnaValue = 0
		elif (fillType == 'mode'):
			fillValue = dataset[dataset.GarageType == 'Detchd'][feature].mode()[0]
			fillnaValue = 'NA'

		condition = (dataset.GarageType == 'Detchd') & (dataset[feature].isnull())
		values = (dataset.loc[condition, [feature]]).transpose()

		datasetFeature = dataset[feature].copy()
		for i in values:
			datasetFeature[values[i].name] = fillValue

		dataset[feature] = datasetFeature
		dataset[feature] = dataset[feature].fillna(fillnaValue)
		return dataset

	# This function handles the cases where the PoolArea is greater than zero but PoolQC is NULL
	# We are going to use the OverallQuality feature of the house to determine the pool quality
	# The Pool quality has 5 categorical features and OverallQuality has 10 categorical features.
	#   We will divide OverallQuality by 2 to get the correlated pool quality value
	def fillPoolQualityMissingValues(self, dataset):
		poolQualityMap = {0: 'NA', 1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}
		poolQuality = (
			(dataset.loc[(dataset.PoolArea > 0) & (dataset.PoolQC.isnull()), ['OverallQual']] / 2).round()).transpose()

		poolQCFeature = dataset['PoolQC'].copy()
		for i in poolQuality:
			poolQCFeature[i] = poolQuality[i].map(poolQualityMap)

		dataset['PoolQC'] = poolQCFeature
		return dataset

	# This function handles the cases where kitchen quality is missing
	# Currently there is only one value missing and it is in the testing set,
	#   so we will just fill with the most common value in the dataset
	def fillKitchenQualityMissingValues(self, dataset):
		kitchenQualityMode = dataset.KitchenQual.mode()[0]
		kitchenQuality = (
			dataset.loc[(dataset.KitchenAbvGr > 0) & (dataset.KitchenQual.isnull()), ['KitchenQual']]).transpose()

		kitchenQualFeature = dataset['KitchenQual'].copy()
		for i in kitchenQuality:
			kitchenQualFeature[kitchenQuality[i].name] = kitchenQualityMode

		dataset['KitchenQual'] = kitchenQualFeature
		return dataset

	def fillSaleTypeMissingValues(self, dataset):
		mode = dataset['SaleType'].mode()[0]
		dataset['SaleType'] = dataset['SaleType'].fillna(mode)
		return dataset

	# This function is called in 'fillBasementFeaturesMissingValues'
	# It is for setting basement features with most common when they
	#    are equal to null but the basement has a non zero area
	def fillBasementFeatureWithMostCommon(self, dataset, feature, condition):
		mostCommon = dataset[feature].value_counts().index[0]
		if (mostCommon == 'No'):
			mostCommon = dataset[feature].value_counts().index[1]

		values = (dataset.loc[condition, [feature]]).transpose()

		datasetFeature = dataset[feature].copy()
		for i in values:
			datasetFeature[values[i].name] = mostCommon

		dataset[feature] = datasetFeature
		return dataset

	# This function handles the Basement Features missing values
	# Need to handle the following cases:
	#    1. TotalBsmtSF is greater than zero but BsmtExposure is NULL
	#    2. TotalBsmtSF is greater than zero but BsmtQual is NULL
	#    3. TotalBsmtSF is greater than zero but BsmtCond is NULL
	#    4. BsmtFinSF2 is greater than zero but BsmtFinType2 is NULL
	#    5. BsmtFinSF2 is zero and BsmtUnfSF is not zero but BsmtFinType2 is finished
	#       - Set BsmtFinSF2 = BsmtUnfSF and set BsmtUnfSF = 0
	#    6. Fill categorical basement data with NA
	#    7. Fill numerical basement data with 0
	def fillBasementFeaturesMissingValues(self, dataset):
		bsmtQualCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtQual.isnull())
		bsmtCondCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtCond.isnull())
		bsmtExposureCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtExposure.isnull())
		bsmtFinType2Condition = (dataset.BsmtFinSF2 > 0) & (dataset.BsmtFinType2.isnull())
		bsmtFinSF2Condition = (dataset.BsmtFinSF2 == 0) & (dataset.BsmtFinType2 != 'Unf') & (
			~dataset.BsmtFinType2.isnull())

		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtQual', bsmtQualCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtCond', bsmtCondCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtExposure', bsmtExposureCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtFinType2', bsmtFinType2Condition)
		dataset = self.handleBsmtFinSF2SpecialCase(dataset, bsmtFinSF2Condition)

		basementFeaturesToFillWithNA = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
		basementFeaturesToFillWith0 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
									   'BsmtHalfBath']
		dataset = Utils.fillNullLabels(dataset, basementFeaturesToFillWithNA, 'NA')
		dataset = Utils.fillNullLabels(dataset, basementFeaturesToFillWith0, 0)

		return dataset

	# This function handles the missing values of Electrical
	# Since there are very few missing values we will fill with the most common
	def fillElectricalMissingValues(self, dataset):
		mostCommon = dataset['Electrical'].value_counts().index[0]
		dataset['Electrical'] = dataset['Electrical'].fillna(mostCommon)

		return dataset

	def fillExteriorCoveringMissingValues(self, dataset):
		exterior1stMode = dataset['Exterior1st'].mode()[0]
		exterior2ndMode = dataset['Exterior2nd'].mode()[0]
		dataset['Exterior1st'] = dataset['Exterior1st'].fillna(exterior1stMode)
		dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(exterior2ndMode)

		return dataset



	# This function is called in 'fillBasementFeaturesMissingValues'
	# It handles the case where BsmtFinSF2 is zero and BsmtUnfSF is zero,
	#   but BsmtFinType2 is finished
	def handleBsmtFinSF2SpecialCase(self, dataset, condition):
		values = (dataset.loc[condition, ['BsmtFinSF2', 'BsmtUnfSF']]).transpose()

		bsmtFinSF2Feature = dataset['BsmtFinSF2'].copy()
		bsmtUnfSFFeature = dataset['BsmtUnfSF'].copy()
		for i in values:
			currentUnfSFValue = values[i].values[1]
			bsmtFinSF2Feature[values[i].name] = currentUnfSFValue
			bsmtUnfSFFeature[values[i].name] = 0

		dataset['BsmtFinSF2'] = bsmtFinSF2Feature
		dataset['BsmtUnfSF'] = bsmtUnfSFFeature
		return dataset


# Currently unused
class DT(TransformerMixin):

	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].value_counts().index[0]
			if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
			index=X.columns)

		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)
