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

from Preprocessing.DataObject import DataObject

class DataFrameImputer(TransformerMixin):

	def __init__(self):
		"""
		Impute missing values:
		- Columns of dtype object are imputed with the most frequent value in column.
		- Columns of other types are imputed with mean of column.
		"""
	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].value_counts().index[0]
			if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
			index=X.columns)

		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)


class CheckDataQuality:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		all_data = pd.concat((self.trainingData, self.testingData)).reset_index(drop=True)
		all_data.drop('Utilities', axis=1, inplace=True)
		all_data.Electrical = all_data.Electrical.fillna('SBrkr')

		all_data.GarageType = all_data.GarageType.fillna('NA')

		# Group by GarageType and fill missing value with median where GarageType=='Detchd' and 0 for the others
		cmedian = all_data[all_data.GarageType=='Detchd'].GarageArea.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageArea'] = all_data.loc[all_data.GarageType=='Detchd', 
																				 'GarageArea'].fillna(cmedian)
		all_data.GarageArea = all_data.GarageArea.fillna(0)

		cmedian = all_data[all_data.GarageType=='Detchd'].GarageCars.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageCars'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageCars'].fillna(cmedian)
		all_data.GarageCars = all_data.GarageCars.fillna(0)

		cmedian = all_data[all_data.GarageType=='Detchd'].GarageYrBlt.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageYrBlt'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageYrBlt'].fillna(cmedian)
		all_data.GarageYrBlt = all_data.GarageYrBlt.fillna(0)

		# Group by GarageType and fill missing value with mode where GarageType=='Detchd' and 'NA' for the others
		cmode = all_data[all_data.GarageType=='Detchd'].GarageFinish.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageFinish'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageFinish'].fillna(cmode)
		all_data.GarageFinish = all_data.GarageFinish.fillna('NA')

		cmode = all_data[all_data.GarageType=='Detchd'].GarageQual.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageQual'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageQual'].fillna(cmode)
		all_data.GarageQual = all_data.GarageQual.fillna('NA')

		cmode = all_data[all_data.GarageType=='Detchd'].GarageCond.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageCond'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageCond'].fillna(cmode)
		all_data.GarageCond = all_data.GarageCond.fillna('NA')

		all_data.loc[(all_data.MasVnrType=='None') & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

		# All Types null with Are greater than 0 update to BrkFace type
		all_data.loc[(all_data.MasVnrType.isnull()) & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

		# All Types different from None with Are equal to 0 update to median Area of no None types with Areas
		all_data.loc[(all_data.MasVnrType!='None') & (all_data.MasVnrArea==0), ['MasVnrArea']] = all_data.loc[(all_data.MasVnrType!='None') & (all_data.MasVnrArea>0), ['MasVnrArea']].median()[0]
		# Filling 0 and None for records wheres both are nulls
		all_data.MasVnrArea = all_data.MasVnrArea.fillna(0)
		all_data.MasVnrType = all_data.MasVnrType.fillna('None')

		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtExposure.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtExposure'] = 'Av'
		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtQual.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtQual'] = 'TA'
		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtCond.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtCond'] = 'TA'
		all_data.loc[(all_data.BsmtFinSF2>0) & (all_data.BsmtFinType2.isnull()) , 'BsmtFinType2'] = 'Unf'
		all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtFinSF2'] = 354.0
		all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtUnfSF'] = 0.0

		nulls_cols = {'BsmtExposure': 'NA', 'BsmtFinType2': 'NA', 'BsmtQual': 'NA', 'BsmtCond': 'NA', 'BsmtFinType1': 'NA',
					  'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0 ,'TotalBsmtSF': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0}

		all_data = all_data.fillna(value=nulls_cols)

		NegMean = all_data.groupby('Neighborhood').LotFrontage.mean()

		all_data.loc.LotFrontage = all_data[['Neighborhood', 'LotFrontage']].apply(lambda x: NegMean[x.Neighborhood] if np.isnan(x.LotFrontage) else x.LotFrontage, axis=1)

		PoolQC = {0: 'NA', 1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}

		all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['PoolQC']] =\
				((all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['OverallQual']]/2).round()).\
				apply(lambda x: x.map(PoolQC))

		all_data.PoolQC = all_data.PoolQC.fillna('NA')

		all_data.Functional = all_data.Functional.fillna('Typ')

		all_data.loc[(all_data.Fireplaces==0) & (all_data.FireplaceQu.isnull()), ['FireplaceQu']] = 'NA'

		all_data.loc[(all_data.KitchenAbvGr>0) & (all_data.KitchenQual.isnull()), 
					 ['KitchenQual']] = all_data.KitchenQual.mode()[0]

		all_data.Alley = all_data.Alley.fillna('NA')
		all_data.Fence = all_data.Fence.fillna('NA')
		all_data.MiscFeature = all_data.MiscFeature.fillna('NA')
		all_data.loc[all_data.GarageYrBlt==2207.0, 'GarageYrBlt'] = 2007.0

		all_data = DataFrameImputer().fit_transform(all_data)

		self.trainingData = all_data.loc[(all_data.SalePrice>0)].reset_index(drop=True, inplace=False)
		self.testingData = all_data.loc[(all_data.SalePrice==0)].reset_index(drop=True, inplace=False)
		self.combinedData = [self.trainingData, self.testingData]
		return DataObject(self.trainingData, self.testingData, self.combinedData)
