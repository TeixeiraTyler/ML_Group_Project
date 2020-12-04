import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import tree
from patsy import dmatrices
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, norm, probplot, boxcox

from Preprocessing.DataObject import DataObject


class FeatureEngineer:
	def __init__(self, combinedData, trainingData):
		self.combinedData = combinedData
		self.trainingData = trainingData

	def go(self):
		self.combinedData = self.changeYearsToAge(self.combinedData)

		self.combinedData = self.addRemodAndConvertAge(self.combinedData)

		self.combinedData = self.defineUint8Types(self.combinedData)

		ntrain = self.trainingData.shape[0]
		self.combinedData, y_train, cols, colsP = self.featureEngineer(self.combinedData, ntrain)

		return self.combinedData, self.combinedData, y_train, cols, colsP

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

		dataset["WasNew"] = -1
		dataset.loc[(dataset.YearBuilt == dataset.YrSold), ['WasNew']] = 1
		dataset.loc[(dataset.YearBuilt != dataset.YrSold), ['WasNew']] = 0

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
		dataset.WasNew = dataset.WasNew.astype('uint8')
		dataset.Street = dataset.Street.astype('uint8')
		dataset.PavedDrive = dataset.PavedDrive.astype('uint8')
		dataset.Functional = dataset.Functional.astype('uint8')
		dataset.LandSlope = dataset.LandSlope.astype('uint8')

		return dataset

	def featureEngineer(self, data, ntrain):
		data.loc[(data.PoolArea > 0), ['MiscFeature']] = 'Pool'
		data.loc[(data.PoolArea > 0), ['MiscVal']] = data.loc[(data.PoolArea > 0), ['MiscVal', 'PoolArea']].apply(lambda x: (x.MiscVal + x.PoolArea), axis=1)

		data['TotalExtraPoints'] = data.HeatingQC + data.PoolQC + data.FireplaceQu + data.KitchenQual
		data['TotalPoints'] = (data.ExterQual + data.FireplaceQu + data.GarageQual + data.KitchenQual +
							   data.BsmtQual + data.BsmtExposure + data.BsmtFinType1 + data.PoolQC + data.ExterCond +
							   data.BsmtCond + data.GarageCond + data.OverallCond + data.BsmtFinType2 + data.HeatingQC) + data.OverallQual ** 2

		df = data.loc[(data.SalePrice > 0), ['TotalPoints', 'TotalExtraPoints', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'PoolQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'SalePrice']]

		data['GarageArea_x_Car'] = data.GarageArea * data.GarageCars

		data['TotalBsmtSF_x_Bsm'] = data.TotalBsmtSF * data['1stFlrSF']

		# We don´t have a feature with all construct area, maybe it is an interesting feature to create.
		data['ConstructArea'] = (data.TotalBsmtSF + data.WoodDeckSF + data.GrLivArea + data.OpenPorchSF +
								 data.TSsnPorch + data.ScreenPorch + data.EnclosedPorch + data.MasVnrArea + data.GarageArea + data.PoolArea)

		# all_data['TotalArea'] = all_data.ConstructArea + all_data.LotArea

		data['Garage_Newest'] = data.YearBuilt > data.GarageYrBlt
		data.Garage_Newest = data.Garage_Newest.apply(lambda x: 1 if x else 0)

		data[
			'TotalPorchSF'] = data.OpenPorchSF + data.EnclosedPorch + data.TSsnPorch + data.ScreenPorch + data.WoodDeckSF
		data.EnclosedPorch = data.EnclosedPorch.apply(lambda x: 1 if x else 0)

		data['LotAreaMultSlope'] = data.LotArea * data.LandSlope

		data['BsmtSFPoints'] = (data.BsmtQual ** 2 + data.BsmtCond + data.BsmtExposure +
								data.BsmtFinType1 + data.BsmtFinType2)

		data['BsmtSFMultPoints'] = data.TotalBsmtSF * (
				data.BsmtQual ** 2 + data.BsmtCond + data.BsmtExposure +
				data.BsmtFinType1 + data.BsmtFinType2)

		data['TotBathrooms'] = data.FullBath + (data.HalfBath * 0.5) + data.BsmtFullBath + (
				data.BsmtHalfBath * 0.5)
		data.FullBath = data.FullBath.apply(lambda x: 1 if x else 0)
		data.HalfBath = data.HalfBath.apply(lambda x: 1 if x else 0)
		data.BsmtFullBath = data.BsmtFullBath.apply(lambda x: 1 if x else 0)
		data.BsmtHalfBath = data.BsmtHalfBath.apply(lambda x: 1 if x else 0)



		data.MSSubClass = data.MSSubClass.astype('str')
		data.MoSold = data.MoSold.astype('str')

		data, dummies = self.one_hot_encode(data)

		ZeroTest = data[dummies][ntrain:].sum() == 0
		data.drop(dummies[ZeroTest], axis=1, inplace=True)
		print('Dummins in test dataset with all observatios equal to 0:', len(dummies[ZeroTest]), 'of \n',
			  dummies[ZeroTest], '\n')
		dummies = dummies.drop(dummies[ZeroTest])

		# Find dummies with all training observatiosn are equal to 0
		ZeroTest = data[dummies][:ntrain].sum() == 0
		data.drop(dummies[ZeroTest], axis=1, inplace=True)
		print('Dummins in trainig dataset with all observatios equal to 0:', len(dummies[ZeroTest]), 'of \n',
			  dummies[ZeroTest], '\n')
		dummies = dummies.drop(dummies[ZeroTest])

		del ZeroTest



		data['Remod'] = 2
		data.loc[(data.YearBuilt == data.YearRemodAdd), ['Remod']] = 0
		data.loc[(data.YearBuilt != data.YearRemodAdd), ['Remod']] = 1

		#all_data['Age'] = all_data.YearRemodAdd - all_data.YrSold  # sice I convert both to age

		data["WasNew"] = 2
		data.loc[(data.YearBuilt == data.YrSold), ['WasNew']] = 1
		data.loc[(data.YearBuilt != data.YrSold), ['WasNew']] = 0

		data.drop(
			['FireplaceQu', 'BsmtSFPoints', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OverallQual', 'GrLivArea',
			 'TotalBsmtSF_x_Bsm', '1stFlrSF', 'PoolArea', 'LotArea', 'SaleCondition_Partial', 'Exterior1st_VinylSd',
			 'GarageCond', 'HouseStyle_2Story', 'BsmtSFMultPoints', 'ScreenPorch', 'LowQualFinSF', 'BsmtFinSF2',
			 'TSsnPorch'], axis=1, inplace=True)

		data.rename(columns={'2ndFlrSF': 'SndFlrSF'}, inplace=True)



		# Remove the higest correlations and run a multiple regression
		cols = data.columns
		print(cols)
		cols = cols.drop(['SalePrice'])
		#vif = self.VRF('SalePrice', all_data.loc[all_data.SalePrice > 0, cols], all_data.SalePrice[all_data.SalePrice > 0], cols)

		cols = cols.drop(
			['Condition1_PosN', 'Neighborhood_NWAmes', 'Exterior1st_CBlock', 'BldgType_1Fam', 'RoofStyle_Flat',
			 'MSZoning_Call', 'Alley_Grvl', 'LandContour_Bnk', 'LotConfig_Corner', 'GarageType_2Types', 'MSSubClass_45',
			 'MasVnrType_BrkCmn', 'Foundation_CBlock', 'MiscFeature_Gar2', 'SaleType_COD', 'Exterior2nd_CBlock'])

		#vif = self.VRF('SalePrice', all_data.loc[all_data.SalePrice > 0, cols], all_data.SalePrice[all_data.SalePrice > 0], cols)

		cols = cols.drop(
			['PoolQC', 'BldgType_TwnhsE', 'BsmtFinSF1', 'BsmtUnfSF', 'Electrical_SBrkr', 'Exterior1st_MetalSd',
			 'Exterior2nd_VinylSd', 'GarageQual', 'GarageType_Attchd', 'HouseStyle_1Story', 'MasVnrType_None',
			 'MiscFeature_NA', 'MSZoning_RL', 'RoofStyle_Gable', 'SaleCondition_Normal', 'MoSold_10',
			 'SaleType_New', 'SndFlrSF', 'TotalPorchSF', 'WoodDeckSF', 'BldgType_Duplex', 'MSSubClass_90'])

		print(cols)
		#print(vif)

		df_copy = data[data.SalePrice > 0].copy()

		data.CentralAir = data.CentralAir.astype('uint8')
		data.Garage_Newest = data.Garage_Newest.astype('uint8')
		data.EnclosedPorch = data.EnclosedPorch.astype('uint8')
		data.FullBath = data.FullBath.astype('uint8')
		data.HalfBath = data.HalfBath.astype('uint8')
		data.BsmtFullBath = data.BsmtFullBath.astype('uint8')
		data.BsmtHalfBath = data.BsmtHalfBath.astype('uint8')
		data.Remod = data.Remod.astype('uint8')
		data.WasNew = data.WasNew.astype('uint8')
		data.Street = data.Street.astype('uint8')  # orinal
		data.PavedDrive = data.PavedDrive.astype('uint8')  # ordinal
		data.Functional = data.Functional.astype('uint8')  # ordinal
		data.LandSlope = data.LandSlope.astype('uint8')  # ordinal

		numeric_features = list(data.loc[:, cols].dtypes[(data.dtypes != "category") & (data.dtypes != 'uint8')].index)

		'''
		with warnings.catch_warnings():
		    warnings.simplefilter("ignore", category=RuntimeWarning)
		'''
		skewed_features = data[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

		#compute skewness
		skewness = pd.DataFrame({'Skew' :skewed_features})   

		# Get only higest skewed features
		skewness = skewness[abs(skewness) > 0.7]
		skewness = skewness.dropna()

		l_opt = {}

		for feat in skewness.index:
			data[feat], l_opt[feat] = boxcox((data[feat] + 1))

		skewed_features2 = data[skewness.index].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

		#compute skewness
		skewness2 = pd.DataFrame({'New Skew' :skewed_features2}) 




		y = data.SalePrice[data.SalePrice > 0]
		X = data.loc[data.SalePrice > 0, ['ConstructArea']]
		#self.poly(X, y, 'ConstructArea')

		X = data.loc[data.SalePrice > 0, ['ConstructArea', 'TotalPoints']]
		#self.poly(X, y)

		X = data.loc[
			data.SalePrice > 0, ['ConstructArea', 'TotalPoints', 'LotAreaMultSlope', 'GarageArea_x_Car']]
		#self.poly(X, y)

		poly_cols = ['ConstructArea', 'TotalPoints', 'LotAreaMultSlope', 'GarageArea_x_Car']

		pf = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
		res = pf.fit_transform(data.loc[:, poly_cols])

		target_feature_names = [feat.replace(' ', '_') for feat in pf.get_feature_names(poly_cols)]
		output_df = pd.DataFrame(res, columns=target_feature_names, index=data.index).iloc[:, len(poly_cols):]
		print('Polynomial Features included:', output_df.shape[1])
		# display(output_df.head())
		data = pd.concat([data, output_df], axis=1)
		print('Total Features after Polynomial Features included:', data.shape[1])
		colsP = output_df.columns

		del output_df, target_feature_names, res, pf

		y_train = (data.SalePrice[data.SalePrice > 0].reset_index(drop=True, inplace=False))
		#self.combinedData = all_data.loc[(all_data.SalePrice>0), cols].reset_index(drop=True, inplace=False)
		#self.testingData = all_data.loc[(all_data.SalePrice==0), cols].reset_index(drop=True, inplace=False)

		return data, y_train, cols, colsP

	def one_hot_encode(self, df):
		categorical_cols = df.select_dtypes(include=['object']).columns

		# print(len(categorical_cols), "categorical columns")
		# print(categorical_cols)
		# Remove special characters and white spaces.
		for c in categorical_cols:
			df[c] = df[c].str.replace('\W', '').str.replace(' ', '_')

		dummies = pd.get_dummies(df[categorical_cols], columns=categorical_cols).columns
		df = pd.get_dummies(df, columns=categorical_cols)

		# print("Total Columns:", len(df.columns))
		# print(df.info())

		return df, dummies

	# Unused currently
	def VRF(self, predict, data, y, cols):
		scale = StandardScaler(with_std=False)
		df = pd.DataFrame(scale.fit_transform(data), columns=cols)
		features = "+".join(cols)
		df['SalePrice'] = y.values

		# get y and X dataframes based on this regression:
		y, X = dmatrices(predict + ' ~' + features, data=df, return_type='dataframe')

		# Calculate VIF Factors
		# For each X, calculate VIF and save in dataframe
		vif = pd.DataFrame()
		vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
		vif["features"] = X.columns

		# Inspect VIF Factors
		# display(vif.sort_values('VIF Factor'))
		return vif

	# Unused currently
	def poly(self, X, y, feat=''):

		# Initializatin of regression models
		regr = LinearRegression()
		regr = regr.fit(X, y)
		y_lin_fit = regr.predict(X)
		linear_r2 = r2_score(y, regr.predict(X))

		# create polynomial features
		quadratic = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
		cubic = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
		fourth = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)
		fifth = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
		X_quad = quadratic.fit_transform(X)
		X_cubic = cubic.fit_transform(X)
		X_fourth = fourth.fit_transform(X)
		X_fifth = fifth.fit_transform(X)

		# quadratic fit
		regr = regr.fit(X_quad, y)
		y_quad_fit = regr.predict(quadratic.fit_transform(X))
		quadratic_r2 = r2_score(y, y_quad_fit)

		# cubic fit
		regr = regr.fit(X_cubic, y)
		y_cubic_fit = regr.predict(cubic.fit_transform(X))
		cubic_r2 = r2_score(y, y_cubic_fit)

		# Fourth fit
		regr = regr.fit(X_fourth, y)
		y_fourth_fit = regr.predict(fourth.fit_transform(X))
		four_r2 = r2_score(y, y_fourth_fit)

		# Fifth fit
		regr = regr.fit(X_fifth, y)
		y_fifth_fit = regr.predict(fifth.fit_transform(X))
		five_r2 = r2_score(y, y_fifth_fit)

		if len(feat) > 0:
			fig = plt.figure(figsize=(20, 5))
			# Plot lowest Polynomials
			fig1 = fig.add_subplot(121)
			plt.scatter(X[feat], y, label='training points', color='lightgray')
			plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=0.5,
					 linestyle=':')
			plt.plot(X[feat], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2, color='red', lw=0.5,
					 linestyle='-')
			plt.plot(X[feat], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2, color='green', lw=0.5,
					 linestyle='--')

			plt.xlabel(feat)
			plt.ylabel('Sale Price')
			plt.legend(loc='upper left')

			# Plot higest Polynomials
			fig2 = fig.add_subplot(122)
			plt.scatter(X[feat], y, label='training points', color='lightgray')
			plt.plot(X[feat], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2, color='blue', lw=2,
					 linestyle=':')
			plt.plot(X[feat], y_fifth_fit, label='Fifth (d=5), $R^2=%.3f$' % five_r2, color='yellow', lw=2,
					 linestyle='-')
			plt.plot(X[feat], y_fifth_fit, label='Fourth (d=4), $R^2=%.3f$' % four_r2, color='red', lw=2, linestyle=':')

			plt.xlabel(feat)
			plt.ylabel('Sale Price')
			plt.legend(loc='upper left')
		else:
			# Plot initialisation
			fig = plt.figure(figsize=(20, 10))
			ax = fig.add_subplot(121, projection='3d')
			ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, s=40)

			# make lines of the regressors:
			plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_lin_fit, label='linear (d=1), $R^2=%.3f$' % linear_r2,
					 color='blue', lw=2, linestyle=':')
			plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_quad_fit, label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2,
					 color='red', lw=0.5, linestyle='-')
			plt.plot(X.iloc[:, 0], X.iloc[:, 1], y_cubic_fit, label='cubic (d=3), $R^2=%.3f$' % cubic_r2,
					 color='green', lw=0.5, linestyle='--')
			# label the axes
			ax.set_xlabel(X.columns[0])
			ax.set_ylabel(X.columns[1])
			ax.set_zlabel('Sales Price')
			ax.set_title("Poly up to 3 degree")
			plt.legend(loc='upper left')

		# plt.tight_layout()
		# plt.show()
