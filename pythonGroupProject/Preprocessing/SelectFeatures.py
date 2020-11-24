import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.base import clone
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
from xgboost import XGBRegressor

from Preprocessing.DataObject import DataObject

class SequentialFeatureSelection():
	def __init__(self, estimator, k_features, scoring=r2_score, test_size=0.25, random_state=101):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = list(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score]
		
		while dim > self.k_features:
			scores = []
			subsets = []
			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, list(p))
				scores.append(score)
				subsets.append(list(p))
				
			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])
			
		self.k_score_ = self.scores_[-1]
		return self

	def transform(self, X):
		return X.iloc[:, self.indices_]
	
	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train.iloc[:, indices], y_train)
		y_pred = self.estimator.predict(X_test.iloc[:, indices])
		score = self.scoring(y_test, y_pred)
		return score

class SelectFeatures:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self, all_data, cols, colsP):
		train = all_data.loc[(all_data.SalePrice>0), cols].reset_index(drop=True, inplace=False)
		y_train = all_data.SalePrice[all_data.SalePrice>0].reset_index(drop=True, inplace=False)
		test = all_data.loc[(all_data.SalePrice==0), cols].reset_index(drop=True, inplace=False)
		# Main script here
		scale = RobustScaler();
		df = pd.DataFrame(scale.fit_transform(train[cols]), columns= cols)
	   #select features based on P values
		ln_model=sm.OLS(y_train,df)
		result=ln_model.fit()
		print(result.summary2())

		pv_cols = cols.values
		SL = 0.051
		pv_cols, LR = self.backwardElimination(df, y_train, SL, pv_cols)

		pred = LR.predict(df[pv_cols])

		y_pred = pred.apply(lambda x: 1 if x > 0.5 else 0)

		print('Fvalue: {:.6f}'.format(LR.fvalue))
		print('MSE total on the train data: {:.4f}'.format(LR.mse_total))

		ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
		rfecv = RFECV(estimator=ls, n_jobs = -1, step=1, scoring = 'neg_mean_squared_error' ,cv=5)
		rfecv.fit(df, y_train)

		select_features_rfecv = rfecv.get_support()
		RFEcv = cols[select_features_rfecv]
		print('{:d} Features Select by RFEcv:\n{:}'.format(rfecv.n_features_, RFEcv.values))

		score = r2_score
		ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
		sbs = SequentialFeatureSelection(ls, k_features=1, scoring= score)
		sbs.fit(df, y_train)

		print('Best Score: {:2.2%}\n'.format(max(sbs.scores_)))
		print('Best score with:{0:2d}.\n'.\
			  format(len(list(df.columns[sbs.subsets_[np.argmax(sbs.scores_)]]))))
		SBS = list(df.columns[list(sbs.subsets_[max(np.arange(0, len(sbs.scores_))[(sbs.scores_==max(sbs.scores_))])])])
		print('\nBest score with {0:2d} features:\n{1:}'.format(len(SBS), SBS))

		skb = SelectKBest(score_func=f_regression, k=80)
		skb.fit(df, y_train)
		select_features_kbest = skb.get_support()
		kbest_FR = cols[select_features_kbest]
		scores = skb.scores_[select_features_kbest]

		skb = SelectKBest(score_func=mutual_info_regression, k=80)
		skb.fit(df, y_train)
		select_features_kbest = skb.get_support()
		kbest_MIR = cols[select_features_kbest]
		scores = skb.scores_[select_features_kbest]

		X_train, X_test, y, y_test = train_test_split(df, y_train, test_size=0.30, random_state=101)

		# fit model on all training data
		#importance_type='gain'
		model =  XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, max_delta_step=0, 
							  random_state=101, min_child_weight=1, missing=None, n_jobs=4,  
							  scale_pos_weight=1, seed=None, silent=True, subsample=1)


		model.fit(X_train, y)

		# Using each unique importance as a threshold
		thresholds = np.sort(np.unique(model.feature_importances_)) 
		best = 1e36
		colsbest = 31
		my_model = model
		threshold = 0

		for thresh in thresholds:
			# select features using threshold
			selection = SelectFromModel(model, threshold=thresh, prefit=True)
			select_X_train = selection.transform(X_train)
			# train model
			selection_model =  XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, max_delta_step=0, 
											random_state=101, min_child_weight=1, missing=None, n_jobs=4, 
											scale_pos_weight=1, seed=None, silent=True, subsample=1)
			selection_model.fit(select_X_train, y)
			# eval model
			select_X_test = selection.transform(X_test)
			y_pred = selection_model.predict(select_X_test)
			predictions = [round(value) for value in y_pred]
			r2 = r2_score(y_test, predictions)
			mse = mean_squared_error(y_test, predictions)
			print("Thresh={:1.3f}, n={:d}, R2: {:2.2%} with MSE: {:.4f}".format(thresh, select_X_train.shape[1], r2, mse))
			if (best >= mse):
				best = mse
				colsbest = select_X_train.shape[1]
				my_model = selection_model
				threshold = thresh

		feature_importances = [(score, feature) for score, feature in zip(model.feature_importances_, cols)]
		XGBest = pd.DataFrame(sorted(sorted(feature_importances, reverse=True)[:colsbest]), columns=['Score', 'Feature'])
		XGBestCols = XGBest.iloc[:, 1].tolist()

		bcols = set(pv_cols).union(set(RFEcv)).union(set(kbest_FR)).union(set(kbest_MIR)).union(set(XGBestCols)).union(set(SBS))
		intersection = set(SBS).intersection(set(kbest_MIR)).intersection(set(RFEcv)).intersection(set(pv_cols)).intersection(set(kbest_FR)).intersection(set(XGBestCols))
		print(intersection, '\n')
		print('_'*75,'\nUnion All Features Selected:')
		print('Total number of features selected:', len(bcols))
		print('\n{0:2d} features removed if use the union of selections: {1:}'.format(len(cols.difference(bcols)), cols.difference(bcols)))

		totalCols = list(bcols.union(set(colsP)))
		#self.trainingData = self.trainingData.loc[list(totalCols)].reset_index(drop=True, inplace=False)
		#self.testingData = self.testingData.loc[list(totalCols)].reset_index(drop=True, inplace=False)
		#self.combinedData = [self.trainingData, self.testingData]

		return DataObject(self.trainingData, self.testingData, self.combinedData), totalCols, RFEcv, XGBestCols

	def backwardElimination(self, x, Y, sl, columns):
		ini = len(columns)
		numVars = x.shape[1]
		for i in range(0, numVars):
			regressor = sm.OLS(Y, x).fit()
			maxVar = max(regressor.pvalues) #.astype(float)
			if maxVar > sl:
				for j in range(0, numVars - i):
					if (regressor.pvalues[j].astype(float) == maxVar):
						columns = np.delete(columns, j)
						x = x.loc[:, columns]
					
		print('\nSelect {:d} features from {:d} by best p-values.'.format(len(columns), ini))
		print('The max p-value from the features selecte is {:.3f}.'.format(maxVar))
		print(regressor.summary())

		# odds ratios and 95% CI
		conf = np.exp(regressor.conf_int())
		conf['Odds Ratios'] = np.exp(regressor.params)
		conf.columns = ['2.5%', '97.5%', 'Odds Ratios']
		#display(conf)

		return columns, regressor

# pv_cols = cols.values
# SL = 0.051
# pv_cols, LR = backwardElimination(df, y_train, SL, pv_cols)

#next method recursive method for feature selection 
#ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
#rfecv = RFECV(estimator=ls, n_jobs = -1, step=1, scoring = 'neg_mean_squared_error' ,cv=5) rfecv.fit(df, y_train)

#	select_features_rfecv = rfecv.get_support()
#	RFEcv = cols[select_features_rfecv]
#	print('{:d} Features Select by RFEcv:\n{:}'.format(rfecv.n_features_, RFEcv.values))
	
