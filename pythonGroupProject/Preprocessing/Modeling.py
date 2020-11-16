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

from DataObject import DataObject
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)   



class Modeling:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData
		
		
	def RMSLE (self, y, y_pred):
		return (np.sqrt(mean_squared_error(y, y_pred)))


    def get_results(self, model, name='NAN', log=False):
    
		rcols = ['Name','Model', 'BestParameters', 'Scorer', 'Index', 'BestScore', 'BestScoreStd', 'MeanScore', 
				 'MeanScoreStd', 'Best']
		res = pd.DataFrame(columns=rcols)
		
		results = gs.cv_results_
		modelo = gs.best_estimator_

		scoring = {'MEA': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_mean_squared_error'}

		for scorer in sorted(scoring):
			best_index = np.nonzero(results['rank_test_%s' % scoring[scorer]] == 1)[0][0]
			if scorer == 'RMSE': 
				best = np.sqrt(-results['mean_test_%s' % scoring[scorer]][best_index])
				best_std = np.sqrt(results['std_test_%s' % scoring[scorer]][best_index])
				scormean = np.sqrt(-results['mean_test_%s' % scoring[scorer]].mean())
				stdmean = np.sqrt(results['std_test_%s' % scoring[scorer]].mean())
				if log:
					best = np.expm1(best)
					best_std = np.expm1(best_std)
					scormean = np.expm1(scormean)
					stdmean = np.expm1(stdmean)
			elif scorer == 'MEA':
				best = (-results['mean_test_%s' % scoring[scorer]][best_index])
				best_std = results['std_test_%s' % scoring[scorer]][best_index]
				scormean =(-results['mean_test_%s' % scoring[scorer]].mean())
				stdmean = results['std_test_%s' % scoring[scorer]].mean()
				if log:
					best = np.expm1(best)
					best_std = np.expm1(best_std)
					scormean = np.expm1(scormean)
					stdmean = np.expm1(stdmean)
			else:
				best = results['mean_test_%s' % scoring[scorer]][best_index]*100
				best_std = results['std_test_%s' % scoring[scorer]][best_index]*100
				scormean = results['mean_test_%s' % scoring[scorer]].mean()*100
				stdmean = results['std_test_%s' % scoring[scorer]].mean()*100
			
			r1 = pd.DataFrame([(name, modelo, gs.best_params_, scorer, best_index, best, best_std, scormean, 
								stdmean, gs.best_score_)],
							  columns = rcols)
			res = res.append(r1)
			
		if log:
			bestscore = np.expm1(np.sqrt(-gs.best_score_))
		else:
			bestscore = np.sqrt(-gs.best_score_)
			
		print("Best Score: {:.6f}".format(bestscore))
		print('---------------------------------------')
		print('Best Parameters:')
		print(gs.best_params_)
    
    return res
	
	############# redisual
	def resilduals_plots(self, lr, X, Y, log=False):
		y_pred = lr.predict(X)
		residual = pd.DataFrame()
		residual['Predict'] = y_pred
		residual['Residual'] = Y - y_pred
		residual['Predicted'] = np.expm1(residual.Predict)
		residual['StdResidual'] = np.expm1(Y) - residual.Predicted
		residual.StdResidual = residual.StdResidual / residual.StdResidual.std()
		residual['IDX'] = X.index
		
		if log:
			fig = plt.figure(figsize=(20,10))
			ax = fig.add_subplot(121)
			g = sns.regplot(y='Residual', x='Predict', data = residual, order=1, ax = ax) 
			plt.xlabel('Log Predicted Values')
			plt.ylabel('Log Residuals')
			plt.hlines(y=0, xmin=min(Y)-1, xmax=max(Y)+1, lw=2, color='red')
			plt.xlim([min(Y)-1, max(Y)+1])

			ax = fig.add_subplot(122)
			g = sns.regplot(y='StdResidual', x='Predicted', data = residual, order=1, ax = ax) 
			plt.xlabel('Predicted Values')
			plt.ylabel('Standardized Residuals')
			plt.hlines(y=0, xmin=np.expm1(min(Y))-1, xmax=np.expm1(max(Y))+1, lw=2, color='red')
			plt.xlim([np.expm1(min(Y))-1, np.expm1(max(Y))+1])
		else:
			residual.StdResidual = residual.Residual / residual.Residual.std()
			residual.drop(['Residual', 'Predicted'], axis = 1, inplace=True)
			g = sns.regplot(y='StdResidual', x='Predict', data = residual, order=1) 
			plt.xlabel('Predicted Values')
			plt.ylabel('Standardized Residuals')
			plt.hlines(y=0, xmin=min(Y)-1, xmax=max(Y)+1, lw=2, color='red')
			plt.xlim([min(Y)-1, max(Y)+1])

		plt.show()  

    return residual
	
	def go(self):
        train = self.trainingData
		y_train = self.testingData
		
		#LASSOO
		model = Pipeline([
			('pca', PCA(random_state = 101)),
			('model', Lasso(random_state = 101))]) 

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-5, len(SEL)-3, len(SEL)] 
		whiten = [False, True]
		max_iter = [5] #, 10, 100, 200, 300, 400, 500, 600]  
		alpha = [0.0003, 0.0007, 0.0005, 0.05, 0.5, 1.0]
		selection = ['random', 'cyclic'] 
		tol = [2e-03, 0.003, 0.001, 0.0005]
		param_grid =\
					dict(
						  model__alpha = alpha
						  ,model__max_iter = max_iter
						  ,model__selection = selection
						  ,model__tol = tol
						  ,pca__n_components = n_components
						  ,pca__whiten = whiten 
						) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error' #, iid=False
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)

		lasso = Pipeline([
				('sel', select_fetaures(select_cols=SEL)), 
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		lasso.fit(train,y_train)

		results = self.get_results(lasso, 'lasso Lg1', log=True)
		display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
		r = self.resilduals_plots(lasso, train, y_train, log=True)
			
		fica =  list(r.IDX[abs(r.Residual)<=0.3])
		print('Outliers removed:', r.shape[0]-len(fica))
		t = train.iloc[fica, :].reset_index(drop=True, inplace=False)
		y_t = y_train.iloc[fica].reset_index(drop=True, inplace=False)

		lasso.fit(t, y_t)
		results = self.get_results(lasso, 'lasso Lg2', log=True)
		display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
		r = self.resilduals_plots(lasso, t, y_t, log=True)
		del  t, y_t, fica

		# 2
		y_log = y_train.copy()
		y_train = np.expm1(y_train)

		lasso.fit(train, y_train)

		results = self.get_results(lasso, 'lasso', log=False)
		display(results.loc[:, 'Scorer' : 'MeanScoreStd'])
		r = self.resilduals_plots(lasso, train, y_train, log=False)

		#3
		model = Pipeline([
			('pca', PCA(random_state = 101)),
			('model', XGBRegressor(random_state=101, silent=False))])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [90] # [len(SEL)-18, len(SEL)-19, len(SEL)-20] 
		whiten = [True] #, False]
		n_est = [3500] # [500, 750, 1000, 2000, 2006] # np.arange(1997, 2009, 3) # 
		max_depth = [3] #, 4]
		learning_rate = [0.01] #, 0.03] #, 0.1, 0.05
		reg_lambda = [1] #0.1, 1e-06, 1e-04, 1e-03, 1e-02, 1e-05, 1, 0.0] 
		reg_alpha= [1] # , 0.5, 1, 0.0]
		booster = ['gblinear'] #'dart', 'gbtree']  
		objective = ['reg:tweedie'] #, 'reg:linear', 'reg:gamma']

		param_grid =\
					dict(
						  pca__n_components = n_components,
						  pca__whiten = whiten, 
						  model__n_estimators= n_est
						  ,model__booster = booster
						  ,model__objective = objective
						  ,model__learning_rate = learning_rate
						  ,model__reg_lambda = reg_lambda
						  ,model__reg_alpha = reg_alpha
						  ,model__max_depth = max_depth
						) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		XGBR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		XGBR.fit(train, y_train)

		res = self.get_results(XGBR, 'XGBRegressor', log=False)
		self.resilduals_plots(XGBR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#4

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', GradientBoostingRegressor(random_state=101))])

		SEL = list(set(XGBestCols).union(set(colsP)))
		# n_components = [len(SEL)] 
		whiten = [True] #, False]
		n_est = [3000]
		learning_rate = [0.05] #, 0.01, 0.1, 0.005]
		loss = ['huber'] #, 'ls', 'lad', 'quantile']
		max_features = ['auto'] #, 'sqrt', 'log2']
		max_depth = [3] #, 2] # , 5]
		min_samples_split = [3] #, 4] 
		min_samples_leaf = [3] # , 3, 2 ,4 ]
		criterion = ['friedman_mse'] #, 'mse', 'mae']
		alpha = [0.8] #, 0.75, 0.9, 0.7] 

		param_grid =\
					dict(
						  #pca__n_components = n_components,
						  pca__whiten = whiten, 
						   model__n_estimators= n_est 
						  ,model__learning_rate = learning_rate
						  ,model__loss = loss
						  ,model__criterion = criterion
						  ,model__max_depth = max_depth
						  ,model__alpha = alpha
						  ,model__max_features = max_features
						  ,model__min_samples_split = min_samples_split
						  ,model__min_samples_leaf = min_samples_leaf
						   )

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						  , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						  ,cv=5, verbose=1, n_jobs=4)
		 
		GBR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		GBR.fit(train, y_train)
		res = self.get_results(GBR, 'GBR' , log=False)
		self.resilduals_plots(GBR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#5

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', ElasticNet(random_state=101))])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-5, len(SEL)-3, len(SEL)] 
		whiten = [False] #, True]
		max_iter = [5] #, 100] 
		alpha = [1e-05] #, 0.001, 0.01, 0.003, 0.00001] 
		l1_ratio =  [0.00003] 
		selection = ['cyclic'] #, 'random', 'cyclic']

		param_grid =\
					dict(
						  model__max_iter= max_iter
						  ,pca__n_components = n_components
						  ,pca__whiten = whiten 
						  ,model__alpha = alpha
						  ,model__l1_ratio = l1_ratio
						  ,model__selection = selection
					   ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		ELA = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		ELA.fit(train, y_train)

		res = self.get_results(ELA, 'ELA', log=False)
		self.resilduals_plots(ELA, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']


		#bayseian ridge

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', BayesianRidge())]) #compute_score=False, fit_intercept=True, normalize=False

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-9] #, len(SEL)-8, len(SEL)-7] 
		whiten = [True] # , False]
		n_iter=  [36] # np.arange(36, 45) # [40, 35, 45, 70, 100, 200, 300, 500, 700, 1000] #  
		alpha_1 = [1e-06] #0.1, 1e-04, 1e-03, 1e-02, 1e-05]
		alpha_2 = [0.1] # 1e-06 , , 1e-02, 1e-04, 1e-03]
		lambda_1 = [0.001] # 0.1, 1e-06, 1e-04, 1e-02, 1e-05] 
		lambda_2 = [0.01] # 0.1, 1e-06, 1e-04, 1e-03, 1e-05]

		param_grid =\
					dict(
						   model__n_iter = n_iter
						  ,model__alpha_1 = alpha_1
						  ,model__alpha_2 = alpha_2
						  ,model__lambda_1 = lambda_1
						  ,model__lambda_2 = lambda_2
						  ,pca__n_components = n_components
						  ,pca__whiten = whiten 
					  ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		BayR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		BayR.fit(train, y_train)
		res = self.get_results(BayR, 'BayR', log=False)
		self.resilduals_plots(BayR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#linear regression

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', LinearRegression())])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-10, len(SEL)-11, len(SEL)-9] 
		whiten = [True, False]

		param_grid =\
					dict(
						   pca__n_components = n_components,
						   pca__whiten = whiten
					   ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		LR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		LR.fit(train, y_train)

		res = self.get_results(LR, 'LR', log=False)
		self.resilduals_plots(LR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#Orthogonal Matching Pursuit model (OMP)

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', OrthogonalMatchingPursuit())])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [100] # [len(SEL)-11, len(SEL)-10, len(SEL)-9] 
		whiten = [False]
		tol = [5e-05] # [None, 0.00005, 0.0001, 0.00000, 0.002]

		param_grid =\
					dict(
						   model__tol = tol
						   ,model__n_nonzero_coefs = [2] # range(2, 6) # [10, 20, 30, 40, 50, 60, 70, 80, 90, None] # 
						   ,pca__n_components = n_components
						   ,pca__whiten = whiten
						   ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		ORT = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		ORT.fit(train, y_train)
		res = self.get_results(ORT, 'ORT', log=False)
		self.resilduals_plots(ORT, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#Robust Regressor
		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', HuberRegressor())])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-9] #, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
		whiten = [True] #, False]
		max_iter = [2000] 
		alpha = [0.0001] #, 5e-05, 0.01, 0.00005, 0.0005, 0.5, 0.001] 
		epsilon = [1.005] #, 1.05, 1.01, 1.001] 
		tol = [1e-01, 1e-02] #, 2e-01, 3e-01, 4e-01, 5e-01, 6e-01] 

		param_grid =\
					dict(
						  model__max_iter= max_iter
						  ,pca__n_components = n_components
						  ,pca__whiten = whiten 
						  ,model__alpha = alpha
						  ,model__epsilon = epsilon
						  ,model__tol = tol
					   ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=3)
		 
		Hub = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		Hub.fit(train, y_train)

		res = self.get_results(Hub, 'Hub', log=False)
		self.resilduals_plots(Hub, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#Passive Aggressive Regressor

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', PassiveAggressiveRegressor(random_state = 101))])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-9, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
		whiten = [True] #, False]
		loss = ['squared_epsilon_insensitive'] #, 'epsilon_insensitive']
		C = [0.001] #, 0.005, 0.003]
		max_iter = [1000] 
		epsilon = [0.00001] # , 0.00005
		tol = [1e-03] #, 1e-05,1e-02, 1e-01, 1e-04, 1e-06]

		param_grid =\
					dict(
						  pca__n_components = n_components,
						  pca__whiten = whiten, 
						  model__loss = loss
						  ,model__epsilon = epsilon
						  ,model__C = C
						  ,model__tol = tol
						  ,model__max_iter = max_iter
					   ) 
		 
		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		PassR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		PassR.fit(train, y_train)
		res = self.get_results(PassR, 'PassR', log=False)
		self.resilduals_plots(PassR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']

		#SGD Regressor

		model = Pipeline([
				('pca', PCA(random_state = 101)),
				('model', SGDRegressor(random_state = 101))])

		SEL = list(set(RFEcv).union(set(colsP)))
		n_components = [len(SEL)-9, len(SEL)-8, len(SEL)-7, len(SEL)-1] 
		whiten = [True] #, False]
		loss = ['squared_loss'] #, 'huber', 'squared_epsilon_insensitive', 'epsilon_insensitive']
		penalty = ['l2'] #, 'elasticnet', 'l1']
		l1_ratio = [0.7] #, 0.8] #[0.2, 0.5, 0.03]
		learning_rate = ['invscaling'] #, 'constant', 'optimal']
		alpha = [0.001] # [1e-01, 1e-2, 1e-03, 1e-4, 1e-05]
		epsilon =  [1e-01] #, 1e-2, 1e-03, 1e-4, 1e-05]
		tol = [0.001] #, 0.003] 
		eta0 = [0.01] #, 1e-1, 1e-03, 1e-4, 1e-05] 
		power_t = [0.5]
		 
		param_grid =\
					dict(
						   pca__n_components = n_components
						   ,pca__whiten = whiten, 
						   model__penalty = penalty
						   ,model__l1_ratio = l1_ratio
						   ,model__loss = loss
						   ,model__alpha = alpha
						   ,model__epsilon = epsilon
						   ,model__tol = tol
						   ,model__eta0 = eta0
						   ,model__power_t = power_t
						   ,model__learning_rate = learning_rate
					   ) 

		gs = GridSearchCV(estimator = model, param_grid = param_grid, refit = 'neg_mean_squared_error'
						   , scoring=list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2']) 
						   ,cv=5, verbose=1, n_jobs=4)
		 
		SGDR = Pipeline([
				('sel', select_fetaures(select_cols=SEL)),
				('scl', RobustScaler()),
				('gs', gs)
		 ])

		SGDR.fit(train, y_train)
		res = self.get_results(SGDR, 'SGDR', log=False)
		self.resilduals_plots(SGDR, train, y_train, log=False)
		results = pd.concat([results, res], axis=0)
		res.loc[:, 'Scorer' : 'MeanScoreStd']
		
		averaged_models = AveragingModels(models = (XGBR, BayR, PassR))
		averaged_models.fit(train, y_train) 
		stacked_train_pred = averaged_models.predict(train)
		rmsle = self.RMSLE(y_train,stacked_train_pred)
		
		print('RMSLE score on the train data: {:.4f}'.format(rmsle))
		print('Accuracy score: {:.6%}'.format(averaged_models.score(train, y_train)))
		
		ensemble = stacked_pred *1
		submit = pd.DataFrame()
		submit['id'] = test_ID
		submit['SalePrice'] = ensemble
		
		return(submit)
	

		
