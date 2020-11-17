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


class SelectFeatures:
    def __init__(self, dataObject):
        self.trainingData = dataObject.trainingData
        self.testingData = dataObject.testingData
        self.combinedData = dataObject.combinedData

    def go(self):
        # Main script here
	    scale = RobustScaler();
        df = pd.DataFrame(scale.fit_transform(self.trainingData[cols]), columns= cols)
	   #select features based on P values
	    ln_model=sm.OLS(y_train,df)
        result=ln_model.fit()
        print(result.summary2())
	 def backwardElimination(x, Y, sl, columns):
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
     display(conf)
    
     return columns, regressor

# pv_cols = cols.values
# SL = 0.051
# pv_cols, LR = backwardElimination(df, y_train, SL, pv_cols)

#next method recursive method for feature selection 
ls = Lasso(alpha = 0.0005, max_iter = 161, selection = 'cyclic', tol = 0.002, random_state = 101)
rfecv = RFECV(estimator=ls, n_jobs = -1, step=1, scoring = 'neg_mean_squared_error' ,cv=5) rfecv.fit(df, y_train)

    select_features_rfecv = rfecv.get_support()
    RFEcv = cols[select_features_rfecv]
    print('{:d} Features Select by RFEcv:\n{:}'.format(rfecv.n_features_, RFEcv.values))
	
