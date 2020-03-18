# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:21:34 2020

@author: zcapjdb
"""

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")



import pandas as pd
import numpy as np

from bdtPlotting import *
from sensitivity import * # sensitivity.py file, which has "calc_sensitivity_with_error" Function in it
from sklearn.ensemble import RandomForestClassifier


from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)


# Reading Data
#if nJets == 2:
 #   dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
  #  dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

#else:
 #   dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
  #  dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')


variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv').head(62858)
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv').head(62858)


 
hyperparameters = {
                    'max_depth': hp.quniform("max_depth", 1, 15, 1),
                    'n_estimators': hp.randint('n_estimators', 300),
                    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01,0.5),
                    'max_features': hp.uniform('max_features',0.1,1)
                    }
   

 
def objective_function(hyperparameters):
    modelEven = RandomForestClassifier(n_estimators = int(hyperparameters['n_estimators']), max_depth = int(hyperparameters['max_depth']), 
                          min_samples_leaf = hyperparameters['min_samples_leaf'], max_features = hyperparameters['max_features'])
    
    modelOdd = RandomForestClassifier(n_estimators = int(hyperparameters['n_estimators']), max_depth = int(hyperparameters['max_depth']), 
                          min_samples_leaf = hyperparameters['min_samples_leaf'], max_features = hyperparameters['max_features'])
    
    
    modelEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
    modelOdd.fit(dfOdd[variables], dfEven['Class'], sample_weight=dfOdd['training_weight'])
    
    scoresEven = modelOdd.predict_proba(dfEven[variables])[:,1]
    scoresOdd = modelEven.predict_proba(dfOdd[variables])[:,1]

    dfEven['decision_value'] = ((scoresEven-0.5)*2)
    dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
    df = pd.concat([dfEven,dfOdd])
    
    sensitivity = calc_sensitivity_with_error(df)
    
    return {'loss': -1*sensitivity[0], 'status': STATUS_OK}

    
trials = Trials()

best = fmin(
    fn=objective_function,
    space=hyperparameters,
    algo=tpe.suggest, # This is the optimization algorithm hyperopt uses, a tree of parzen estimators
    max_evals=50 # The number of iterations
)

print(best)