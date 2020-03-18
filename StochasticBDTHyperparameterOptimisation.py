# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:46:44 2020

@author: zcapjdb
"""

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")


import pandas as pd
import numpy as np

from bdtPlotting import *
from sensitivity import * # sensitivity.py file, which has "calc_sensitivity_with_error" Function in it
from xgboost import XGBClassifier


from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, rand


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
#variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv').head(62858)
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv').head(62858)
#dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv').head(62858)
#dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv').head(62858)



 # Dictionary of hyperparameters of BDT and possible range of values
hyperparameters = {'learning_rate': hp.loguniform('learning_rate',  np.log(0.01), np.log(0.5)),
                    'max_depth': hp.quniform("max_depth", 8, 10, 1),
                    'n_estimators': hp.randint('n_estimators', 100),
                    'subsample': hp.uniform('subsample', 0.1, 1.0)
                    }
   

# Objective function to be minimised 
def objective_function(hyperparameters):
    modelEven = XGBClassifier(n_estimators = int(hyperparameters['n_estimators']), max_depth = int(hyperparameters['max_depth']), 
                          learning_rate = hyperparameters['learning_rate'], subsample = hyperparameters['subsample'])
    
    modelOdd = XGBClassifier(n_estimators = int(hyperparameters['n_estimators']), max_depth = int(hyperparameters['max_depth']), 
                          learning_rate =  hyperparameters['learning_rate'], subsample = hyperparameters['subsample'])
    
    
    modelEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
    modelOdd.fit(dfOdd[variables], dfEven['Class'], sample_weight=dfOdd['training_weight'])
    
    scoresEven = modelOdd.predict_proba(dfEven[variables])[:,1]
    scoresOdd = modelEven.predict_proba(dfOdd[variables])[:,1]

    dfEven['decision_value'] = ((scoresEven-0.5)*2)
    dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
    df = pd.concat([dfEven,dfOdd])
    
    sensitivity = calc_sensitivity_with_error(df)
    
    # We wish to maximise sensitivity therefore minimise -1*sensitivity 
    return {'loss': -1*sensitivity[0], 'status': STATUS_OK}

    
trials = Trials()

# This function does the optimisation using the objective function and the hyperparameters given
best = fmin(
    fn=objective_function,
    space=hyperparameters,
    algo=rand.suggest, # Random search/ stochastic
    max_evals=50 # The number of iterations
)

print(best)
      
         