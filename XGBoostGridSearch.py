# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:10:32 2020

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
from sklearn.grid_search import GridSearchCV

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



 # Dictionary of hyperparameters of BDT and possible range of values
hyperparameters = {'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
                    'max_depth': [7,8,9,10,11,12],
                    'n_estimators': [150,160,170,180,190,200,210],
                    'subsample': [0.6,0.65,0.7,0.75,0.8,0.85]
                    }
   

xgb_model = XGBClassifier()

clf = GridSearchCV(xgb_model, hyperparameters, scoring = roc_auc)

  
clfEven = clf.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
clfOdd =  clf.fit(dfOdd[variables], dfEven['Class'], sample_weight=dfOdd['training_weight'])


#trust your CV!
best_parametersEven, scoreEven, _ = max(clfEven.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', scoreEven)
for param_name in sorted(best_parametersEven.keys()):
    print("Even %s: %r" % (param_name, best_parametersEven[param_name]))
    
    
best_parametersOdd, scoreOdd, _ = max(clfOdd.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', scoreOdd)
for param_name in sorted(best_parametersOdd.keys()):
    print("Odd %s: %r" % (param_name, best_parametersOdd[param_name]))

scoresEven = clfEven.predict_proba(dfOdd[variables])[:,1]
scoredOdd = clfOdd.predict_proba(dfEven[variables])[:,1]

dfEven['decision_value'] = ((scoresEven-0.5)*2)
dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
df = pd.concat([dfEven,dfOdd])
    
sensitivity = calc_sensitivity_with_error(df)

print("Sensitivity is " + sensitivity[0] + " +/- " + sensitivity[1])

  


      
         