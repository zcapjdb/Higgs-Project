#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:02:01 2020

@author: gracehymas

"""

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")

import pandas as pd
import numpy as np

import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import scale

from nnPlotting import *

from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, rand


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

for nJets in [2,3]:

    if nJets == 2:
        variables = ['dRBB','mBB','pTB1', 'pTB2', 'MET','dPhiVBB','dPhiLBmin','Mtop','dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTags', 'nTrackJetsOR']

    else:
        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont', 'nTags', 'nTrackJetsOR']

    # Read in Data
    if nJets == 2:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

    else:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')
        
    # Process Even Events
    xEven = scale(dfEven[variables].to_numpy())
    yEven = dfEven['Class'].to_numpy()
    wEven = dfEven['training_weight'].to_numpy()

    # Process Odd Events
    xOdd = scale(dfOdd[variables].to_numpy())
    yOdd = dfOdd['Class'].to_numpy()
    wOdd = dfOdd['training_weight'].to_numpy()

     # Dictionary of hyperparameters of BDT and possible range of values
     
    hyperparameters = {'epochs': hp.quniform('epochs', 100, 300, 10),
                        'batch_size': hp.quniform("batch_size", 50, 200, 1),
                       }
    
    
     
    def DNNClassifier():

        model = Sequential()

        # Add Layers
        model.add(Dense(units=14, input_shape=(xEven.shape[1],), activation='relu')) # 1st layer
        model.add(Dense(14, init='uniform', activation='relu')) # hidden layer
        model.add(Dense(14, init='uniform', activation='relu')) # hidden layer
        model.add(Dense(1, activation='sigmoid')) # output layer
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        return model
   
    # Objective function to be minimised 
    def objective_function(hyperparameters):
        modelEven = DNNClassifier()
    
        modelOdd = DNNClassifier()
    
        # Train Model
        modelEven.fit(xEven,
                      yEven, 
                      sample_weight = wEven, 
                      epochs = int(hyperparameters['epochs']), 
                      batch_size=int(hyperparameters['batch_size']), 
                      verbose = 0)
        
        modelOdd.fit(xOdd,
                     yOdd, 
                     sample_weight = wOdd, 
                     epochs = int(hyperparameters['epochs']), 
                     batch_size=int(hyperparameters['batch_size']),  
                     verbose = 0)
    
        ## EVALUATION DNN & Plots
        dfOdd['decision_value'] = modelEven.predict_proba(xOdd)
        dfEven['decision_value'] = modelOdd.predict_proba(xEven)
        df = pd.concat([dfOdd,dfEven])
    
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
      
         
