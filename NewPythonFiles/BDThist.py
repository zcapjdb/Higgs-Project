#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:38:28 2020

@author: gracehymas
"""

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import pickle

from sklearn.preprocessing import scale

from bdtPlotting import *
from nnPlotting import *
from sensitivity import *
from xgboost import XGBClassifier
import time


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

start = time.time()

for nJets in [2,3]:

    print("************")
    print("STARTED XGBoost Classifier")

    if nJets == 2:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

        variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV']

        n_estimators = 200#176#200
        max_depth = 4#10#4
        learning_rate = 0.15#0.025#0.15
        subsample = 0.5#0.85#0.5

    else:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')

        variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']

        n_estimators = 200
        max_depth = 4
        learning_rate = 0.15
        subsample = 0.5
    
   
    dataStorage = np.zeros((500, 7))

    start = time.time()

    

    xgbEven = XGBClassifier(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             subsample=subsample)
    xgbOdd = XGBClassifier(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             subsample=subsample)

    xgbEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
    xgbOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

    # Calculate Score of Trained BDT
    scoresEven = xgbOdd.predict_proba(dfEven[variables])[:,1]
    scoresOdd = xgbEven.predict_proba(dfOdd[variables])[:,1]

    dfEven['decision_value'] = ((scoresEven-0.5)*2)
    dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
    df = pd.concat([dfEven,dfOdd])
    
    df.to_csv("Plotcsv" + str(nJets) + ".csv")

    #### GRAPHS ####
    #dfSorted = df.sort_values('decision_value')

    ##### Signal Events in LOW (-1.0 to +0.8) & HIGH (+0.8 to +1.0) NN REGION GRAPHS #####
    #dfSorted_minus1plus08 = dfSorted.loc[dfSorted['decision_value'] <= 0.8]
    #dfSorted_plus08plus1 = dfSorted.loc[dfSorted['decision_value'] > 0.8]

    #dfNEW = dfSorted_minus1plus08
    #savedSignalData = (dfNEW.loc[dfNEW['Class'] == 1])['mBB_raw'] #you can also use dfNEW['column_name']
    #plt.hist([savedSignalData], 400,  label="(-1.0 to +0.8) of $BDT_{output}$", stacked=True, alpha=0.75)

    #dfNEW2 = dfSorted_plus08plus1
    #savedSignalData = (dfNEW2.loc[dfNEW2['Class'] == 1])['mBB_raw'] #you can also use dfNEW['column_name']
    #plt.hist([savedSignalData], 50,  label="(+0.8 to +1.0) of $BDT_{output}$", stacked=True, alpha=0.75)

    #plt.xlabel("mBB, MeV")
    #plt.xlim(0,250000)
    #plt.xticks(rotation=45)
    #plt.ylabel('Events')
    #plt.legend()
    #plt.title('mBB Signal - in Low & High BDT_output Regions')
    #plt.grid(True)

    # save figure
    #figureName = "mBB_BDT_SignalLowVsHighRegion" + str(nJets) + "Jet.pdf"
    #fig = plt.gcf()
    #plt.savefig(figureName, dpi=100, bbox_inches='tight')
    #plt.show()
    #fig.clear()
    #####

print("Time Taken", time.time() - start)
print("FINISHED")
print("************")