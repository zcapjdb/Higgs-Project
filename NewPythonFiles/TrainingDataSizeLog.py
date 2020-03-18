# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:38:55 2019

@author: zcapjdb
"""

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import time
import threading

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from bdtPlotting import *
from sensitivity import * # sensitivity.py file, which has "calc_sensitivity_with_error" Function in it
from xgboost import XGBClassifier


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)


probability = np.array([0.001,0.002,0.003,0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


dataset = np.zeros((len(probability), 8))



categories = ["VH", "diboson", "stop", "ttbar_mc_a", "V+jets"]

for x in categories:
    i = 0
#dfOddParticularEventType = dfOdd.loc[dfOdd['category'] == x]
#dfEvenParticularEventType = dfEven.loc[dfEven['category'] == x]
    

    for p in probability:
        print("Varying " + str(x) + ":")
        print("Training Model with " + str(p*100) + "% " "of training data used.")

        start = time.time()


        for nJets in [2,3]:

            # Defining BDT Parameters
            if nJets == 2:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
                n_estimators = 200 
                max_depth = 4 
                learning_rate = 0.15
                subsample = 0.5

            else:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
                n_estimators = 200
                max_depth = 4 
                learning_rate = 0.15 
                subsample = 0.5 

            # Reading Data
            if nJets == 2:
                dfEvenTest = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
                dfOddTest = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')
                
                
                dfEvenSkip = dfEvenTest[dfEvenTest['category'] == x].sample(frac = p, replace = True, random_state = np.random.randint(1,1000))
                dfEvenNoSkip = dfEvenTest[dfEvenTest['category'] != x]
                dfEven = pd.concat([dfEvenSkip, dfEvenNoSkip])
                
                dfOddSkip = dfOddTest[dfOddTest['category'] == x].sample(frac = p, replace = True, random_state = np.random.randint(1,1000))
                dfOddNoSkip = dfOddTest[dfOddTest['category'] != x]
                dfOdd = pd.concat([dfOddSkip, dfOddNoSkip])
            

            
            else:
                 dfEvenTest = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
                 dfOddTest = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')
                 
                                 
                 dfEvenSkip = dfEvenTest[dfEvenTest['category'] == x].sample(frac = p, replace = True, random_state = np.random.randint(1,1000))
                 dfEvenNoSkip = dfEvenTest[dfEvenTest['category'] != x]
                 dfEven = pd.concat([dfEvenSkip, dfEvenNoSkip])
                
                 dfOddSkip = dfOddTest[dfOddTest['category'] == x].sample(frac = p, replace = True, random_state = np.random.randint(1,1000))
                 dfOddNoSkip = dfOddTest[dfOddTest['category'] != x]
                 dfOdd = pd.concat([dfOddSkip, dfOddNoSkip])
    
    
            # Initialising BDTs
            xgbEven = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)
            xgbOdd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)

            #dfEven[variables] = dfEven[variables].apply(pd.to_numeric, errors='ignore')
            #dfOdd[variables] = dfOdd[variables].apply(pd.to_numeric, errors='ignore')

            # Setup multiple thread training of BDT
            def train_even():
                xgbEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
            def train_odd():
                xgbOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

            t = threading.Thread(target=train_even)
            t2 = threading.Thread(target=train_odd)

            t.start()
            t2.start()
            t.join()
            t2.join()

            # Scoring
            scoresEven = xgbOdd.predict_proba(dfEvenTest[variables])[:,1]
            scoresOdd = xgbEven.predict_proba(dfOddTest[variables])[:,1]

            dfEvenTest['decision_value'] = ((scoresEven-0.5)*2)
            dfOddTest['decision_value'] = ((scoresOdd-0.5)*2)
            df = pd.concat([dfEvenTest,dfOddTest])

            # Calculating Sensitivity
            if nJets == 2:
                sensitivity2Jet = calc_sensitivity_with_error(df)
                print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

            else:
                sensitivity3Jet = calc_sensitivity_with_error(df)
                print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

        sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])

        print("Combined Sensitivity = ", sensitivityCombined[0], "±",sensitivityCombined[1])
        print("Time Taken = ß", time.time() - start)

        dataset[i,0] = p
        dataset[i,1] = sensitivity2Jet[0] # 2 jet sensitivity
        dataset[i,2] = sensitivity2Jet[1] # 2 jet uncertainty
        dataset[i,3] = sensitivity3Jet[0] # 3 jet sensitivity
        dataset[i,4] = sensitivity3Jet[1] # 3 jet uncertainty
        dataset[i,5] = sensitivityCombined[0] #combined
        dataset[i,6] = sensitivityCombined[1] #combined Uncertainty
        dataset[i,7] = time.time() - start

        i = i + 1
        
    # Save dataset
    if x == "VH":
        dfDataset = pd.DataFrame(dataset)
        dfDataset.to_csv("Log XGBoost Sensitivity for varying " + str(x) + " dataset size .csv") 
            
    elif x == "diboson":
        dfDataset = pd.DataFrame(dataset)
        dfDataset.to_csv("Log XGBoost Sensitivity for varying " + str(x) + " dataset size .csv") 
            
    elif x == "ttbar_mc_a":
        dfDataset = pd.DataFrame(dataset)
        dfDataset.to_csv("Log XGBoost Sensitivity for varying " + str(x) + " dataset size .csv") 
            
    elif x == "stop":
        dfDataset = pd.DataFrame(dataset)
        dfDataset.to_csv("Log XGBoost Sensitivity for varying " + str(x) + " dataset size .csv") 
            
    else:
        dfDataset = pd.DataFrame(dataset)
        dfDataset.to_csv("Log XGBoost Sensitivity for varying " + str(x) + " dataset size .csv") 
   
    ############ Plot Sensitivty Against Training Data Size ######################
    graphs = ['2 Jets', '3 Jets', 'Combined']

    percentage = dfDataset[0] * 100
    x_value = np.log10(percentage) 
    
    y_2Jet = dfDataset[1]
    y_2JetError = dfDataset[2]
    
    y_3Jet = dfDataset[3]
    y_3JetError = dfDataset[4]
    
    y_Combined = dfDataset[5]
    y_CombinedError = dfDataset[6]    
 
    plt.figure()

    plt.plot(x_value, y_2Jet, 'r-', label = "2 Jet")
    plt.fill_between(x_value, y_2Jet - y_2JetError, y_2Jet + y_2JetError, facecolor = 'red', alpha = 0.5)
    
    plt.plot(x_value, y_3Jet, 'b-', label = "3 Jet")
    plt.fill_between(x_value, y_3Jet - y_3JetError, y_3Jet + y_3JetError, facecolor = 'blue', alpha = 0.5)
    
    plt.plot(x_value, y_Combined, 'm-', label = "Combined")
    plt.fill_between(x_value, y_Combined - y_CombinedError, y_Combined + y_CombinedError, facecolor = 'magenta', alpha = 0.5)

    #plt.xticks(np.arange(0, 63146, 10000))
    plt.xlim(np.min(x_value), np.max(x_value))
    
    plt.xlabel("Log of percentage of training dataset size")
    plt.ylabel("Sensitivity")
    plt.legend(loc ='lower right')
    plt.grid(True)  
  
    #figureName =(x +"Training Size.pdf")
    fig = plt.gcf()
    plt.savefig(str(x) + " Log Training Size.pdf", bbox_inches='tight',dpi=300)
    plt.show()    

    
    
    
    
    
    