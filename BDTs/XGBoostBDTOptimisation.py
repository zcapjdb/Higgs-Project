# XGBoost Boosted Decision Tree Classifier: Hyperparameter optimisation script which finds the optimal hyperparameters to train the XGBoost BDT with
# Author: Louis Heery

# How to Use:
# 1. Replace 'hyperparameterOne = ' with the desired Hyperparameter from: max_depth, n_estimators, learning_rate, subSample
# 2. Replace 'hyperparameterTwo = ' with the desired Hyperparameter from: max_depth, n_estimators, learning_rate, subSample
# 3. Assign hyperparameterOneValue & hyperparameterTwoValue to their correct hyperparameter, and set the other two hyperparameters to their default value.
# 4. Assign hyperparameterOneValue & hyperparameterTwoValue to their correct hyperparameter, and set the other two hyperparameters to their default value.

import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import time
import threading

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bdtPlotting import *
from sensitivity import * # sensitivity.py file, which has "calc_sensitivity_with_error" Function in it
from xgboost import XGBClassifier


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)


#max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
max_depth = [1,2,3,4,5,6,7,8, 9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]    
#n_estimators = [1, 5, 20, 50, 100, 250, 500]
n_estimators = [150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
#learning_rate = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
learning_rate = [0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3]
#sampleSize = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
sampleSize =[0.6,0.65,0.7,0.75,0.8,0.85]

hyperparameterOne = n_estimators
hyperparameterTwo = learning_rate

hyperparameterOneName = "n_estimators"
hyperparameterTwoName = "learning_rate"
numberOfIterations = len(hyperparameterOne) * len(hyperparameterTwo)

dataset = np.zeros((numberOfIterations, 9))

i = 0

for hyperparameterTwoValue in hyperparameterTwo:
    for hyperparameterOneValue in hyperparameterOne:

        print("Training Model " + str(i + 1) + "/" + str(numberOfIterations) + " with Hyperparameters of")
        print("Hyperparameter One = ", hyperparameterOneValue)
        print("Hyperparameter Two = ", hyperparameterTwoValue)

        start = time.time()

        for nJets in [2,3]:

            # Defining BDT Parameters
            if nJets == 2:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
                n_estimators = hyperparameterOneValue # Default = 200   176
                max_depth = 10 # Default = 4   10
                learning_rate = hyperparameterTwoValue # Default = 0.15     0.02375
                subsample =  0.85# Default = 0.5   0.84696

            else:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
                n_estimators = hyperparameterOneValue # Default = 200
                max_depth = 10 # Default = 4
                learning_rate = hyperparameterTwoValue # Default = 0.15
                subsample =  0.85 # Default = 0.5

            # Reading Data
            if nJets == 2:
                dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
                dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

            else:
                dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
                dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')

            # Initialising BDTs
            xgbEven = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)
            xgbOdd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)

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
            scoresEven = xgbOdd.predict_proba(dfEven[variables])[:,1]
            scoresOdd = xgbEven.predict_proba(dfOdd[variables])[:,1]

            dfEven['decision_value'] = ((scoresEven-0.5)*2)
            dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
            df = pd.concat([dfEven,dfOdd])

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

        dataset[i,0] = hyperparameterOneValue
        dataset[i,1] = hyperparameterTwoValue
        dataset[i,2] = sensitivity2Jet[0]
        dataset[i,3] = sensitivity2Jet[1]
        dataset[i,4] = sensitivity3Jet[0]
        dataset[i,5] = sensitivity3Jet[1]
        dataset[i,6] = sensitivityCombined[0] #combined
        dataset[i,7] = sensitivityCombined[1] #combined Uncertainty
        dataset[i,8] = time.time() - start

        i = i + 1

# Save Optimisation Matrix of sensitivity to CSV for further analysis
dfDataset = pd.DataFrame(dataset)
dfDataset.to_csv("MonsterLONGXGBoost_" + str(hyperparameterOneName) + "_vs_" + str(hyperparameterTwoName) + ".csv")

#### Plot Sensitivity Grid ####
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:

    if i == '2 Jets':
        data = np.reshape(dataset[:,2], (len(hyperparameterTwo), len(hyperparameterOne)))

    if i == '3 Jets':
        data = np.reshape(dataset[:,4], (len(hyperparameterTwo), len(hyperparameterOne)))

    if i == 'Combined':
        data = np.reshape(dataset[:,6], (len(hyperparameterTwo), len(hyperparameterOne)))

    # Draw a heatmap with the numeric values in each cell
    plt.figure(figsize=(60,20))
    ax = plt.axes()

    index = hyperparameterTwo
    cols = hyperparameterOne

    df = pd.DataFrame(data, index=index, columns=cols)
    sns.heatmap(data,  cmap="RdYlGn", annot=True, yticklabels=index, xticklabels=cols,  annot_kws={"size":20}, fmt='.3f', cbar=True, cbar_kws={'label': 'Sensitivity'})

    ax.tick_params(axis='y', labelsize=20, rotation=0)
    ax.tick_params(axis='x', labelsize=20, rotation=0)

    ax.set_xlabel(hyperparameterOneName, size=20)
    ax.set_ylabel(hyperparameterTwoName, size=20)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.figure.axes[-1].yaxis.label.set_size(20)

    # Save heatmap figure to PDF
    figureName = "MonsterLONGXGBoost_" + str(hyperparameterOneName) + "_vs_" + str(hyperparameterTwoName) + str(i) +".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, dpi=100, bbox_inches='tight')
    plt.show()
