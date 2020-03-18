# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:24:05 2020

@author: zcapjdb
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset = pd.read_csv('NEWXGBoost_max_depth_vs_Num_Estimators.csv', sep='delimiter', header=1)
#### Plot Sensitivity Grid ####
graphs = ['2 Jets', '3 Jets', 'Combined']

hyperparameterOne = [1,2,3,4,5,6,7,8, 9, 10, 11, 12,13,14,15,16,17,18,19,20]
hyperparameterTwo = [150,160,170,175,180,190,200,210,220,230,240,250]

for i in graphs:

    index = hyperparameterTwo
    cols = hyperparameterOne
    
    if i == '2 Jets':
        data = dataset.pivot(index = 'a', columns = 'b', values = 'c')

    if i == '3 Jets':
        data = dataset.pivot(index = 'a', columns = 'b', values = 'e')

    if i == 'Combined':
        data = dataset.pivot(index = 'a', columns = 'b', values = 'g')

    # Draw a heatmap with the numeric values in each cell
    plt.figure(figsize=(40,16))
    ax = plt.axes()



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
    figureName = "LongXGBoost_" + str(hyperparameterOneName) + "_vs_" + str(hyperparameterTwoName) + str(i) +".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, dpi=100, bbox_inches='tight')
    plt.show()
