# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:08:59 2020

@author: zcapjdb
"""

import numpy as np
import pandas as pd
from keras.models import load_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
import os



#generator = load_model('UnscaledGenerator.hdf5')
#generator = load_model('OtherGenerator.hdf5')
generator = load_model('ScaledGenerator.hdf5')
print(generator.summary())

GAN_noise_size = 100
n_events = 1000000

X_noise = np.random.normal(0, 1, size=(n_events, GAN_noise_size))

X_generated = generator.predict(X_noise)


variables = ['pTB1','pTB2', 'mBB', 'dRBB']

dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')
df = pd.concat([dfEven,dfOdd])
indexNames = df[ (df['pTB1'] >= 600000) & (df['pTB2'] >= 300000) ].index
df.drop(indexNames , inplace=True)
dfTrain = df.loc[df['category'] == 'VH']
x_train = dfTrain[variables].to_numpy()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

X_unscaled = scaler.inverse_transform(X_generated)

#Events = pd.DataFrame(X_unscaled)
Events = pd.DataFrame(X_generated)
#Events.to_csv('UnscaledGeneratedEvents.csv')
Events.to_csv('ScaledGeneratedEvents.csv')

