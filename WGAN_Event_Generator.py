# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:01:42 2020

@author: zcapjdb
"""

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



variables = ['pTB1','pTB2']

dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

df = pd.concat([dfEven,dfOdd])

dfTrain = df.loc[df['category'] == 'VH']
x_train_array = dfTrain[variables].to_numpy()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_array)


generator = load_model('WassersteinGenerator.hdf5')
print(generator.summary())

GAN_noise_size = 64
n_events = 10000

X_noise = np.random.normal(0, 1, size=(n_events, GAN_noise_size))

X_generated = generator.predict(X_noise)

#scaler = StandardScaler()
X_unscaled = scaler.inverse_transform(X_generated)

Events = pd.DataFrame(X_unscaled)
Events.to_csv('WassersteinGeneratedEvents.csv')

