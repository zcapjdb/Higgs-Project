# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:37:55 2020

@author: zcapjdb
"""

import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.preprocessing import scale
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D, Activation)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import scale
from keras.utils import plot_model
from keras.callbacks import History
from keras import metrics
import time
import os
from copy import deepcopy
import tensorflow as tf


import sys
sys.path.append("../dataset-and-plotting")
from nnPlotting import *
from sensitivity import *

mirrored_strategy = tf.distribute.MirroredStrategy()


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)


#variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB',
          #   'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH',
            # 'mTW', 'pTV']




variables = ['pTB1','pTB2']

np.random.seed(10)


dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

df = pd.concat([dfEven,dfOdd])

dfTrain = df.loc[df['category'] == 'VH']
x_train_array = dfTrain[variables].to_numpy()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_array)


batch_size = 64
steps_per_epoch = int(len(dfTrain) / batch_size)
epochs = 5000

noise_dim = 100

optimizer = Adam(0.0001, 0.5)
#optimizer = SGD(0.01)


start = time.time()

def create_generator():
    
    generator = Sequential()
    
    generator.add(Dense(50,input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(25))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(5))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(len(variables), activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_descriminator():
    discriminator = Sequential()
     
    discriminator.add(Dense(4, input_dim=len(variables)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(3))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(2))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


discriminator = create_descriminator()
generator = create_generator()

# Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
discriminator.trainable = False

# Link the two models to create the GAN
gan_input = Input(shape=(noise_dim,))
fake_event = generator(gan_input)

gan_output = discriminator(fake_event)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)


dloss = []
gloss = []

for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        
        x = np.concatenate((real_x, fake_x))


        # First half of y-values are 0.9 corresponding to real data, not using 1 for label smoothing
        # Second half are 0 corresponding to the fake data we made
        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, disc_y)
        
        discriminator.trainable = False
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)
        
        
    dloss.append(d_loss)
    gloss.append(g_loss)   
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

df = pd.DataFrame(np.array([dloss,gloss]).T, columns = ['discriminator', 'generator'])
df.to_csv("ThisIsLossSimpleLong.csv")

discriminator.save('Discriminator.hdf5')
generator.save('Generator.hdf5')



