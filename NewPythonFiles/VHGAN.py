# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:37:55 2020

@author: zcapjdb
"""

import numpy as np
from numpy import mean
import pandas as pd
from keras import backend
from sklearn.preprocessing import scale
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import *
from keras.optimizers import SGD, Adam, RMSprop
from keras.constraints import Constraint

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

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)


#variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB',
    #        'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH',
         #    'mTW', 'pTV']


variables = ['pTB1','pTB2','mBB','dRBB']

#np.random.seed(10)


dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

df = pd.concat([dfEven,dfOdd])

#dfTrain = df.loc[df['category'] == 'ttbar_mc_a']
dfTrain = df.loc[df['category'] == 'VH']
x_train = dfTrain[variables].to_numpy()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


batch_size = 64
steps_per_epoch = int(len(dfTrain) / batch_size)
epochs = 500

noise_dim = 64

#optimizer = Adam(0.0002, 0.5)
#optimizer = SGD(0.01)
optimizer = RMSprop(lr = 0.0001)


def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)


class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

start = time.time()

def create_generator():
    generator = Sequential()
    
    generator.add(Dense(20, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    #generator.add(Dense(20))
    #generator.add(LeakyReLU(0.2))

    generator.add(Dense(20))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(10))
    generator.add(LeakyReLU(0.2))
    
   

    generator.add(Dense(len(variables), activation='tanh'))
    
    #generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_descriminator():
    discriminator = Sequential()
    
    const = ClipConstraint(0.01)
     
    #,kernel_constraint=const
    discriminator.add(Dense(4, input_dim=len(variables),kernel_constraint=const))
    #discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(10,kernel_constraint=const))
    #discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(10,kernel_constraint=const))
    #discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='linear'))
    
    discriminator.compile(loss= wasserstein_loss, optimizer=optimizer)
    return discriminator

discriminator = create_descriminator()
generator = create_generator()

def define_gan(generator, critic):
	# make weights in the critic not trainable
	critic.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	model.compile(loss=wasserstein_loss, optimizer=optimizer)
	return model

gan = define_gan(generator, discriminator)

dloss = []
gloss = []

dlosstemp = []

def data_gen():
    noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
    fake_x = generator.predict(noise)
    real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]        
    x = np.concatenate((real_x, fake_x))
    return noise, x

for epoch in range(epochs):
    for batch in range(steps_per_epoch):
      
        # Wasserstein
        disc_y = np.ones(2*batch_size)
        disc_y[:batch_size] = -1
        
        # Train discriminator more than generator
        for _ in range(5):
            noise, x = data_gen()
            d_loss = discriminator.train_on_batch(x, disc_y)
            dlosstemp.append(d_loss)
            
        y_gen = np.ones(batch_size)
        noise, x = data_gen()
        g_loss = gan.train_on_batch(noise, y_gen)
        
        
    dloss.append(mean(dlosstemp))
    gloss.append(g_loss)   
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

df = pd.DataFrame(np.array([dloss,gloss]).T, columns = ['discriminator', 'generator'])
df.to_csv("ThisIsLossAgainWasserstein.csv")

discriminator.save('WassersteinDiscriminator.hdf5')
generator.save('WassersteinGenerator.hdf5')



