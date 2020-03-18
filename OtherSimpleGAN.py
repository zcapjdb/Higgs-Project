
"""
Created on Wed Mar  4 12:10:56 2020

@author: gracehymas
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import scale
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
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

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def create_generator():
    
    generator = Sequential()
    
    generator.add(Dense(30, activation="tanh",input_dim=noise_dim))
    generator.add(Dense(2, activation="softmax"))
    
    generator.add(Dense(len(variables), activation='tanh'))
                  
    generator.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return generator


def create_descriminator():
    
    discriminator = Sequential()
    
    discriminator.add(Dense(40, activation="linear", input_dim=len(variables)))
    discriminator.add(Dense(40, activation="tanh"))
    discriminator.add(Dense(40, activation="tanh"))
    
    discriminator.add(Dense(1, activation="sigmoid"))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimiserClassifier)#, metrics=['binary_accuracy'])
    
    return discriminator


mirrored_strategy = tf.distribute.MirroredStrategy()

variables = ['pTB1','pTB2']

np.random.seed(10)

dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

df = pd.concat([dfEven,dfOdd])

dfTrain = df.loc[df['category'] == 'VH']
x_train_array = dfTrain[variables].to_numpy()

#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train_array)
x_train = x_train_array

batch_size = 64
steps_per_epoch = int(len(dfTrain) / batch_size)

noise_dim = 100
epochs = 5000

optimizer = SGD(0.01)

optimiserClassifier = SGD(lr=0.001, momentum=0.5, decay=0.00001)

discriminator = create_descriminator()
generator = create_generator()

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
        
    #dloss.append(d_loss[0])
    dloss.append(d_loss)
    gloss.append(g_loss)   
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

df = pd.DataFrame(np.array([dloss,gloss]).T, columns = ['discriminator', 'generator'])
df.to_csv("ThisIsLossOther.csv")

discriminator.save('OtherDiscriminator.hdf5')
generator.save('OtherGenerator.hdf5')