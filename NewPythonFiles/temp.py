# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import pandas as pd
import random as random


dfEven = pd.read_csv('VHbb_data_3jet_even.csv')
dfEven2 = pd.read_csv('VHbb_data_2jet_even.csv')
print(len(dfEven))
print(len(dfEven2))

#a = range(1, 300, 10)
#print(a)




#p = np.linspace(0,1, num = 11)
#print(p)


#df = pd.read_csv('csvtest.csv',header = 0, skiprows = lambda j : j>0 and random.random()>0.1)
#print(df)

#df2 = pd.read_csv('csvtest.csv')
#print(df2)

#print(np.arange(0, 63416, 10000))

#for i in p:
    #dfEven = pd.read_csv('csvtest.csv',header = 0, skiprows = lambda j : j>0 and random.random()>i)
    #print("Using "+ str(i*100) +  "% " + "of the data")
    #print(dfEven)

#x = "VH"
#p = 0.2

#dfEvenTest = pd.read_csv('VHbb_data_2jet_even.csv')
#dfEvenSkip = dfEvenTest[dfEvenTest['category'] == x].sample(frac = p, replace = True, random_state = np.random.randint(1,1000))
#dfEvenNoSkip = dfEvenTest[dfEvenTest['category'] != x]
#dfEven = pd.concat([dfEvenSkip, dfEvenNoSkip])

#print(dfEven)
#import numpy as np
#import matplotlib.pyplot as plt

#a = np.array([0.1,1,5,10,20,30,40,50,60,70,80,90,100])

#b = a/100
#print(a)
#print(b)

#plt.figure()

#x = np.arange(0,100)
#y = np.sin(x)

#z= "abc"

#plt.plot(x,y)
#plt.savefig(str(z) + ".pdf")


#fruit = ['apple', 'orange', 'banana']
#for x in fruit:
 #   print(str(x) + "apple")


#dfEven2 = pd.read_csv('VHbb_data_2jet_even.csv', skiprows = lambda j : j > 3 and random.random() > p and dfEven.loc[dfEven['category']] == x)

#df = pd.read_csv('csvtest.csv', skiprows = lambda j: j>0 and random.random() > 0.3).query("e != 'apple'")
#print(df)

#df2 = df.loc[df['e'] == 'apple']
#print(df2)

#df = pd.read_csv('csvtest.csv')
#print(df)

#df2 = pd.read_csv('csvtest.csv', skiprows = lambda j: j>0 and random.random() > 0.3).query("e == 'apple'")
#df3 = pd.read_csv('csvtest.csv').query("e != 'apple'")
#print(df2)
#print(df3)

#df4 = pd.concat([df2, df3])
#print(df4)


#df5 = df[df['e'] == 'orange' ]
#print(df5)

#df6 = df[df['e'] == 'banana'].sample(frac = 0.5, replace = True, random_state = np.random.randint(1,1000))
#print(df6)

#df7 = pd.concat([df5,df6])
#print(df7)
