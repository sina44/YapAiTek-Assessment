# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:50:26 2020

@author: TS
"""

import pandas as pd
import numpy as np

train=pd.read_csv(r'C:\Users\TS\Desktop\1.csv')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
train = pd.read_excel(r'/content/1.csv')
'''print(train.head())
train=train.drop(train.index[0])
train1 = train.iloc[:, 2:18].values
train1=pd.DataFrame(train1)

#تعریف tr
tr=train1.iloc[112:120]
tr=tr.values
#
#حذف tr
train1=train1.drop(train1.index[[112,113,114,115,116,117,118]])
training_set=train1
'''
train.columns
train.info()
train.drop( 'timestamp',axis=1,inplace=True)

def tenure_lab(train) :
    
    if train["national"] <= 55 :
        return "Normal"
    elif (train["national"] > 55) & (train["national"] <= 150 ):
        return "Elevated"
    elif (train["national"] > 150) & (train["national"] <= 250) :
        return "High"
    elif train["national"] > 250 :
        return "Very High"   
    
train["national"] = train.apply(lambda train:tenure_lab(train),
                                      axis = 1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['national']=le.fit_transform(train['national'])
# Feature Scaling
