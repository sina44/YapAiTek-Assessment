# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:50:26 2020

@author: TS
"""
# Import important library
import pandas as pd
# Importing dataset
climate=pd.read_csv(r'C:\Users\TS\Desktop\1.csv')

# Understanding the dataset
climate.info()
climate.columns 
climate.drop( 'timestamp',axis=1,inplace=True)

# Define polution index
def index(climate) :
    
    if climate["national"] <= 55 :
        return "Normal"
    elif (climate["national"] > 55) & (climate["national"] <= 150 ):
        return "Elevated"
    elif (climate["national"] > 150) & (climate["national"] <= 250) :
        return "High"
    elif climate["national"] > 250 :
        return "Very High"   
   
climate["national"] = climate.apply(lambda climate:index(climate),
                                      axis = 1)

# Using labelencoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
climate['national']=le.fit_transform(climate['national'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
climate[['south', 'north', 'east', 'central', 'west']]=std.fit_transform(
        climate[['south', 'north', 'east', 'central', 'west']])

# Define train & test dataset
from sklearn.model_selection import train_test_split
y=climate['national']
x=climate.drop('national',axis=1)
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

# Model Implementation
from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(x_train,y_train)

# Calculating precision score
from sklearn.metrics import precision_score
pred=model.predict(x_test)
precision= precision_score(y_test,pred)
print(precision)