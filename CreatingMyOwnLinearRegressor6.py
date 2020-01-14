# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:58:33 2020

@author: aakash.patel
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv("./data/medical_costs.csv")



#split catergorical data
#first sex - split into 1
dataEncoded = pd.get_dummies(data,columns = ['sex'], drop_first=True)

#then smoker - split into 1
dataEncoded = pd.get_dummies(dataEncoded,columns = ['smoker'], drop_first=True)

#then region - split into 3
dataEncoded = pd.get_dummies(dataEncoded,columns = ['region'], drop_first=True)

#standardisation
scalar = MinMaxScaler()
dataScaled = dataEncoded.copy()
dataScaled[['age', 'bmi', 'children', 'charges']] = \
scalar.fit_transform(dataEncoded[['age', 'bmi', 'children', 'charges']])

#create independent and dependent variables
x = np.array(dataScaled[["age","bmi","smoker_yes","children",'region_northwest', 'region_southeast', 'region_southwest']])
y = np.array(dataScaled["charges"])

NumOfDependentVariables = x.shape[1]


def gradient_descent(x,y,learning_rate=0.001,num_iterations = 100):
    #initialise y-intercept and gradient
    m = {}
    y_pred = y*0 
    #dictinarys are unordered therefore for loop won't be in order necesserily 
    #try and order via key?
    for i in range(0,NumOfDependentVariables):
        key = "m_curr_"+str(i)
        value=0
        m[key] = value
    
    
    sorted(m) # puts keys in alphabetical order
    b_curr = 0
    n = len(y)
    
    for i in range(num_iterations):
       y_pred = y_pred + b_curr
       j=0
       for key in m:
          y_pred = y_pred + (m[key] * x[:,j]) 
          j=+1 
        
       
       #list comprehension 
       cost = (1/(2*n)) * sum(val ** 2 for val in (y-y_pred))
       #if cost = something could break
       
       
       #need to do for each m
       k=0
       for key in m:
           m_grad = -(1/n) * sum(x[:,k]*(y-y_pred))
           m[key] = m[key] - (learning_rate*m_grad)
           k=+1
       
       b_grad = -(1/n) * sum(y-y_pred)
       b_curr = b_curr - (learning_rate*b_grad)
       print ("The gradient of term 1 in this iteration is {}".format(m['m_curr_0']))
       print ("The gradient of term 2 in this iteration is {}".format(m['m_curr_1']))
       print("The y_intercept in this iteration is {}".format(b_curr))
       print("This is iteration number {} and it costs {}".format(i, cost))
       print('\n')

       if abs(learning_rate*m_grad)<0.00001 and abs(learning_rate*b_grad)<0.00001:
           print (learning_rate*m_grad)
           break
       
       if cost == 0:
           print("You have perfectly fit the data")
           break

   
       #need to account for multiple independent variables 
       #multiple m's
       
       #increase learning rate as algorithm proceeds - maybe
       
       #
       
gradient_descent(x,y) 
