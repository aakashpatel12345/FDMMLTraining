# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:41:54 2020

@author: aakash.patel
"""

import pandas as pd
import numpy as np



data = pd.read_csv("./my_linear_regressor.csv")

x=np.array(data[["x1" , "x2", "x3"]])
y=np.array(data[["y"]])


def gradient_descent(x,y,learning_rate=0.001,num_iterations = 10):
    NumOfDependentVariables = x.shape[1]
    
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
       j = 0 
       for key in m:
          y_pred = y_pred + (m[key] * np.array([x[:,j]]).T) 
          j=+1 
        
       
       #list comprehension 
       cost = (1/(2*n)) * sum(val ** 2 for val in (y-y_pred))
       #if cost = something could break
       
       
       #need to do for each m
       k=0
       for key in m:
           m_grad = -(1/n) * sum(np.array([x[:,k]]).T*(y-y_pred))
           m[key] = m[key] - (learning_rate*m_grad)
           k=+1
       
       b_grad = -(1/n) * sum(y-y_pred)
       b_curr = b_curr - (learning_rate*b_grad)
       
       
       print ("The gradient of term 1 in this iteration is {}".format(m['m_curr_0']))
       print ("The gradient of term 2 in this iteration is {}".format(m['m_curr_1']))
       print("The y_intercept in this iteration is {}".format(b_curr[0]))
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

