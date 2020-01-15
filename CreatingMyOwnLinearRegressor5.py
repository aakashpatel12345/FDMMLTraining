# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:16:35 2020

@author: Aakash
"""
import pandas as pd
import numpy as np


data = pd.read_csv("./Datasets/admissions.csv")

#create independent and dependent variables
x=np.array(data[["TOEFL Score" , "GRE Score"]])
y=np.array(data["CGPA"])

NumOfDependentVariables = x.shape[1]
def gradient_descent(x,y,learning_rate=0.0000001,num_iterations = 10):
    #initialise y-intercept and gradient
    m = {}
    y_pred = y*0 
    #dictinarys are unordered therefore for loop won't be in order necesserily 
    #try and order via key?
    for i in range(0,NumOfDependentVariables):
        key = "m_curr_"+str(i)
        value=0
        m[key] = value
    
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
       print("\n")


   
       #need to account for multiple independent variables 
       #multiple m's
       
       #increase learning rate as algorithm proceeds - maybe
       
       #
       
gradient_descent(x,y) 
