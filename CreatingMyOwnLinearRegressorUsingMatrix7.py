# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:30:12 2020

@author: Aakash
"""

import pandas as pd
import numpy as np
#import statsmodels.formula.api as sm


data = pd.read_csv("./Datasets/admissions.csv")

#create independent and dependent variables
x=np.array(data[["TOEFL Score"]])
y=np.array(data[["CGPA"]]) #have in same form as x

sample_size = len(x)

constant_column = np.array(np.ones((sample_size,1)).astype(int))

#add a column of 1's to the front of the data set 
x = np.append(arr = constant_column, values = x ,axis = 1)
#in x, 1st column = y-intercept coefficent, 

num_of_coefs = x.shape[1] # includes y-intercept

def gradient_descent(x,y,learning_rate=0.0001,num_iterations = 10):
    #use matrices
    #create a matrix of coeffs - num_of_coefs by 1
    matrix_of_coef = np.array(np.zeros((num_of_coefs, 1)))
    #maybe have coefs that aren't zero to start with
    #how to determine initial gradient and t-intercept
    
    for i in range(num_iterations):
       #y_pred = x * matrix_of_coef  #should be this
       y_pred = np.dot(x,matrix_of_coef) #why doesn't return a scalar - documentation - x is not 1d
       
       #list comprehension 
       cost = (1/(2*sample_size)) * sum(val ** 2 for val in (y-y_pred))
       
       #need to calculate the gradient for each independent variable
       matrix_of_gradients_of_coefs = np.array(np.zeros((num_of_coefs, 1)))
       for j in range(num_of_coefs):
           difference = y-y_pred
           matrix_of_gradients_of_coefs[j,0] = -(1/sample_size) * sum(np.dot(x[:,j],difference)) # maybe transverse and multiply
           #matrix_of_gradients_of_coefs[j,0] = -(1/sample_size) * sum(x[:,j]*(y-y_pred))
           matrix_of_coef[j,0] = matrix_of_coef[j,0] - (learning_rate*matrix_of_gradients_of_coefs[j,0])
           #problem with optimising all variables at once
           
           
       print ("The gradient in this iteration is {}".format(matrix_of_coef[1,0]))
       print("The y_intercept in this iteration is {}".format(matrix_of_coef[0,0]))
       print("This is iteration number {} and it costs {}".format(i, cost))
       print('\n')
       
       if abs(learning_rate*m_grad)<0.00001 and abs(learning_rate*b_grad)<0.00001:
           print (learning_rate*m_grad)
           break
       
       if cost == 0:
           print("You have perfectly fit the data")
           break

   
gradient_descent(x,y)    
    
    
    
    
    
    
    