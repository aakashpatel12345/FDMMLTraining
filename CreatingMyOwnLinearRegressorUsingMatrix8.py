# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:35:34 2020

@author: Aakash
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:30:12 2020

@author: Aakash
"""

import pandas as pd
import numpy as np
#import statsmodels.formula.api as sm


data = pd.read_csv("./my_linear_regressor.csv")

x=np.array(data[["x1" , "x2", "x3"]])
y=np.array(data[["y"]])



def gradient_descent(x,y,learning_rate=0.011,num_iterations = 10000):
    
    sample_size = len(x)
    constant_column = np.array(np.ones((sample_size,1)).astype(int))
    x = np.append(arr = constant_column, values = x ,axis = 1)

    num_of_coefs = x.shape[1] # includes y-intercept
    
    matrix_of_coef = np.array(np.zeros((num_of_coefs, 1)))

    for i in range(num_iterations):
       y_pred = np.dot(x,matrix_of_coef) #why doesn't return a scalar - check documentation
       
       cost = (1/(2*sample_size)) *  sum((y-y_pred)** 2)
       
       matrix_of_gradients_of_coefs = np.array(np.zeros((num_of_coefs, 1)))
       for j in range(num_of_coefs):
           difference = y-y_pred
           matrix_of_gradients_of_coefs[j,0] = -(1/sample_size) * sum(np.dot(x[:,j],difference))
           matrix_of_coef[j,0] = matrix_of_coef[j,0] - (learning_rate*matrix_of_gradients_of_coefs[j,0])
           
       print ("The gradient of term 1 in this iteration is {}".format(matrix_of_coef[1,0]))
       print ("The gradient of term 2 in this iteration is {}".format(matrix_of_coef[2,0]))
       print ("The gradient of term 3 in this iteration is {}".format(matrix_of_coef[3,0]))
       print("The y_intercept in this iteration is {}".format(matrix_of_coef[0,0]))
       print("This is iteration number {} and it costs {}".format(i, cost))
       print('\n')
       
       
       '''
       
       #calculate step sizes
       break_trigger = False
       for k in range(num_of_coefs):
           placeholder = learning_rate*matrix_of_gradients_of_coefs[k,0]
           if abs(placeholder)<0.00001:
               break_trigger = True
           
       if break_trigger:
           print('Step size is too small')
           break
       
       if cost == 0:
           print("You have perfectly fit the data")
           break
           
           '''

   
    
gradient_descent(x,y)    
    
    
    
    
    
    
    