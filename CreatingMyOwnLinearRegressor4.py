# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:16:35 2020

@author: Aakash
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("./Datasets/admissions.csv")

#create independent and dependent variables
x=np.array(data["TOEFL Score"])
y=np.array(data["CGPA"])

def gradient_descent(x,y,learning_rate=0.0001,num_iterations = 10):
    m_curr = 0
    b_curr = 0
    n = len(y)
    
    for i in range(num_iterations):
       y_pred = (m_curr*x) + b_curr
       #list comprehension 
       cost = (1/(2*n)) * sum(val ** 2 for val in (y-y_pred))
       
       m_grad = -(1/n) * sum(x*(y-y_pred))
       
       b_grad = -(1/n) * sum(y-y_pred)
       
       m_curr = m_curr - (learning_rate*m_grad)
       
       b_curr = b_curr - (learning_rate*b_grad)
       ''' print ("The gradient in this iteration is {}".format(m_curr))
       print("The y_intercept in this iteration is {}".format(b_curr))
       print("This is iteration number {} and it costs {}".format(i, cost))'''
    return m_curr,b_curr,y_pred

   
       #need to account for multiple independent variables 
       #multiple m's
       
       #increase learning rate as algorithm proceeds - maybe
       #
       
mCurrent, bCurrent, y_pred_train = gradient_descent(x,y) 

'''#for training set
#y_pred = mCurrent*x_train + bCurrent
plt.scatter(x_train,y_train, c = "red")
plt.plot(x_train,y_pred, c = "blue") #model line of best fit'''


'''#for testing set
#y_pred = mCurrent*x_train + bCurrent
plt.scatter(x_test,y_test, c = "red")
plt.plot(x_train,y_pred, c = "blue") #same as before'''

plt.scatter(x,y, c = "blue")
plt.plot(x,y_pred_train, c = "red")