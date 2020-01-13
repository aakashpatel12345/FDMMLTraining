# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:43:22 2020

@author: Aakash
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def cal_cost(theta,X,y):
    m = len(y)
    predictions = np.dot(X, theta)
    #predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost
    
def gradient_descent(X,y,theta,learning_rate=0.00001,iterations=1000):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):    
        prediction = np.dot(X,theta)   
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        #theta_history[it,:] = theta.T
        
        firstNum = np.array([theta.T[0,0]])
        theta_history[it,0] = firstNum
        
        secondNum = np.array([theta.T[0,1]])
        theta_history[it,1] = secondNum
        
        cost_history[it]  = cal_cost(theta,X,y)
    return theta, cost_history, theta_history
    

data = pd.read_csv("./Datasets/admissions.csv")

#create independent and dependent variables
x=np.array(data["TOEFL Score"])
y=np.array(data["CGPA"])

#thetaPractice = np.array([[0],[0]])
thetaPractice = np.random.randn(2,1)
X_b = np.c_[np.ones((len(x),1)),x]

#initial guess is 0 and 0

theta,cost_history,theta_history=gradient_descent(X_b,y,thetaPractice)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))





