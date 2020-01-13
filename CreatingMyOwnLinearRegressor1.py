# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:54:43 2020

@author: Aakash
"""

#import libraries 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


class AakashLinearRegression:
    coef_ = 0
    intercept_ = 0
    
    
    def  cal_cost(theta,X,y):
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions-y))
        return cost
    
    def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
        m = len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations,2))
        for it in range(iterations):    
            #prediction = np.dot(float(X),theta)   
            
            theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
            theta_history[it,:] =theta.T
            cost_history[it]  = cal_cost(theta,X,y)
        return theta, cost_history, theta_history


    def fit(x,y):
        #should be able to handle x being a 1d array or '2+'d array
        
        pass
    
    def predict(x,y):
        pass
    
    def score(x,y):
        pass

ALR = AakashLinearRegression()
ALR.coef_




#w = np.array([[3,4,5,6]]).T
#q = np.array([[3,4,5,6]])

#read data int
data = pd.read_csv("admissions.csv")

#check form of data
data.head()

#create independent and dependent variables
x=data["TOEFL Score"]
y=data["CGPA"]

#plot a scatter to see the relationship between the results
plt.scatter(x,y)

#ccreate instance of the Linear Regression Class
lr = LinearRegression()
x=data["TOEFL Score"].values.reshape(-1,1)
y=data["CGPA"]
lr.fit(x,y)

#obtain the gradient and intercept
lr.coef_
#gives 0.0806
lr.intercept_
#gives -0.0640585

ALR = AakashLinearRegression()
x=data["TOEFL Score"]
y=data["CGPA"]
thetaPractice = np.array([[0],[0]])
#thetaPractice = np.random.randn(2,1)
X_b = np.c_[np.ones((len(x),1)),x]

#initial guess is 0 and 0

theta,cost_history,theta_history=ALR.gradient_descent(X_b,y,thetaPractice)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

#plot predictions vs actual
yPredict = lr.predict(x)
plt.scatter(x,y, c = "blue")
plt.plot(x,yPredict, c = "red")
lr.score(x,y)


