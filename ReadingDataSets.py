# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:18:12 2020

@author: aakash.patel
"""

import pandas as pd

def read_car_insurance_cold_calls():
    '''
    

    Returns - A DataFrame Object - 
    Includes Independent Variables - Id, Age, Job, Marital, Education, Default,
    Balance, HHInsurance, CarLoan, Communication, LastContactDay,
    LastContactMonth, NoOfContacts, DaysPassed, PrevAttempts, Outcome,
    CallStart, CallEnd
    
    Includes Dependent Variables - CarInsurance 
    -------
    TYPE
        DESCRIPTION - Reads in all Data from car_insurance_cold_calls.csv
        Could do feature selection/feature engineering here and pass into IPNB

    '''
    return pd.read_csv('./data/car_insurance_cold_calls.csv')

#car_insurance_cold_calls_tester = read_car_insurance_cold_calls()