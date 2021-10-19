
#import relevant packages
import pandas as pd
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from geopy.distance import geodesic


# import Data
data=pd.read_csv(r'C:\Users\mohammad.jakaria\OneDrive - University of South Carolina - Moore School of Business\Desktop\Problem Set 5-Computational\radio_merger_data.csv')
data.describe()


# convert price in millions of dollar  
###data['price_million'] = data['price']/1000000

# convert population in millions of numbers 
###data['population_target_million'] = data['population_target']/1000000


# scalling the price and the population as thousands
data['price'] = data['price'] / 1000000
data['population_target'] = data['population_target'] / 1000000


# Creating two separate dataframes for each year
data2007 = data[data.year == 2007].reset_index(drop=True)
data2008 = data[data.year == 2008].reset_index(drop=True)


# Define a function that calculates the distance between the buyer and the target       
def dist_calc(d):
    buyer_loc = (d['buyer_lat'], d['buyer_long'])
    target_loc = (d['target_lat'], d['target_long'])
    return geodesic(buyer_loc, target_loc).miles


# Make a dataset with counterfactual mergers
b_char = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer','corp_owner_buyer']
t_char = ['target_id', 'target_lat', 'target_long', 'price', 'hhi_target', 'population_target']
datasets = [data2007, data2008]
counterfact = [x[b_char].iloc[i].values.tolist() + x[t_char].iloc[j].values.tolist() for x in datasets for i in range(len(x)) for j in range(len(x)) if i != j]
counterfact = pd.DataFrame(counterfact, columns = b_char + t_char)


# Calculat distance between real buyer and real target
data['distance'] = data.apply (lambda d: dist_calc(d),axis = 1)


# Calculate distance between counterfactual buyer and counterfactual target
counterfact['distance'] = counterfact.apply (lambda d: dist_calc(d),axis = 1)


#########################################################
def score1(coeffs):
    
    '''
    This function calculates the payoff functions inside the indication function. If LHS= f(b,t) + f(b',t') > RHS= f(b't) + f(b,t')  holds,
    then the indicator equals 1, 0 otherwise.
    
    '''
    LHS = data['num_stations_buyer'] * data['population_target']+ coeffs[0] * data['corp_owner_buyer'] * data['population_target'] + coeffs[1] * data['distance'] + counterfact['num_stations_buyer'] * counterfact['population_target'] + coeffs[0] * counterfact['corp_owner_buyer'] * counterfact['population_target'] + coeffs[1] * counterfact['distance']
    RHS = counterfact['num_stations_buyer'] * data['population_target'] + coeffs[0] * counterfact['corp_owner_buyer'] * data['population_target'] + coeffs[1] * data['distance'] + data['num_stations_buyer'] * counterfact['population_target'] + coeffs[0] * data['corp_owner_buyer'] * counterfact['population_target'] + coeffs[1] * data['distance']
   
    row_score1=(LHS>RHS)
    neg_total_row_score1 = - row_score1.sum()
    return neg_total_row_score1
bounds1=[(-0.50, 0.50), (-0.75, 0.75)]
#bounds1 = [(.25,.25),(-.25,-.25)]
results1 = differential_evolution(score1, bounds1)

print('Model 1 results')
print(results1.x)


##########################################################

def score2(coeffs):
    
    '''
    This function calculates the payoff functions inside the indication function. If LHS= f(b,t) + f(b',t') > RHS= f(b't) + f(b,t')  holds,
    then the indicator equals 1, 0 otherwise.
    
    '''

 
    LHS = coeffs[0] * data['num_stations_buyer'] * data['population_target'] + coeffs[1] * data['corp_owner_buyer'] * data['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']+ coeffs[0] * counterfact['num_stations_buyer'] * counterfact['population_target'] + coeffs[1] * counterfact['corp_owner_buyer'] * counterfact['population_target'] + coeffs[2]*counterfact['hhi_target'] + coeffs[3] * counterfact['distance']
  
    RHS = coeffs[0] * counterfact['num_stations_buyer'] * data['population_target'] + coeffs[1] * counterfact['corp_owner_buyer'] * data['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']+ coeffs[0] * data['num_stations_buyer'] * counterfact['population_target'] + coeffs[1] * data['corp_owner_buyer'] * counterfact['population_target'] + coeffs[2] * data['hhi_target'] + coeffs[3] * data['distance']
    
   
    row_score2=(LHS>RHS)
    
    neg_total_row_score2 = - row_score2.sum()
    return neg_total_row_score2

bounds2 = [(-.5,.5),(-.5,.5),(-.5,.5),(-.5,.5)]
results2 = differential_evolution(score2, bounds2)
print('Model 2 results')
print(results2.x)

    
    
