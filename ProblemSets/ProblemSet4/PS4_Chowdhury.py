import pandas as pd
import numpy as np
from geopy.distance import vincenty
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import minimize, differential_evolution

# Importing data
radio1=(pd.read_excel
        ('C:\\Users\\mmic\\Documents\\Python Tutorial\\Python\\'+
         'radio_merger_data.xlsx'))
# In the above imported data, I have two extra columns:N1 and N2 showing the 
# number of repeatation necessary for each row to get f(b,t) and f(b',t') 
# combinations respectively.
# Importing an index to sort the f(b',t') data
ind=(pd.read_excel
     ('C:\\Users\\mmic\\Documents\\Python Tutorial\\Python\\'+
      'radio_merger_data.xlsx','Sheet2', columns=['Index']))
index=ind['Index']

# I will prepare two dataframes: one with all f(b,t) combinations and other
# with all f(b',t') combinations
# Generating f(b,t) combinations
radio_1=radio1.loc[radio1.index.repeat(radio1.N1)]
radio_1 = radio_1.drop("N1",axis=1).reset_index(drop=True)
radio_1

# Preparation for f(b',t') combinations
# Dropping the (1,1) combination for both markets from radio1 dataframe.
radio_2 = radio1[radio1.buyer_id != 1]

# Generating f(b',t') combinations
radio_2=radio_2.loc[radio_2.index.repeat(radio_2.N2)]
radio_2 = radio_2.drop("N2",axis=1).reset_index(drop=True)
radio_2

# Adding an index to sort the generated combinatons
radio_2 = pd.concat([radio_2, index], axis=1)
radio_3=radio_2.sort_values(by='Index',ascending=True)
radio_4 = radio_3.drop("Index",axis=1).reset_index(drop=True)
# Renaming the variables of radio_4
radio_4.rename(inplace=True, columns=
               {"year":"year_P","buyer_id":"buyer_id_P",
                "target_id":"target_id_P","buyer_lat":"buyer_lat_P",
                "buyer_long":"buyer_long_P","target_lat":"target_lat_P",
                "target_long":"target_long_P","price":"price_P",
                "hhi_target":"hhi_target_P",
                "num_stations_buyer":"num_stations_buyer_P",
                "population_target":"population_target_P",
                "corp_owner_buyer":"corp_owner_buyer_P"})
# Merging the dataframes with two actual pairs.
# Here, the variables of second actual pair (e.g. 2,2) has suffix 'P'

radio = pd.concat([radio_1, radio_4], axis=1)
radio

# Coverting population and price in million unit
m=1000000
radio['price_mil']=radio['price']/m
radio['price_mil_P']=radio['price_P']/m
radio['population_mil']=radio['population_target']/m
radio['population_mil_P']=radio['population_target_P']/m
# Converting population into ln
radio['ln_price']=np.log(radio['price'])/1000
radio['ln_price_P']=np.log(radio['price_P'])/1000
radio['ln_population']=np.log(radio['population_target'])/1000
radio['ln_population_P']=np.log(radio['population_target_P'])/1000

# Creating distance between actual buyer target combinations and counterfactual
# buyer target combinations. Here, I am using the variables adjusted by million.

radio['dist_A1']=(radio.apply(lambda dist: vincenty
                              ((dist['buyer_lat'],dist['buyer_long']),
                               (dist['target_lat'],
                                dist['target_long'])).miles, axis=1))
radio['dist_A2']=(radio.apply(lambda dist: vincenty
                             ((dist['buyer_lat_P'],dist['buyer_long_P']),
                              (dist['target_lat_P'],
                               dist['target_long_P'])).miles, axis=1))
radio['dist_C1']=(radio.apply(lambda dist: vincenty
                             ((dist['buyer_lat'],dist['buyer_long']),
                              (dist['target_lat_P'],
                               dist['target_long_P'])).miles, axis=1))
radio['dist_C2']=(radio.apply(lambda dist: vincenty
                             ((dist['buyer_lat_P'],dist['buyer_long_P']),
                              (dist['target_lat'],
                               dist['target_long'])).miles, axis=1))

# Maximum score function for first payoff model

def msef1(params,data):
    alpha, beta = params
    A = (data['num_stations_buyer']*data['population_mil'] +
         alpha*data['corp_owner_buyer']*data['population_target'] +
         beta*data['dist_A1'])
    B = (data['num_stations_buyer_P']*data['population_mil_P'] +
         alpha*data['corp_owner_buyer_P']*data['population_target_P'] +
         beta*data['dist_A2'])
    C = (data['num_stations_buyer']*data['population_mil_P'] +
         alpha*data['corp_owner_buyer']*data['population_target_P'] +
         beta*data['dist_C1'])
    D = (data['num_stations_buyer_P']*data['population_mil'] +
         alpha*data['corp_owner_buyer_P']*data['population_target'] +
         beta*data['dist_C2'])

    X=A+B
    Y=C+D
    inq = (X>=Y)
    msef1=-inq.sum()/len(inq)

    return msef1

# Maximization of Maximum score function for the first model

b=(0.25,0.25)
model1=opt.minimize(msef1, b, radio, method = 'Nelder-Mead')
print(model1)

# Maximum score function for second payoff model with price

def msef2(params,data):
    delta, alpha, gamma, beta = params
    A = (delta*data['num_stations_buyer']*data['population_mil'] +
         alpha*data['corp_owner_buyer']*data['population_target'] +
         gamma*data['hhi_target'] + beta*data['dist_A1'])
    B = (delta*data['num_stations_buyer_P']*data['population_mil_P'] +
         alpha*data['corp_owner_buyer_P']*data['population_target_P'] +
         gamma*data['hhi_target_P'] + beta*data['dist_A2'])
    C = (delta*data['num_stations_buyer']*data['population_mil_P'] +
         alpha*data['corp_owner_buyer']*data['population_target_P'] +
         gamma*data['hhi_target_P'] + beta*data['dist_C1'])
    D = (delta*data['num_stations_buyer_P']*data['population_mil'] +
         alpha*data['corp_owner_buyer_P']*data['population_target'] +
         gamma*data['hhi_target'] + beta*data['dist_C2'])
    E = data['price_mil'] - data['price_mil_P']
    F = data['price_mil_P'] - data['price_mil']

    X=A-C
    Y=B-D
    inq = (X>=E)&(Y>=F)
    msef2=-inq.sum()/len(inq)

    return msef2

#Maximization of Maximum score funntion for the second payoff model.

y=(1,0.04,0.05,0.05)
model2=opt.minimize(msef2, y, radio, method = 'Nelder-Mead')
print(model2)
