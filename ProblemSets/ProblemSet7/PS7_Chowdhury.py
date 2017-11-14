# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:11:36 2017

@author: mmic
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pandas_datareader import wb

## Searching variables with the key word CO2

dat=wb.search('CO2').iloc[:,:2]
print(dat)
 
# Importing Data from World Bank API
var = ['EN.ATM.GHGT.KT.CE','EN.CO2.TRAN.ZS','EN.CO2.BLDG.ZS',
       'EN.CO2.MANF.ZS','EN.CO2.OTHX.ZS','EN.CO2.ETOT.ZS']
con = ['WLD','USA','CHN','IND','JPN','RUS']
st=1960
en=2016
df= wb.download(indicator=var, country=con, start=st, end=en).dropna()
# Renaming the variable names
df=df.rename(columns={'EN.ATM.GHGT.KT.CE':'TE'})

# Graph 1

df1=df.unstack(level=0)
plt.style.use('ggplot')
np.log(df1).plot(y='TE', kind='line',color = ('b', 'g', 'r', 'c', 'm', 'B', 'k'),
       style=['-', '--', '-.',':','--',':','-.','-.'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0); 
plt.title("Fig.1: Greenhouse Gas (CO2) Emission Trend"); 
plt.xlabel('Year'); plt.ylabel('ln(Emission)');
plt.savefig('Graph-A.png', transparent=False, dpi=100, bbox_inches="tight")

# Graph 2

## Data preparation

# Subsampling for Total Emission Variable
df2=df1['TE'].dropna()

# Creating new variables
df2['Top Emitter']=(df2['China']+df2['Japan']+df2['India']+
                    df2['United States']+df2['Russian Federation'])
df2['Rest World']=df2['World']-df2['Top Emitter']
df2['Share of Top Five Greenhouse Gas Emitting Countries']=(df2['Top Emitter']/
                                                            df2['World'])
df2['Share of Rest of the World']=df2['Rest World']/df2['World']

## Graph

Y=['Share of Top Five Greenhouse Gas Emitting Countries', 'Share of Rest of the World']
df2.plot(y=Y, kind='bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0); 
plt.title("Fig.2: Comparative Contribution to Total Emission by \n the Top Five Emitting Countries  and Rest of the World"); 
plt.xlabel('Year'); plt.ylabel('Contribution (Share)');
plt.savefig('Graph-B.png', transparent=False, dpi=100, bbox_inches="tight")

## Graph 3

## Data Preparation

# Sub-sample for World
df3=df.loc['World']
# Sorting the data by year
df3= df3.iloc[::-1]
# Renaming the variables
df3=df3.rename(columns={'EN.CO2.TRAN.ZS':'Transport',
                        'EN.CO2.BLDG.ZS':'Residential building, commercial and public service',
                        'EN.CO2.MANF.ZS':'Manufacturing and Construction',
                        'EN.CO2.OTHX.ZS':'Other Service',
                        'EN.CO2.ETOT.ZS':'Electricity and Heat Production'})
## Graph

Y=['Transport', 'Residential building, commercial and public service', 
     'Manufacturing and Construction', 'Other Service', 
     'Electricity and Heat Production']
df3.plot(y=Y, kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0); 
plt.title("Fig.3: Contribution to Total Emission by Sector"); 
plt.xlabel('Year'); plt.ylabel('Contribution');
plt.savefig('Graph-C.png', transparent=False, dpi=120, bbox_inches="tight")