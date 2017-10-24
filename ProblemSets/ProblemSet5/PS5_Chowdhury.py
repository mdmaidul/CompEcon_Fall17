# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:31:41 2017

@author: mmic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the excel file into python with necessary rows and columns

r=pd.read_excel('C:\\Users\\mmic\\Documents\\Python Tutorial\\Python\\'+
                'Bangladesh_Remittance_HIES2010.xlsx',sheetname="Sheet1",
                index_col=None,na_values=['NA'],skiprows=7,
                parse_cols="A,F:G,J:M,P,U")

## Data cleaning

r['Total_Remittance']=r['Total_money_sent']+r['Value of product sent']
# Converting total remittance in USD
# US-BDT exchange rate in 2010
E=71.17
r['Total_Remittance']=r['Total_Remittance']/E
# Converting total stay in abroad in months
r['Total_Stay']=r['Months spent in abroad']+12*r['Years Spent in abroad']
r=r[r.Total_Remittance !=0]
# Deleting the expatriates' information reported as living in Bangladesh 
#(0 is the coutry code for Banglades)
r=r[r.Country_Code !=0]

# Recoding the country code into country name
r['Country'] = (r['Country_Code'].map({1:'Saudi Arabia',2:'Qatar',3:'Kuwait',
 4:'Oman',5:'Malaysia',6:'Singapore',7:'Iraq',8:'Iran',9:'Libya',10:'UAE',
 11:'Canada',12:'Australia',13:'UK',14:'USA',15:'South Korea',16:'Japan',
 17:'Turkey',18:'Germany',19:'Sweden',20:'Russia',21:'Other European Countries',
 23:'Brunei',24:'Mauritius',25:'South Africa',26:'Others'}))

## Graph 1
plt.style.use('ggplot')
fig1, ax = plt.subplots()
group_names = ['0 year', '1-5 years','6-10 years','11-12 years','13-16 years',
               '>16 years']
r['categories'] = pd.cut(r['Level of Education'], bins=[-1,0,5,10,12,16,18], 
                         labels=['0 year', '1-5 years','6-10 years',
                                 '11-12 years','13-16 years', '>16 years'])
counts = r['categories'].value_counts(sort=False)
plt.axis('equal')
explode = (0.2, 0.1,0.1,0.2,0.1,0.1)
colors = ['#c0d6e4','#6a6aa7','#40e0d0','#ee6363','#0071C6','#008DB8',]
counts.plot(kind='pie', fontsize=12,colors=colors,explode=explode,autopct='%.2f')
plt.legend(labels=group_names,loc=2,bbox_to_anchor=(0.8,0.4))
plt.ylabel('')
plt.title('Graph-1: Level of Education of the expatriates (In Percentage)')
# save graph 1
fig1.savefig('Graph-1.png', transparent=False, dpi=90, bbox_inches="tight")

# Graph 2
fig2, ax = plt.subplots()
r.groupby('Country')['Total_Remittance'].sum().plot(kind='bar')
plt.ylabel('Remittance Amount (USD)')
plt.title('Graph-2: Remittance Inflow by the Location of Expatriates')
# save graph 2
fig2.savefig('Graph-2.png', transparent=False, dpi=90, bbox_inches="tight")

# Graph 3
fig3, ax = plt.subplots()
plt.scatter(r['Level of Education'], np.log(r['Total_Remittance']), alpha=0.15,
            marker='o')
plt.plot(np.unique(r['Level of Education']),
         np.poly1d(np.polyfit(r['Level of Education'],
                              np.log(r['Total_Remittance']), 1))
         (np.unique(r['Level of Education'])),color='Black', 
         linestyle="--", linewidth=2)
plt.scatter(r['Level of Education'], np.log(r['Total_Remittance']), 
            alpha=0.15, marker='o')
plt.ylabel('Remittance (Log)')
plt.xlabel('Level of Education (Years)')
plt.title('Graph-3: Remittance and Level of Education')
# save graph 3
fig3.savefig('Graph-3.png', transparent=False, dpi=90, bbox_inches="tight")