#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/User/Documents/Data Analysis Project Datasets/Campaign-Data.csv")


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


df['Calendardate']=pd.to_datetime(df['Calendardate'])
df['Calendar_Month']=df['Calendardate'].dt.month
df['Calendar_Year']=df['Calendardate'].dt.year


# In[6]:


df['Client Type'].value_counts(normalize=True)


# In[7]:


pd.crosstab(df['Number of Competition'],df['Client Type'],margins=True,normalize='columns')


# In[8]:


df.groupby('Number of Competition').mean()


# In[9]:


df.groupby('Client Type').mean()


# In[10]:


import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)
correlation_analysis=pd.DataFrame(df[['Amount Collected',
'Campaign (Email)', 'Campaign (Flyer)', 'Campaign (Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5']].corr()['Amount Collected']).reset_index()
correlation_analysis.columns=['Impacting Variable','Degree of Linear Impact (Correlation)']
correlation_analysis=correlation_analysis[correlation_analysis['Impacting Variable']!='Amount Collected']
correlation_analysis=correlation_analysis.sort_values('Degree of Linear Impact (Correlation)',ascending=False)
correlation_analysis.style.background_gradient(cmap=cm).set_precision(2)


# In[11]:


import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)
correlation_analysis=pd.DataFrame(df.groupby('Client Type')[['Amount Collected',
       'Campaign (Email)', 'Campaign (Flyer)', 'Campaign (Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5']].corr()['Amount Collected']).reset_index()
correlation_analysis=correlation_analysis.sort_values(['Client Type','Amount Collected'],ascending=False)
correlation_analysis.columns=['Acc Type','Variable Impact on Sales','Impact']
correlation_analysis=correlation_analysis[correlation_analysis['Variable Impact on Sales']!='Amount Collected'].reset_index(drop=True)
correlation_analysis.style.background_gradient(cmap=cm).set_precision(2)


# In[12]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
df.columns=[mystring.replace(" ", "_") for mystring in df.columns]
df.columns=[mystring.replace("(", "") for mystring in df.columns]
df.columns=[mystring.replace(")", "") for mystring in df.columns]
results = smf.ols('Amount_Collected ~ Campaign_Email+Campaign_Flyer+Campaign_Phone+\
       Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3+Sales_Contact_4 + Sales_Contact_5',data=df).fit()
print(results.summary())


# In[13]:


dt = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]


# In[14]:


dt=dt.reset_index()
dt=dt[dt['P>|t|']<0.05][['index','coef']]
dt


# In[15]:


consolidated_summary=pd.DataFrame()
for acctype in list(set(list(df['Client_Type']))):
    temp_data=df[df['Client_Type']==acctype].copy()
    results = smf.ols('Amount_Collected ~ Campaign_Email+Campaign_Flyer+Campaign_Phone+\
       Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3+Sales_Contact_4 + Sales_Contact_5', data=temp_data).fit()
    dt = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0].reset_index()
    dt=dt[dt['P>|t|']<0.05][['index','coef']]
    dt.columns=['Variable','Coefficent (Impact)']
    dt['Account Type']=acctype
    dt=dt.sort_values('Coefficent (Impact)',ascending=False)
    dt=dt[dt['Variable']!='Intercept']
    print(acctype)
    consolidated_summary=consolidated_summary.append(dt)
    print(dt)
    #print(results.summary())


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
consolidated_summary=pd.DataFrame()
for acctype in list(set(list(df['Client_Type']))):
    print(acctype)
    temp_data=df[df['Client_Type']==acctype].copy()
    results = smf.ols('Amount_Collected ~ Campaign_Email+Campaign_Flyer+Campaign_Phone+\
       Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3+Sales_Contact_4 + Sales_Contact_5', data=temp_data).fit()
    dt = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0].reset_index()
    dt=dt[dt['P>|t|']<0.05][['index','coef']]
    dt.columns=['Variable','Coefficent (Impact)']
    dt['Account Type']=acctype
    dt=dt.sort_values('Coefficent (Impact)',ascending=False)
    dt=dt[dt['Variable']!='Intercept']
    consolidated_summary=consolidated_summary.append(dt)
    print(results.summary())


# In[17]:


consolidated_summary


# In[18]:


consolidated_summary.reset_index(inplace=True)
consolidated_summary.drop('index',inplace=True,axis=1)


# In[19]:


consolidated_summary.columns = ['Variable','Return on Investment','Account Type']
consolidated_summary['Return on Investment']= consolidated_summary['Return on Investment'].apply(lambda x: round(x,1))
consolidated_summary.style.background_gradient(cmap='RdYlGn')


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[21]:


def format(x):
        return "${:.1f}".format(x)
consolidated_summary['Return on Investment']  = consolidated_summary['Return on Investment'].apply(format)


# In[22]:


consolidated_summary.columns = ['Variable','Return on Investment','Account Type']
consolidated_summary.style.background_gradient(cmap='RdYlGn')


# In[ ]:





# In[ ]:




