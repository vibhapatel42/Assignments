#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Set 1 Q.1


# In[6]:


import numpy as np
import math
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pylab as py
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import norm


# In[4]:


# Uploading dataset
df=pd.read_csv("D:/Work/Data Science and Analyst Course/ExcelR/Data Science/Assignments/2_Basic Statistics Level 2/Dataset1.csv")
df


# In[26]:


#calculating mean, standard deviation and variance
df1=df['Measure X']
df2=df1.iloc[0:15]*100
df2
df3=list(df['Name of company'])
df3
#df3.loc(:,1)
#


# In[7]:


df3.loc(:,0:14)


# In[ ]:





# In[ ]:





# In[27]:


df2.mean()


# In[28]:


df2.var()


# In[29]:


df2.std()


# In[30]:


plt.boxplot(df2,vert = 0)


# In[ ]:





# In[33]:


df2


# In[34]:


df3


# In[36]:


fig = plt.figure(figsize =(10, 7))
explode = (0, 0, 0, 0,0,0,0,0,0,0,0.1,0,0,0,0)
my_labels=df3
#["Allied Signal","Bankers Trust","General Mills","ITT Industries","J.P.Morgan & Co.","Lehman Brothers","Marriott","MCI","Merrill Lynch","Microsoft","Morgan Stanley","Sun Microsystems","Travelers","US Airways","Warner-Lambert"]

plt.pie(df2, explode=explode,labels=my_labels, autopct='%1.1f%%')
plt.show() 


# In[47]:





# In[28]:


list(df.columns)


# In[46]:


q1=df2.quantile(0.25)

q3=df2.quantile(0.75)

IQR=q3-q1

outliers = df2[((df2<(q1-1.5*IQR)) | (df2>(q3+1.5*IQR)))]
outliers


# In[1]:


#Set 2 Q.1
#1.	The time required for servicing transmissions is normally distributed with  = 45 minutes and  = 8 minutes. 
#The service manager plans to have work begin on the transmission of a customer’s car 10 minutes after the car is dropped off 
#and the customer is told that the car will be ready within 1 hour from drop-off. 
#What is the probability that the service manager cannot meet his commitment? 


# In[2]:


#let us first find z-score for mean 40 and standard deviation 8


# In[4]:


z=(50-45)/8
z


# In[ ]:


# here, we need to find probability service manager will take time greater than 50 minutes.
#P(X>50) = 1-stats.norm.cdf(abs(z_score))


# In[11]:


P=1-stats.norm.cdf(abs(z))
P


# In[13]:


#Set 2 Q.2
#2.	The current age (in years) of 400 clerical employees at an insurance claims processing center is normally distributed 
#with mean = 38 and Standard deviation =6


# In[12]:


#A.	More employees at the processing center are older than 44 than between 38 and 44.


# In[15]:


#P(x>44); Employees are older than 44
P1=1-stats.norm.cdf(44,loc=38,scale=6)
P1


# In[16]:


# Probability between 38 and 44
P2=stats.norm.cdf(44,loc=38,scale=6)-stats.norm.cdf(38,loc=38,scale=6)
P2


# In[ ]:


# B. A training program for employees under the age of 30 at the center would be expected to attract about 36 employees.


# In[17]:


# P(x<30)
P3=stats.norm.cdf(30,loc=38,scale=6)
P3


# In[18]:


#No. of employees under the age of 30 attending training program= Total employee* probability of emplyee under age 30


# In[20]:


N1=400*P3
N1


# In[ ]:


#Set 2 Q.4
#4.	Let X ~ N(100, 202). Find two values, a and b, symmetric about the mean, such that the probability of the random 
#variable taking a value between them is 0.99. 


# In[22]:


stats.norm.ppf(.005)


# In[6]:


#Set 2 Q.5
#Profit1 ~ N(5, 32) and Profit2 ~ N(7, 42) respectively. Both the profits are in $ Million. Answer the following questions about the total profit of the company in Rupees. 
#Assume that $1 = Rs. 45

#A. A.	Specify a Rupee range (centered on the mean) such that it contains 95% probability
#for the annual profit of the company.


# In[ ]:


# Mean profits from both division is mean1+mean2


# In[10]:


total_mean=5+7
meanprofit=total_mean*45
print('Mean Profit in Rs. is',meanprofit,'million')


# In[12]:


# Variance of profits from two different divisions of a company = SD^2 = SD1^2 + SD2^2
SD = np.sqrt((9)+(16))
sdinmillion=SD*45
print('Standard Deviation is Rs', sdinmillion, 'million')


# In[14]:


#A. Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company
Range=stats.norm.interval(0.95,540,225)
Range


# In[ ]:


# B.Specify the 5th percentile of profit (in Rupees) for the company


# In[16]:


#To compute 5th Percentile, we use the formula X=μ + Zσ; where z is
z=stats.norm.ppf(.05)
z


# In[17]:


x=540+z*225
x


# In[ ]:


#C. Which of the two divisions has a larger probability of making a loss in a given year?


# In[18]:


# Probability that division 1 is making loss P(x<0)
stats.norm.cdf(0,5,3)


# In[19]:


# Probability that division 2 is making loss P(x<0)
stats.norm.cdf(0,7,4)


# In[ ]:


#Set 3 Q.5
#H0=Mozilla has a higher share than 5%
#Let's find p-value and compared it with alpha=0.05
#given information are 


# In[3]:


z=(0.046-0.05)/(np.sqrt((0.05*(1-0.05))/2000))
z


# In[4]:


p=1-stats.norm.cdf(abs(z))
p


# In[ ]:


# Set 3 Q.8
#For 95% confidence interva, alpha=0.05
#p-value is 1-0.05/2


# In[5]:


p=1-0.05/2
p


# In[8]:


z=stats.norm.ppf(p)
z


# In[ ]:


#Set 4 Q.3
#Given Data: n=100, Population mean=50, Population SD=40, as sample size n>30 we can consider normal distribution
#let us find Z-scores for interval between 45 and 55
#Z-score=(sample mean-Popultaon mean)/(population SD/sqrt(n))
#z-score at sample mean 45


# In[8]:


z_45=(45-50)/(40/math.sqrt(100))
z_45


# In[9]:


#z-score at sample mean 55
z_55=(55-50)/(40/math.sqrt(100))
z_55


# In[14]:


#for No investigation P(45)
x=stats.norm.cdf(z_55)-stats.norm.cdf(z_45)


# In[15]:


# For Investigation 1-P(45)
1-x


# In[ ]:


#to maintain the probability of investigation to 5%, alpha=0.05, for that z-score is +/-1.96
#Z-score=(sample mean-Popultaon mean)/(population SD/sqrt(n)), need to find minimum number of transactions
#z-score for alpha=0.05 is 1.96


# In[20]:


sqrt_n=(1.96*40)/5
sqrt_n


# In[22]:


n=sqrt_n**2
n

