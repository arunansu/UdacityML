
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# 
# ## Capstone Project in Finance
# 
# ### Predicting Credit Risk
# 
# #### Introduction
# 
# Many investment banks trade loan backed securities. If a loan defaults, it leads to devaluation of the securitized product. That is why banks use risk models to identify loans at risk and predict loans that might default in near future. Risk models are also used to decide on approving a loan request by a borrower.
# 
# #### The Data
# 
# I will be using data from Lenging Club (https://www.lendingclub.com/info/download-data.action). Lending Club is the world’s largest online marketplace connecting borrowers and investors. 
# 
# The file LoansImputed.csv contains complete loan data for all loans issued through the time period stated. 

# In[1]:

#Import Libraries
import pandas as pd

#Read data file
df = pd.read_csv('LoansImputed.csv')
#Print top 5 rows
print df.head()


# #### Variables in the dataset
# 
# ##### Dependent Variable
# 
# * not.fully.paid: A binary variable. 1 means borrower defaulted and 0 means monthly payments are made on time
# 
# ##### Independent Variables
# 
# * credit.policy: 1 if borrower meets credit underwriting criteria and 0 otherwise
# * purpose: The reason for the loan
# * int.rate: Interest rate for the loan (14% is stored as 0.14)
# * installment: Monthly payment to be made for the loan
# * log.annual.inc: Natural log of self reported annual income of the borrower
# * dti: Debt to Income ratio of the borrower
# * fico: FICO credit score of the borrower
# * days.with.cr.line: Number of days borrower has had credit line
# * revol.bal: The borrower's rovolving balance (Principal loan amount still remaining)
# * revol.util: Amount of credit line utilized by borrower as percentage of total available credit
# * inq.last.6mths: Borrowers credit inquiry in last 6 months
# * delinq.2yrs: Number of times borrower was deliquent in last 2 years
# * pub.rec: Number of derogatory pulic record borrower has (Bankruptcy, tax liens and judgements etc.)

# In[2]:

# Convert purpose to category
df['purpose'] = pd.Categorical.from_array(df['purpose']).codes
print "purpose converted to factors"
print df.head()

#extract dependent variable as label
Y = df['not.fully.paid']
X = df.drop('not.fully.paid', 1)
print "not.fully.paid as label"
print Y.head()
print "not.fully.paid removed from features"
print X.head()


# In[6]:

#Select only important features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print "Shape of X: "
print X.shape
X_imp = SelectKBest(chi2, k=4).fit_transform(X, Y)
print "New shape of X"
print X_imp.shape


# In[7]:

#Shuffle and split data into 70% in training and 30% in testing
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_imp, Y, test_size=0.3, random_state=42)


# In[10]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

#Model using Logistic Regression
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)
print "Model Accuracy:"
print logreg.score(X_train, Y_train)

#Predict using the logistic regression model
Y_pred = logreg.predict(X_test)
#probability of the predicted labels
Y_prob = logreg.predict_proba(X_test)

#accuracy score
print metrics.accuracy_score(Y_test, Y_pred)
#auc score
print metrics.roc_auc_score(Y_test, Y_prob[:,1])


# #### Conclusion
# Our model has an accuracy of 70%. This model can be used to predict loans that will be at risk. This can be used to decide whether to approve a loan or not and proactive action can be taken on existing loans that are likely to default.
