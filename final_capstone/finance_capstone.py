
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
print df.columns.values


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
# * annualincome: Annual income of borrpwer

# In[2]:

#Print statistics of variables
print df.describe()
print df[df['not.fully.paid'] == 1].describe()
print df[df['not.fully.paid'] == 0].describe()
print "Number of loans that have 50% credit utilization and defaulted: "
print len(df[(df['revol.util'] > 50.00) & (df['not.fully.paid'] == 1)])
print "Number of loans that have 50% credit utilization and not defaulted: "
print len(df[(df['revol.util'] > 50.00) & (df['not.fully.paid'] == 0)])


# In[3]:

#plot correlation between each feature
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()


# In[6]:

from sklearn import preprocessing
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder

#Print unique values of purpose
print pd.Series.unique(df['purpose'])
# Convert purpose to category
df['purpose'] = pd.Categorical.from_array(df['purpose']).codes
print "purpose converted to factors"
print df.head()

#extract dependent variable as label
Y = df['not.fully.paid']
#Drop dependent variable and categorical variable
X = df.drop('not.fully.paid', 1)

print "not.fully.paid as label"
print Y.head()
print "not.fully.paid removed from features"
print X.head()

#One Hot Encode purpose
enc = OneHotEncoder()
df['purpose'] = enc.fit(df['purpose'])

#scale dependent variable
X = preprocessing.scale(X)

#print first row of X
print X[0]
print X[0].mean(axis=0)
print X[0].std(axis=0)


# In[7]:

#Shuffle and split data into 70% in training and 30% in testing
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print "X_train size = {0}".format(len(X_train))
print "Y_train size = {0}".format(len(Y_train))
print "X_test size = {0}".format(len(X_test))
print "Y_test size = {0}".format(len(Y_test))


# In[8]:

#Using Logistic Regression
from sklearn import metrics
from sklearn import linear_model
import time

# fit an Extra Trees model to the data
logreg = linear_model.LogisticRegression()
startTrain = time.time()
logreg.fit(X_train, Y_train)
endTrain = time.time()

print "LogisticRegression training time (secs): {0}".format(endTrain - startTrain)
print "LogisticRegression accuracy in training set: {0}".format(logreg.score(X_train, Y_train))

#Predict using the logistic regression model
startTest = time.time()
Y_pred = logreg.predict(X_test)
endTest = time.time()

print "LogisticRegression prediction time (secs): {0}".format(endTest - startTest)
print "LogisticRegression accuracy in testing set: {0}".format(metrics.accuracy_score(Y_test, Y_pred))
print "Logistic Regression F1 Score: {0}".format(metrics.f1_score(Y_test, Y_pred))


# In[9]:

#Using Support Vector Machine
from sklearn import metrics
from sklearn import svm
import time

# fit an Extra Trees model to the data
clm = svm.SVC(probability=True)
startTrain = time.time()
clm.fit(X_train, Y_train)
endTrain = time.time()

print "SVM training time (secs): {0}".format(endTrain - startTrain)
print "SVM accuracy in training set: {0}".format(clm.score(X_train, Y_train))

#Predict using the logistic regression model
startTest = time.time()
Y_pred = clm.predict(X_test)
endTest = time.time()

print "SVM prediction time (secs): {0}".format(endTest - startTest)
#accuracy score
print "SVM accuracy in testing set: {0}".format(metrics.accuracy_score(Y_test, Y_pred))
#auc score
print "SVM F1 score in testing set: {0}".format(metrics.f1_score(Y_test, Y_pred))


# In[10]:

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import time

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
startTrain = time.time()
model.fit(X_train, Y_train)
endTrain = time.time()

print "ExtraTreeClassifier training time (secs): {0}".format(endTrain - startTrain)
print "ExtraTreeClassifier accuracy in training set: {0}".format(model.score(X_train, Y_train))

#Predict using the logistic regression model
startTest = time.time()
Y_pred = model.predict(X_test)
endTest = time.time()

print "ExtraTreeClassifier prediction time (secs): {0}".format(endTest - startTest)
print "ExtraTreeClassifier accuracy in testing set: {0}".format(metrics.accuracy_score(Y_test, Y_pred))
print "ExtraTreeClassifier F1 score in testing set: {0}".format(metrics.f1_score(Y_test, Y_pred))


# In[14]:

from sklearn import grid_search
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import ExtraTreesClassifier

parameters = {'n_estimators': [1, 32]}
model = ExtraTreesClassifier()
f1_scorer = make_scorer(f1_score, pos_label='yes')
clf = grid_search.GridSearchCV(model, param_grid=parameters, scoring=f1_scorer)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print model.get_params()
print "F1 score for test set: {}".format(metrics.f1_score(Y_test, Y_pred))


# In[12]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
features = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 
            'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
            'delinq.2yrs', 'pub.rec', 'annualincome']


# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlim([-1, X.shape[1]])
plt.show()


# In[11]:

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('ROC Curve')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### Conclusion
# The final model has accuracy score of 78% and F1 score 0.53, which is more than the rough estimate of accuracy score 54.59% and F1 score 0.44 

# In[ ]:



