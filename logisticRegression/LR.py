#coding:utf8
'''
Created on 2016-08-23

@author: 71071
'''
'''
the program is a module of logistic regression.
Recorded in order to learn and practice.

Through the program,
First we can learn how to choose features,
second how to train logistic regression,
third how to test logistic regression,
finally to effect evaluation including the rate of Recall and Precision. 
'''

'''
step1.Data Discovery. 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#we need to install a seaborn module to plot variable distribution.
import seaborn as sns 

data=pd.read_excel(r'') #we need to install a xlrd module to use read_excel.

#to get overall situation.
print data.describe()

#to plot variable distribution.
sns.distplot(data[u''].dropna())

plt.show()

#to get measures of dispersion.
data1=data[u''].dropna()

plt.boxplot(data1)
plt.show()

'''
step2.Feature Selection.
we can get the better feature in the following methods. 
1.model coefficient significantly.
2.choose stability variables.

eg.No2
'''
x=data.iloc[:,:8].as_matrix()
y=data.iloc[:,8].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

#to create random logistic regression model.
rlr=RLR(selection_threshold=0.25)

#to train RLR model by training data.
rlr.fit(x,y)

#to get supporting features.
rlr.get_support()

print data.columns[rlr.get_support()]

'''
step3.Fit Logistic Regression Model.
'''
x=data[data.columns[rlr.get_support()]].as_matrix()

#to create logistic regression model.
lr=LR()

#to train LR model.
lr.fit(x,y)

#to get mean precision.
print lr.score(x, y)

'''
step4.to predict use the test set.
'''
te=pd.read_excel(r'')

xte=te.iloc[:,:8].as_matrix()
yte=te.iloc[:,8].as_matrix()

xte1=xte[xte.columns[rlr.get_support()]].as_matrix()

#to get result of predict.
pte=lr.predict(xte1)

'''
step4.To effect evaluation.
'''
from sklearn.metrics import confusion_matrix

confusion_matrix=confusion_matrix(yte, pte)

plt.matshow(confusion_matrix)
plt.ylabel(u'true')
plt.xlabel(u'pred')

#to plot 2*2 matrix.
plt.show()

#to get the rate of Recall and Precision.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score

#to get the precision.the parameter of cv is iterative number.
precisions=cross_val_score(lr,xte1,yte,cv=5,scoring='precision')

print np.mean(precisions),precisions

#to get the recalls.the parameter of cv is iterative number.
recalls=cross_val_score(lr,xte1,yte,cv=5,scoring='recall')

print np.mean(recalls),recalls

#to get the F1 measure.the parameter of cv is iterative number.
f1s=cross_val_score(lr,xte1,yte,cv=5)

print np.mean(f1s),f1s

#to plot curve of roc.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_predict
from sklearn.metrics import roc_curve,auc

false_positive_rate,recall,thresholds=roc_curve(yte,pte)
roc_auc=auc(false_positive_rate,recall)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,recall,'b',label='AUC=%4.2f'%roc_auc)
plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()










