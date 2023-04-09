#!/usr/bin/env python
# coding: utf-8

# In[67]:


from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

train = pd.read_csv(r'D:\General\Tech\loan prediction\train.csv')
test = pd.read_csv(r'D:\General\Tech\loan prediction\test.csv')


# In[22]:


train.head()


# In[23]:


train.shape


# In[24]:


train['Loan_Status'].value_counts()
# as data set is not terribly imbalanced, we're moving on with this data


# In[25]:


test.shape


# In[26]:


data = pd.concat([train, test], axis =0)
data


# In[27]:


data.shape


# In[28]:


data.info()


# In[29]:


# drop unwanted coloumns
data.drop('Loan_ID',axis =1, inplace= True)


# In[30]:


data


# In[68]:


train.isnull().sum()


# In[69]:


train.dropna(inplace=True)
train
# data['Married'].dropna(inplace=True)
# data


# In[70]:


train.drop(['Loan_ID'], axis=1, inplace=True)
train


# In[71]:


train.describe(include='all').round(2)


# In[83]:


train['Property_Area'].value_counts()


# In[72]:


train.corr()


# In[73]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


# In[74]:


column=['ApplicantIncome','CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
for col in column:
    sns.boxplot(data= train, x='Loan_Status', y=col)
    plt.show()


# In[84]:


train['Loan_Status']=train['Loan_Status'].map({'Y':1,'N':0})
train


# In[87]:


column=['Dependents','Education','Self_Employed','Credit_History','Property_Area','Gender','Married']
for col in column:
    sns.catplot(x=col, y='Loan_Status', data=train, kind='point', aspect=2,)
    plt.ylim(0,1)


# In[106]:


#train['Dependents']=train['Dependents'].map({'3+':'3'})


# In[117]:


from sklearn.preprocessing import LabelEncoder
column =['Dependents','Property_Area','Gender','Married','Education','Self_Employed']
for col in column:
    le=LabelEncoder()
    train[col]=le.fit_transform(train[col].astype(str))
    
train
    


# In[119]:


train_feature= train.drop(['Loan_Status'],axis=1)
train_labels= train['Loan_Status']


# In[193]:


test = pd.read_csv(r'D:\General\Tech\loan prediction\test.csv')


# In[120]:


train_labels


# In[121]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



# In[122]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        


# In[131]:


# Conduct search for best params while running cross-validation (GridSearchCV)
rf = RandomForestClassifier()
parameters = {
    'n_estimators': [i for i in range(3, 10)],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(train_feature, train_labels.values.ravel())

print_results(cv)


# In[132]:


model_RFC = cv.best_estimator_
model_RFC.fit(train_feature, train_labels.values.ravel())
model_RFC


# In[134]:


lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv2 = GridSearchCV(lr, parameters, cv=5)
cv2.fit(train_feature, train_labels.values.ravel())

print_results(cv2)


# In[135]:


model_LR = cv2.best_estimator_
model_LR.fit(train_feature, train_labels.values.ravel())
model_LR


# In[129]:


from sklearn.tree import DecisionTreeClassifier
grid = {'max_depth': [2, 3, 4, 5],
         'min_samples_split': [2, 3, 4],
         'min_samples_leaf': range(1, 7)}


classifier = DecisionTreeClassifier(random_state = 1234)
gcv = GridSearchCV(estimator = classifier, param_grid = grid, cv=5)
gcv.fit(train_feature, train_labels.values.ravel())

print_results(gcv)


# In[136]:


model_DTC = gcv.best_estimator_
model_DTC.fit(train_feature, train_labels.values.ravel())
model_DTC


# In[139]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time


# In[140]:


def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- \tAccuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                     accuracy,
                                                                                     precision,
                                                                                     recall,
                                                                                     round((end - start)*1000, 1)))


# In[141]:


evaluate_model('RFC', model_RFC, train_feature, train_labels)
evaluate_model('LR', model_LR, train_feature, train_labels)
evaluate_model('DTC', model_DTC, train_feature, train_labels)


# In[208]:


test = pd.read_csv(r'D:\General\Tech\loan prediction\test.csv')
test


# In[209]:


test.shape


# In[210]:


from sklearn.preprocessing import LabelEncoder
column =['Dependents','Property_Area','Gender','Married','Education','Self_Employed']
for col in column:
    le=LabelEncoder()
    test[col]=le.fit_transform(test[col].astype(str))
test.drop(['Loan_ID'], axis=1, inplace=True)
test


# In[211]:


test.values


# In[215]:


len(test.values)


# In[213]:


test.dropna(inplace=True)


# In[214]:


test


# In[236]:


test.drop(['Loan_Status'], axis=1, inplace=True)
test


# In[233]:


pre=[]
for arr in test.values:
    pre.append(model_LR.predict([arr]))
series=pd.Series(pre)


# In[230]:


len(pre)


# In[234]:


test['Loan_Status']=series.values
test


# In[237]:


pre=[]
for arr in test.values:
    pre.append(model_RFC.predict([arr]))
series=pd.Series(pre)


# In[239]:


test['Loan_Status']=series.values
test


# In[ ]:




