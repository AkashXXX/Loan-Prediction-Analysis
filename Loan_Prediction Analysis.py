from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

train = pd.read_csv(r'D:\General\Tech\loan prediction\train.csv')
test = pd.read_csv(r'D:\General\Tech\loan prediction\test.csv')

                                       # DATA EXPLORATION AND PREPARATION

train.shape

train['Loan_Status'].value_counts()  # as data set is not terribly imbalanced, we're moving on with this data
test.shape

data = pd.concat([train, test], axis =0)
data.shape
data.info()

# drop unwanted coloumns
data.drop('Loan_ID',axis =1, inplace= True)

# checking missing values
train.isnull().sum()          

# dropping missing values
train.dropna(inplace=True)
train

train.drop(['Loan_ID'], axis=1, inplace=True)

train.describe(include='all').round(2)
train['Property_Area'].value_counts()

# obtain correlation between features
train.corr()


                                   # Using seaborn and matplotlib for visulisation

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

column=['ApplicantIncome','CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
for col in column:
    sns.boxplot(data= train, x='Loan_Status', y=col)
    plt.show()

                                     # mapping the character values to numbers

train['Loan_Status']=train['Loan_Status'].map({'Y':1,'N':0})

column=['Dependents','Education','Self_Employed','Credit_History','Property_Area','Gender','Married']
for col in column:
    sns.catplot(x=col, y='Loan_Status', data=train, kind='point', aspect=2,)
    plt.ylim(0,1)

from sklearn.preprocessing import LabelEncoder
column =['Dependents','Property_Area','Gender','Married','Education','Self_Employed']
for col in column:
    le=LabelEncoder()
    train[col]=le.fit_transform(train[col].astype(str))

train_feature= train.drop(['Loan_Status'],axis=1)
train_labels= train['Loan_Status']

test = pd.read_csv(r'D:\General\Tech\loan prediction\test.csv')


                                               # MODELLING AND EVALUATION

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        

# Conduct search for best parameters while running cross-validation (GridSearchCV)
# Using RANDOM FOREST CLASSIFIER

rf = RandomForestClassifier()
parameters = {
    'n_estimators': [i for i in range(3, 10)],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(train_feature, train_labels.values.ravel())

print_results(cv)

model_RFC = cv.best_estimator_
model_RFC.fit(train_feature, train_labels.values.ravel())
model_RFC


# USING LOGISTIC REGRESSION
lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv2 = GridSearchCV(lr, parameters, cv=5)
cv2.fit(train_feature, train_labels.values.ravel())

print_results(cv2)

model_LR = cv2.best_estimator_
model_LR.fit(train_feature, train_labels.values.ravel())
model_LR



# USING DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
grid = {'max_depth': [2, 3, 4, 5],
         'min_samples_split': [2, 3, 4],
         'min_samples_leaf': range(1, 7)}

classifier = DecisionTreeClassifier(random_state = 1234)
gcv = GridSearchCV(estimator = classifier, param_grid = grid, cv=5)
gcv.fit(train_feature, train_labels.values.ravel())

print_results(gcv)

model_DTC = gcv.best_estimator_
model_DTC.fit(train_feature, train_labels.values.ravel())
model_DTC



# EVALUATION ALL MODELS
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

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


evaluate_model('RFC', model_RFC, train_feature, train_labels)
evaluate_model('LR', model_LR, train_feature, train_labels)
evaluate_model('DTC', model_DTC, train_feature, train_labels)
