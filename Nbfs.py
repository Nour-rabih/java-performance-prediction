# Author: Nour Rabih
# Date: 11/04/2022
# This program uses Naïve Bayes with feature selection to predict student performance

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import csv
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score



file = open("errors.csv")
csvreader = csv.reader(file)
header = next(csvreader)

errors = pd.read_csv("errors.csv")
errors.head()

#split dataset in features and target variable
header.remove("grade")

X = errors[header] # Features is it wrong to include the header line???
y = errors.grade # Target variable
"""
Feature selection
"""
# define feature selection
fs = SelectKBest(score_func=f_classif, k=5)
# apply feature selection
fs.fit(X, y)
#get the selected header names
cols = fs.get_support(indices=True)
X_selected = X.iloc[:,cols]

new_header = fs.get_feature_names_out(header)

print("selected features: ")
print(new_header)


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3,  random_state=0) # 70% training and 30% test X_selected

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0], (y_test != y_pred).sum()))


cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)

print(cm , ac)

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
