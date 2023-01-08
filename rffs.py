# Author: Nour Rabih
# Date: 11/04/2022
# This program creates random forests with feature selection to predict student performance

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Load libraries
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
import statistics

# load dataset
file = open("errors.csv")
csvreader = csv.reader(file)
header = next(csvreader)

dataset = pd.read_csv("errors.csv")

dataset.head()

#split dataset in features and target variable
header.remove("grade")

X = dataset[header] # Features is it wrong to include the header line???
y = dataset.grade # Target variable

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


"""
creating Random forests with n_estimators with multiples of 10
"""
accuracies = []
for i in range(1,21):
    i = i * 100
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3) # 70% training and 30% test X_selected

    rf = RandomForestClassifier(bootstrap = True, max_features = 'sqrt', n_estimators=i, random_state=20, max_depth= 3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracies.append(metrics.accuracy_score(y_test, y_pred))


#Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print("after creating 10 rfs with n_estimators 100 - 2000:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
#print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
