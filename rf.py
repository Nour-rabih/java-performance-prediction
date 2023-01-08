# Author: Nour Rabih
# Date: 11/04/2022
# This program creates random forests to predict student performance

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

# define feature selection
#fs = SelectKBest(score_func=f_classif, k=20)
# apply feature selection
#X_selected = fs.fit_transform(X, y)

"""
creating Random forests with n_estimators with multiples of 10
"""
accuracies = []
ffor i in range(1,21):
    i = i * 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test X_selected

    rf = RandomForestClassifier(bootstrap = True, max_features = 'sqrt', n_estimators=i, random_state=20, max_depth= 6)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracies.append(metrics.accuracy_score(y_test, y_pred))


#Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print("after creating 20 rfs with n_estimators 100 - 2000:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

"""
Cross validation
"""
from sklearn.model_selection import cross_val_score
for i in range(2,12):
    cv_scores = cross_val_score(rf, X, y, cv=i)

print(cv_scores.std())
print(cv_scores.mean())
