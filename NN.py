# Author: Nour Rabih
# Date: 11/04/2022
# This program creates a Neural network to predict student performance

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# load dataset
file = open("errors2.csv")
csvreader = csv.reader(file)
header = next(csvreader)

dataset = pd.read_csv("errors2.csv")

#split dataset in features and target variable
header.remove("grade")
X = dataset[header] # Features
y = dataset.grade # Target variable

normalized_X = preprocessing.normalize(X)

standardized_X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3) # 70% training and 30% test
"""
print(X_train)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"""

mlp= MLPClassifier(activation= 'tanh', alpha= 0.0005, hidden_layer_sizes= (20, 4), learning_rate= 'adaptive', solver= 'sgd')
mlp.fit(X_train, y_train)

y_pred= mlp.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
