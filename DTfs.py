# Author: Nour Rabih
# Date: 11/04/2022
# This program creates decision trees with feature selection to predict student performance

import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #for accuracy calculation
import statistics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif


# load dataset
file = open("errors.csv")
csvreader = csv.reader(file)
header = next(csvreader)
dataset = pd.read_csv("errors.csv")

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


accuracies = []
for i in range(200):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="gini", splitter="best", max_depth=5)

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))
    # Model Accuracy, how often is the classifier correct?
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#confusion_matrix -useless
confusionM= confusion_matrix(y_test, y_pred)

print( "Tree with critereon= gini,  max_depth= 5 with Feature selection")

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


#Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print("after creating 100 trees with depth 5:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))


"""
Decision trees with different depth levels
"""
accuracies = []
for i in range(1,100):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="gini", splitter="best", max_depth=i)

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))

print()
print("Tree with critereon= gini,  max_depth= 99")

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

print("after creating 100 trees with depth from 2-100:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))

"""
Cross validation
"""
from sklearn.model_selection import cross_val_score
for i in range(2,12):
    cv_scores = cross_val_score(tree, X, y, cv=i)

print()
print("Cross validation")
print(cv_scores.std())
print(cv_scores.mean())

"""
A visual example of the last created tree
"""
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'],feature_names = new_header)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DTfs.png')
Image(graph.create_png())
