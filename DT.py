# Author: Nour Rabih
# Date: 11/04/2022
# This program creates decision trees to predict student performance

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #for accuracy calculation
import statistics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# load dataset
file = open("errors.csv") # change here for data set 1 - errors.csv
csvreader = csv.reader(file)
header = next(csvreader)

dataset = pd.read_csv("errors.csv") # change here for data set 1 - errors.csv

#split dataset in features and target variable
header.remove("grade")
X = dataset[header] # Features
y = dataset.grade # Target variable


accuracies = []
for i in range(100):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="gini", splitter="best", max_depth=5) # or "entropy"

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))
    # Model Accuracy, how often is the classifier correct?
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
confusion matrix, Precision, Recall, and F1 Score of the
last created tree in the for loop
"""

print( "Tree with critereon= gini,  max_depth= 5")
#confusion_matrix
confusionM= confusion_matrix(y_test, y_pred)

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

"""
Mean and std of the accuracy of the 100 trees
"""
print()
# Model Accuracy, how often is the classifier correct?
#Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print("after creating 100 trees with depth 5 and critereon = gini:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))

"""
Entropy
"""
accuracies = []
for i in range(100):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="entropy", splitter="best", max_depth=5)

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))
    # Model Accuracy, how often is the classifier correct?
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
confusion matrix, Precision, Recall, and F1 Score of the
last created tree in the for loop
"""
print()
print( "Tree with critereon= entropy,  max_depth= 5")
#confusion_matrix
confusionM= confusion_matrix(y_test, y_pred)

#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

"""
Mean and std of the accuracy of the 100 trees
"""
# Model Accuracy, how often is the classifier correct?
#Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print("after creating 100 trees with depth 5 and critereon = entropy:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))


"""
Decision trees with different depth levels
"""
accuracies = []
for i in range(2,100):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="gini", splitter="best", max_depth=i) #"gini"

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))

"""
Precision, Recall, and F1 Score of the
last created tree in the for loop (having max_depth of 99)
"""
print("Trying different depth levels")
print()
print("Tree with critereon= gini,  max_depth= 99")
#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

"""
Mean and std of the accuracy of the 100 trees
"""
print()
print("after creating 100 trees with depth from 2-100 and critereon= gini:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))



"""
Entropy

Decision trees with different depth levels
"""
accuracies = []
for i in range(2,100):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    # Create Decision Tree classifer object
    tree = DecisionTreeClassifier( criterion="entropy", splitter="best", max_depth=i)

    # Train Decision Tree Classifer
    tree = tree.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = tree.predict(X_test)
    accuracies.append(metrics.accuracy_score(y_test, y_pred))

"""
Precision, Recall, and F1 Score of the
last created tree in the for loop (having max_depth of 99)
"""
print()
print("Tree with critereon= entropy,  max_depth= 99")
#Precision Score = TP / (FP + TP)
print('Precision: %.3f' % precision_score(y_test, y_pred))

#Recall Score = TP / (FN + TP)
print('Recall: %.3f' % recall_score(y_test, y_pred))

#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

"""
Mean and std of the accuracy of the 100 trees with critereon= entropy
"""
print()
print("after creating 100 trees with depth from 2-100 and critereon= gini:")
print("Mean of the accuracies is % s " %(statistics.mean(accuracies)))
print("Standard Deviation of the accuracies is % s "%(statistics.stdev(accuracies)))



"""
Cross validation
"""

from sklearn.model_selection import cross_val_score
for i in range(2,39):
    cv_scores = cross_val_score(tree, X, y, cv=9)

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
                special_characters=True,class_names=['0','1'],feature_names = header)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DT.png')
Image(graph.create_png())
