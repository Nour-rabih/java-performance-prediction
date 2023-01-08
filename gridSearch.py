
# Author: Nour Rabih
# Date: 11/04/2022
# This program implements a gridSearch for Neural networks to predict student performance

from sklearn.neural_network import MLPClassifier
import csv
import pandas as pd

# load dataset
file = open("errors.csv")
csvreader = csv.reader(file)
header = next(csvreader)

dataset = pd.read_csv("errors.csv")

#split dataset in features and target variable
header.remove("grade")
X = dataset[header] # Features is it wrong to include the header line???
y = dataset.grade # Target variable

#GridSearchCV
mlp_gs = MLPClassifier(max_iter=3000)
parameter_space = {
    'hidden_layer_sizes': [(240, 4), (250,50), (150,30) ,(20, 4)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd'],
    'alpha': [0.002, 0.005, 0.02, 0.00002, 0.7],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
clf.fit(X, y) # X is train samples and y is the corresponding labels

print('Best parameters found:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
