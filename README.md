

#packages and libraries
Scikit learn and Graphiz in python 3

This project implements classification methods to predict student performance.


#pre-processing
PMD analysis reports are inputted into the preprocessing code to be processed and ready for the classification methods.

PMD analysis report files:
pmdAnalysis_20_21_1.txt
pmdAnalysis_20_21_2.txt

to run the code just type : python preprocessing.py

the code will write a csv file with the data.

to change the input file just change the file name on line 20.

Output of preprocessing:
errors.csv
errors2.csv


#classification methods

To change the data set: change parameters in lines where the file is opened and the csv is read.

- Decision Tree:
      1.	DT.py
      Tests the performance of the trees on our data set. Considers: Cross-validation, different tree depths and different criteria.

      2. DTfs.py
      Tests the performance of the trees on our data set with feature selection. The trees are all created with ‘gini ‘ criterion.


- Random Forest
      3.	Rf.py
      Tests the performance of random forest. Considers: Cross-validation, different number of trees in the forest

      4.	Rffs.py
      Tests the performance of the random forests on our data set with feature selection.

- Naïve Bayes
      5.	Nb.py
        Implements Naïve bayes

      6. NBfs.py
        Implements Naïve Bayes with feature selection

- Neural network

The hyperparameters can be chosen by ruuning gridSearch.py:
      This program tests the performance of different hyperparameters of the neural network (they are passed as parameters) and returns the best ones.
      Different parameters can to be passed as parameters in lines 25-29.
      7.	NN.py
        Implements Neural network
      8. NNfs.py
        The same Neural network from 6, but with feature selection.
