'''
iris_svm.py
Evan Meade, 2019

Simple script for classifying irises with support vector machines.

Classifying irises is a classic data science problem which requires a model
to operate on a hyperdimensional set of features and produce a non-binary
classification output.

Support vector machines (SVMs) are models which calculate parameters for the
hyperplane which optimally separates datapoints by maximizing distance from
any point to the plane. Then, depending on which side of the plane a datapoint
lies, it is deterministically given a binary classification. This methodology
is extended to higher numbers of classes by generating multiple SVMs, one for
each pair of classes in this case, and voting to find the final classification.

For a dataset like this, with a high number of samples relative to the number
of dimensions, SVM works well. It is also a low enough number of dimensions
to be computationally feasible, since complexity scales quadratically with
dimensionality.

Performance is very good, with success ratios of 1.0 occurring regularly upon
cross-verification with test datasets.

Based on methods outlined in a scikit-learn tutorial:
https://scikit-learn.org/stable/tutorial/basic/tutorial.html

'''

# External package imports
from sklearn import datasets   # Contains common datasets for analysis
from sklearn.model_selection import train_test_split   # Holdout method
from sklearn import svm   # Contains SVM classes


'''
Loads data from the iris classification dataset.

Each iris belongs to one of 3 varieties, given as the target vector, y. Each
datapoint has 4 features:
    [sepal length, sepal width, petal length, petal width]
'''
iris = datasets.load_iris()
x, y = iris.data, iris.target

'''
This function implements the holdout method for cross-validation, where the
full dataset is separated at the beginning into training and testing subsets.
It is trained and evaluated on the corresponding subsets.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

'''
Creates, trains, and tests an SVM model on the iris data.

The SVM model utilizes a support-vector clustering algorithm, which takes a
few parameters:
- gamma: kernel coefficient
- C: penalty parameter of the error term

Data is fitted to the model using the training set. A hyperplane is defined
which separates points with maximum buffer spacing. Since this is a 3 class
target space, a multiclass method must be used: one vs. one approach. Here,
an SVM is trained for each pair of classes, and a voting procedure is used
to combine results.

Predictions are made on the test set of features by finding their placement
in the hyperspace relative to the hyperplanes of the SVM models.
'''
clf = svm.SVC(gamma=.001, C=100)
clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

# Prints predicted and actual classifications
print(f"yhat: {yhat}")
print(f"y:    {y_test}")

# Calculates number of successful predictions
success = 0
for i in range(0, len(yhat)):
    if yhat[i] == y_test[i]:
        success += 1

# Prints success ratio of yhat
print(f"success ratio: {success / len(yhat)}")
