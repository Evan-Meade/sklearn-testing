'''
digits_svm.py
Evan Meade, 2019

'''

# External package imports
from sklearn import datasets   # Contains common datasets for analysis
from sklearn.model_selection import train_test_split   # Holdout method
from sklearn import svm   # Contains SVM classes


digits = datasets.load_digits()
x, y = digits.images, digits.target

n = len(x)
x = x.reshape((n, -1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

'''
Creates, trains, and tests an SVM model on the digits data.

The SVM model utilizes a support-vector clustering algorithm, which takes a
few parameters:
- gamma: kernel coefficient
- C: penalty parameter of the error term

Data is fitted to the model using the training set. A hyperplane is defined
which separates points with maximum buffer spacing. Since this is a 10 class
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
print(f"\nyhat: {yhat}\n")
print(f"y:    {y_test}\n")

# Calculates number of successful predictions
success = 0
for i in range(0, len(yhat)):
    if yhat[i] == y_test[i]:
        success += 1

# Prints success ratio of yhat
print(f"success ratio: {success / len(yhat)}\n")
