'''
digits_svm.py
Evan Meade, 2019

Simple script for classifying handwritten digits using SVMs.

The dataset used here features 8x8 grayscale images of handwritten digits
complete with actual classifications as digits. As such, it is a very
elementary example of how models can be constructed to process images.

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

Based on code outlined in a scikit-learn tutorial:
https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

'''

# External package imports
from sklearn import datasets   # Contains common datasets for analysis
from sklearn.model_selection import train_test_split   # Holdout method
from sklearn import svm   # Contains SVM classes

# Loads digits data
digits = datasets.load_digits()
# Separates digits data into images and classifications
x, y = digits.images, digits.target

'''
Because SVMs process linear feature vectors but the images are 2D arrays, they
must be "flattened" to a 1D array. This can be achieved using the reshape
method for NumPy arrays, which rearranges values with the tuple dimensions
given. Thus, the resulting x has 2 dimensions: image number, and image data.
'''
n = len(x)
x = x.reshape((n, -1))   # The -1 entry automatically calculates a new axis

'''
This function implements the holdout method for cross-validation, where the
full dataset is separated at the beginning into training and testing subsets.
It is trained and evaluated on the corresponding subsets.
'''
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
