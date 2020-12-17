import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor
from tqdm import tqdm

# Load the raw CIFAR-10 data.
cifar10_dir = 'spring1819_assignment1\\assignment1\\cs231n\\datasets\\cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds = np.array_split(range(X_train.shape[0]), num_folds, axis=0)
y_train_folds = np.array_split(range(X_train.shape[0]), num_folds, axis=0)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


# ###############################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
# ###############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

k_choices = tqdm(k_choices)
for choosen_k in k_choices:
    k_to_accuracies[choosen_k] = list()
    for fold_ind in range(num_folds):
        ind_x = X_train_folds[fold_ind]
        ind_y = y_train_folds[fold_ind]
        fold_classifier = KNearestNeighbor()
        fold_classifier.train(np.delete(X_train, ind_x, axis=0), np.delete(y_train, ind_y, axis=0))
        fold_dist = fold_classifier.compute_distances_two_loops(X_train[ind_x])
        fold_predict = fold_classifier.predict_labels(fold_dist, choosen_k)
        k_to_accuracies[choosen_k].append((float)(np.sum(fold_predict==y_train[ind_y]))/fold_predict.shape[0])
        print("val=%d  k=%d  acc=%f"%(fold_ind, choosen_k, (float)(np.sum(fold_predict==y_train[ind_y]))/fold_predict.shape[0]))




# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

with open("result.txt") as f:
    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            f.writelines('k = %d, accuracy = %f\n' % (k, accuracy))
            print('k = %d, accuracy = %f' % (k, accuracy))