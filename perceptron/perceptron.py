# perceptron.py

import numpy as np

class Perceptron(object):
    """Implementation of a perceptron classifier for two-class classification.

    Arguments:
    eta -- learning rate between 0 and 1
    n_iterations -- number of iterations over the training dataset
    """
    def __init__(self, eta=0.01, n_iterations=10):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, X, y):
        """Fit the Perceptron object to the training data. Creates the
        attributes w_, an array of the weights after the data fitting, and
        errors_, which lists the number of misclassifications in each iteration.

        Arguments:
        X -- training data matrix, each row corresponds to a sample and each collumn
             to a certain feature
        y -- target vector, containing the target values for each sample

        Returns:
        self -- fitted Perceptron object with new attributes
        """
        # initialize the new attributes
        self.w_ = np.zeros(1 + X.shape[1])  # create an extra slot for w0
        self.errors_ = []

        # loop through each iteration step
        for _ in range(self.n_iterations):
            errors = 0

            for xi, yi in zip(X, y):
                # update the weights using the predict method that calculates
                # the output label of the percepton model
                delta_w = self.eta * (yi - self.predict(xi))
                self.w_[0] += delta_w * 1 # x0 = 1
                self.w_[1:] += delta_w * xi

                # increase the error count if delta_w changes which means that
                # a prediction was wrong
                errors += int(delta_w != 0)

            self.errors_.append(errors)

        return self
                
    def calculate_net_input(self, x):
        """Calculate the value of the net input function for the given data
        using the current weight attribute.

        Arguemts:
        x -- data vector for one sample or matrix containing features of several
             samples
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        """Calculate the perceptron output for given sample data using the
        calculate_net_input method. ercetron object needs to be fitted before
        the predict method can be used to predict class labels of new data.
        
        Arguments:
        x -- data vector for one sample or matrix containing features of several
             samples
        """
        return np.where(self.calculate_net_input(x) >= 0, 1, -1)
