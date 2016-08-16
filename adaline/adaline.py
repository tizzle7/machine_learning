# adaline.py

import numpy as np

class Adaline(object):
    """Implementation of an adaptive linear neuron classifier for two-class
    classification.

    Arguments:
    eta -- learning rate between 0 and 1
    n_iterations -- number of iterations over the training dataset
    """
    def __init__(self, eta=0.01, n_iterations=10):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, X, y):
        """Fit the Adaline object to the training data. Creates the
        attributes w_, an array of the weights after the data fitting, and
        cost_, which contains the values of the cost function in each iteration
        which is just the sum of squared errors.

        Arguments:
        X -- training data matrix, each row corresponds to a sample and each collumn
             to a certain feature
        y -- target vector, containing the target values for each sample

        Returns:
        self -- fitted Perceptron object with new attributes
        """
        # initialize the new attributes
        self.w_ = np.zeros(1 + X.shape[1])  # create an extra slot for w0
        self.cost_ = []

        # loop through each iteration step
        for _ in range(self.n_iterations):

            errors = y - self.calculate_net_input(X)
            self.w_[0] += self.eta * errors.sum() # x0 = 1
            self.w_[1:] += self.eta * X.T.dot(errors)

            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)

        return self
                
    def calculate_net_input(self, x):
        """Calculate the value of the net input function for the given data
        using the current weight attribute.

        Arguemts:
        x -- data vector for one sample or matrix containing features of several
             samples
        """
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def calculate_activation(self, x):
        """Calculate the value fo the activation function for the given data.
        In Adaline the activation function is simply equal to the identity function
        of the net input.

        Arguments:
        x -- data vector for one sample or matrix containing features of several
             samples
        """
        return self.calculate_net_input(x)
    
    def predict(self, x):
        """Calculate the perceptron output for given sample data using the
        calculate_net_input method. ercetron object needs to be fitted before
        the predict method can be used to predict class labels of new data.
        
        Arguments:
        x -- data vector for one sample or matrix containing features of several
             samples
        """
        return np.where(self.calculate_activation(x) >= 0, 1, -1)
