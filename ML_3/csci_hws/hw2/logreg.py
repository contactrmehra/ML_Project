import random
import argparse
import gzip
import pickle
import math
import numpy as np
from math import exp, log
from collections import defaultdict
from matplotlib import pyplot as py

SEED = 1735

random.seed(SEED)


class Numbers:
    """
    Class to store MNIST data for images of 0 and 1 only
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like

        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8
        #self.train_x = np.insert(self.train_x, 0, values=1, axis=1)

        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8
        #self.valid_x = np.insert(self.valid_x, 0, values=1, axis=1)

        self.test_x, self.test_y = test_set
        test_indices = np.where(self.test_y > 7)
        self.test_x, self.test_y = self.test_x[test_indices], self.test_y[test_indices]
        self.test_y = self.test_y - 8
        #self.test_x = np.insert(self.test_x, 0, values=1, axis=1)

    @staticmethod
    def shuffle(X, y):
        """ Shuffle training data """
        shuffled_indices = np.random.permutation(len(y))
        return X[shuffled_indices], y[shuffled_indices]


class LogReg:
    def __init__(self, num_features, eta):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.w = np.zeros(num_features)
        self.eta = eta
        self.last_update = defaultdict(int)

    def progress(self, examples_x, examples_y):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(y - p) < 0.5:
                num_right += 1

        return -1*logprob, float(num_right) / float(len(examples_y))

    def sgd_update(self, x_i, y):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """
        hypothesis = 0
        self.last_update=self.w
        #z=self.last_update.dot(x_i)
        for r in range(0,len(self.w)):

            self.w[r]=self.last_update[r] -self.eta*(sigmoid(self.last_update.dot(x_i))-y)*x_i[r]

        # TODO: Finish this function to do a single stochastic gradient descent update
        # and return the updated weight vector

        return self.w

def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    # TODO: Finish this function to return the output of applying the sigmoid
    # function to the input score (Please do not use external libraries)
    sigmoutput=1.0/(1.0+math.e**(-score))

    return sigmoutput

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eta", help="Initial SGD learning rate",
                           type=float, default=0.03, required=False)
    argparser.add_argument("--passes", help="Number of passes through training data",
                           type=int, default=1200, required=False)

    args = argparser.parse_args()

    data = Numbers('../data/mnist.pkl.gz')

    # Initialize model
    lr = LogReg(data.train_x.shape[1], args.eta)

    # Iterations
    cost_array=[]
    accuracy = []
    cost=0
    iteration = 0
    iteration_interval=40
    for epoch in range(args.passes):
        iteration=iteration+1
        data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)


        cost+=lr.progress([data.train_x[epoch]],[data.train_y[epoch]])[0]

        lr.sgd_update(data.train_x[epoch],data.train_y[epoch])
        if(iteration%iteration_interval==0):
            cost_array.append(cost)
            accuracy.append(lr.progress(data.test_x, data.test_y)[1])
            cost=0


    xplane=[x for x in range(0,1200,iteration_interval)]
    print("Min:" +str(min(cost_array)))
    print("Index:"+ str(xplane[cost_array.index(min(cost_array))]))
    #py.plot(xplane,cost_array,'b',label="eta= 0.2")

    # py.title('eta= 0000.1')
    # py.xlabel('No. of iterations')
    # py.ylabel('Negative log likelihood(cost)')
    # py.show()

    py.plot(xplane, accuracy)
    py.xlabel('No. of iterations')
    py.ylabel('Accuracy')
    py.show()

    data.test_x, data.test_y = Numbers.shuffle(data.test_x, data.test_y)
    prob, acc = lr.progress(data.test_x, data.test_y)
    print("Probability: %s" % prob)
    print("Accuracy: %s" % acc)



