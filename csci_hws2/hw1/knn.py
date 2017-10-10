
import argparse
import pickle
import gzip
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree
from sklearn.metrics import confusion_matrix as cm

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.


        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """
        
        # Finish this function to store necessary data so you can 
        # do classification later

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        print (item_indices)
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y value for
        # these indices
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        item_indices_length=len(item_indices)



        countoflabels=Counter(numpy.array(self._y[item_indices]).flatten())
        print(countoflabels)
        desired_keys = []

        vals = countoflabels.values()


        for key, value in countoflabels.items():
            if self._k > 1 and Counter(vals)[value] > 1:
                return numpy.median(self._y[item_indices])

        return Counter(self._y[item_indices]).most_common(1)[0][0]






    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the value.
        dist, index = self._kdtree.query([example], k=self._k)
        print (dist)
        print (index)
        maj= self.majority(numpy.array(index).flatten())



        return maj;

        """return self.majority(list(random.randrange(len(self._y)) \
                                  for x in range(self._k)))"""

    def confusion_matrix(self, test_x, test_y, debug=False):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.
        pred_array=[];

        for h in range(len(test_x)):
        #for h in range(40):
            arr=self.classify((test_x[h]))
            pred_array.append(arr)
        d = dict((i, (dict((j,0) for j in range(0,10)))) for i in range(0,10))

        data_index = 0
        cmp = numpy.zeros((10, 10))
        for xx, yy in zip(pred_array, test_y):
            data_index += 1
            d[xx][yy] += 1
            if debug and data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(pred_array)))

        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total > 0:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=1000,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))

