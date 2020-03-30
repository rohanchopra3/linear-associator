# Chopra, Rohan
# 1001-780-925
# 2020-03-01
# Assignment-02-01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)

        self.weights = np.random.randn(self.number_of_nodes ,self.input_dimensions)


    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if np.shape(W) == (self.number_of_nodes,self.input_dimensions):
            self.weights = W

            return None
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        net = np.dot(self.weights, X)

        # Hard Limit Function , followed by conversion of Z to an int array using astype
        if self.transfer_function == "Hard_limit":
             Z = net
             Z[Z >= 0] = 1
             Z[Z < 0] = 0
             Z = Z.astype(int)
        else:
            Z = net
        return Z


    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        P_pseudo = np.linalg.pinv(X)
        self.set_weights(np.dot(y, P_pseudo))

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        for i in range(num_epochs):
                for j in range(0, np.shape(X)[1], batch_size):

                    val = j + batch_size
                    X_Batch = X[: , j:val]
                    y_Batch = y[: , j:val]
                    a = self.predict(X_Batch)
                    e = y_Batch - a
                    if learning == "Delta" or learning == "delta":
                        self.weights = self.weights + (alpha * np.dot(e, np.transpose(X_Batch)))
                    elif learning =="Filtered" or learning == "filtered":
                        self.weights = (1 - gamma)*self.weights + (alpha * (np.dot(y_Batch, np.transpose(X_Batch))))
                    elif learning == "Unsupervised_hebb" or learning == "unsupervised_hebb":
                        self.weights = self.weights + (alpha * (np.dot(a, np.transpose(X_Batch))))


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        e_square = np.square(y - self.predict(X))
        sum = np.sum(e_square)
        val = np.shape(X)
        MSE = sum / (val[1] *val[0])

        return MSE
