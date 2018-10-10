# __author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class Layer(object):

    def __init__(self, in_dim, out_dim, actFun_type):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)
        self.b = np.zeros((1, self.out_dim))
        self.actFun_type = actFun_type

    def feedforward(self, input):
        self.input = input
        self.z = self.input.dot(self.W) + self.b
        self.output = self.actFun(self.z, self.actFun_type)
        return None

    def backprop(self, prev_delta, z, actFun_type):
        self.dW = self.input.T.dot(prev_delta)
        self.db = np.sum(prev_delta, axis=0, keepdims=True)
        self.delta = np.multiply(prev_delta.dot(self.W.T), self.diff_actFun(z, actFun_type))
        return None

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'Tanh':
            return np.tanh(z)
        elif type == 'Sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'ReLU':
            return np.multiply(z, (z > 0))
        elif type == 'Output':
            return z
        else:
            print('Type not supported! ')

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'Tanh':
            return 1 - np.square(np.tanh(z))
        elif type == 'Sigmoid':
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig * (1.0 - sig)
        elif type == 'ReLU':
            return np.where(z > 0, 1.0, 0.0)
        elif type == 'Output':
            return 1
        else:
            print('Type not supported! ')

        return None

class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, num_layer, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='Tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.num_layer = num_layer
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.model = []
        input_layer = Layer(self.nn_input_dim, self.nn_hidden_dim[0], self.actFun_type)
        self.model.append(input_layer)
        for i in range(self.num_layer-3):
            layer = Layer(self.nn_hidden_dim[i], self.nn_hidden_dim[i+1], self.actFun_type)
            self.model.append(layer)
        output_layer = Layer(self.nn_hidden_dim[self.num_layer-3], self.nn_output_dim, 'Output')
        self.model.append(output_layer)


    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'Tanh':
            return np.tanh(z)
        elif type == 'Sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'ReLU':
            return np.multiply(z, (z > 0))
        elif type == 'Output':
            return z
        else:
            print('Type not supported! ')

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'Tanh':
            return 1 - np.square(np.tanh(z))
        elif type == 'Sigmoid':
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig * (1.0 - sig)
        elif type == 'ReLU':
            return np.where(z > 0, 1.0, 0.0)
        elif type == 'Output':
            return 1
        else:
            print('Type not supported! ')

        return None

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.model[0].feedforward(X)
        for i in range(1, self.num_layer-1):
            self.model[i].feedforward(self.model[i-1].output)
        self.model[-1].feedforward(self.model[-2].output)
        exp_scores = np.exp(self.model[-1].output)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = -np.sum(np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        # data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss
        # return 1

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta_out = self.probs
        delta_out[range(num_examples), y] -= 1
        self.model[-1].backprop(delta_out, self.model[-2].z, self.model[-2].actFun_type)

        for i in range(len(self.model)-2, -1, -1):
            self.model[i].backprop(self.model[i+1].delta, self.model[i-1].z, self.model[i-1].actFun_type)

        return None

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        print('fitting')
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for layer in self.model:
                layer.dW += self.reg_lambda * layer.W
                layer.W -= epsilon * layer.dW
                layer.b -= epsilon * layer.db
            # Gradient descent parameter update


            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():


    # generate and visualize Make-Moons dataset
    # print('???????')
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = NeuralNetwork(3, 2, [3], 2, actFun_type='ReLU')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()