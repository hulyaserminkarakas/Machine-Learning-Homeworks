import numpy as np
from numpy import *
import scipy.sparse
import matplotlib.pyplot as plt


"""
class NeuralNetwork(object):
    def __init__(self, sizes = list()):
        self.weights = np.random.normal(0,0.01,(784, 10)).astype(np.float64)
        self.bias = np.random.normal(0,0.01,(10)).astype(np.float64)

       
        self.sizes = sizes
        self.hidden_layer_count = len(sizes)
        self.mini_batch_size = 16
        self.learning_rate = 1.0
        self.epocs = 10
    """

train_data = np.load('train-data.npy')
train_data = np.expand_dims(train_data,axis=1)
train_data = train_data.astype(np.float64)
train_label = np.load('train-label.npy')

validation_data = np.load('validation-data.npy')
validation_data = validation_data.astype(np.float64)

validation_label = np.load('validation-label.npy')

test_data = np.load('test-data.npy')
test_data = test_data.astype(np.float64)


def normalize(array):
    for i in range(784):
        array[i] = np.float64(array[i] / 256)


#ACTIVATION FUNCTIONS

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1. - x * x

def ReLU(x):
    return x * (x > 0)

def ReLU_derivative(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1. - x)

def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T

def softmax_derivative(x):
    exps = np.exp(x)
    others = exps.sum() - exps
    return 1 / (2 + exps / others + others / exps)


def one_hot(Y):
    x = Y.shape[0]
    temp = scipy.sparse.csr_matrix((np.ones(x), (Y, np.array(range(x)))))
    temp = np.array(temp.todense()).T
    return temp

def single_layer():

    normalize(train_data)
    normalize(validation_data)

    weights = np.random.normal(0, 0.01, (784, 10)).astype(np.float64)
    bias = np.random.normal(0, 0.01, (10)).astype(np.float64)
    batch_size = 10
    learning_rate = 0.05
    epoc = 10

    #print (weights[0])
    for i in range(epoc):
        for i in range(0,len(train_data),batch_size):
            x= train_data[i:i+batch_size].shape[0]
            y=train_label[i: i+batch_size]

            perceptron = dot(x, weights) + bias
            delta= softmax(perceptron)
            mat = one_hot(y)

            #LOSS FUNCTION
            loss = (-1 / x) * np.sum(mat * np.log(delta)) + (learning_rate / 2) * np.sum(weights * weights)
            grad = (-1 / x) * np.dot(x.T, (mat - delta)) + loss * weights

            delta_bias = delta* softmax_derivative(grad)
            delta_weight =dot(x.T, delta) * softmax_derivative(grad)

            weights =weights + delta_weight
            bias = bias + delta_bias

    #print(weights[0])
    score = 0
    for i in range(0, len(validation_data), batch_size):
        x = validation_data[i:i + batch_size]
        perc = dot(x, weights) + bias
        pred = softmax(perc)
        for j in range(len(pred)):
            if argmax(pred[j]) == validation_label[i+j]:
                score = score + 1

    print (score / len(validation_data))

def multi_layer(hiddenSize):

    normalize(train_data)
    normalize(validation_data)

    learning_rate = 0.03

    X = train_data
    y = train_label

    np.random.seed(1)

    # randomly initialize our weights with mean 0
    weight_0 = 2*np.random.random((3,hiddenSize)) - 1
    weight_1 = 2*np.random.random((hiddenSize,1)) - 1

    for j in range(len(train_data)):

        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,weight_0))
        layer_2 = sigmoid(np.dot(layer_1,weight_1))

        layer_2_error = layer_2 - y
        layer_2_delta = layer_2_error*sigmoid_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(weight_1.T)
        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

        weight_1 -= learning_rate * (layer_1.T.dot(layer_2_delta))
        weight_0 -= learning_rate * (layer_0.T.dot(layer_1_delta))

def visualize(x):
    img = train_data[x]
    img = np.reshape(img, (28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

single_layer()
multi_layer(25)
visualize(0)