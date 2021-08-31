# Mauricio Abarca J.
# 19.319.550-4

import numpy as np
import pandas as pd


# Functions

def normalize(df):
    global a
    global b
    return ((df - df.min()) / (df.max() - df.min())) * (b - a) + a


def weightsFromCSV(arch_name):
    return pd.read_csv(arch_name, header=None)


def loadData(arch_name):
    data = pd.read_csv(arch_name, header=None)
    X = normalize(data.drop(5, axis=1))
    y = normalize(data[[5]])

    X = X.to_numpy()
    y = y.to_numpy()

    np.random.shuffle(X)
    np.random.shuffle(y)

    return X, y


def loadConfig(arch_name):
    config = pd.read_csv(arch_name, header=None)
    return int(config[0][0]), int(config[0][1]), config[0][2]


# Classes

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_neurons, n_inputs) # W = Nodes of Next layer , Nodes of previous layer

    def forward(self, inputs):
        # self.output = np.dot(self.weights, inputs ) # Z = W X
        self.output = self.weights @ inputs # we are getting less loss value using @ instead of .dot (and as far as i know they are the same)

    def backward(self, dE, current_deriv, prev_activ):
        self.dZ = np.multiply(dE, current_deriv) # hadamard product
        # self.dW = np.dot(self.dZ, prev_activ.T)
        self.dW = self.dZ @ prev_activ.T # refactorear codigo
        self.updateWeights()

    def updateWeights(self):
        global lr
        self.weights = self.weights - lr * self.dW


    def loadWeights(self, weights): # weights should be a dataframe
        self.weights = weights.to_numpy()
    
    def saveWeights(self, tag): # change to npx later
        df = pd.DataFrame(data=self.weights)
        df.to_csv('weight_{}.csv'.format(tag), header=None, index=None)


class ActivationLayer:
    def sigmoid(self, inputs):
        sigmoid = 1 / (1 + np.exp(- inputs))
        self.derivateSigmoid(sigmoid)
        self.output = sigmoid


    def derivateSigmoid(self, inputs): # the input must be the output of the sigmoid
        self.sigmoidPrime = inputs * (1 - inputs)

    
class Metrics: # refactor to Metrics and add other metrics methods

    def __init__(self):
        self.loss_df = pd.DataFrame(columns=['epoch', 'mse'])

    def calculateError(self, y_real, y_pred):
        # self.error = np.subtract(y_real, y_pred)
        # self.error = y_real - y_pred
        self.error = y_pred - y_real

    def metrics(self, y_real, y_pred, epoch):
        self.calculateError(y_real, y_pred)
        self.calculateMSE()
        self.calculateRMSE()
        self.calculateMAE()
        self.calculateR2(y_real)
        self.updateLoss(epoch)

    def metricsTest(self, y_real, y_pred):
        self.calculateError(y_real, y_pred)
        self.calculateMSE()
        self.calculateRMSE()
        self.calculateMAE()
        self.calculateR2(y_real)

    def calculateMSE(self):
        self.mse = np.mean((self.error ** 2))

    def calculateRMSE(self):
        self.rmse = np.sqrt(self.mse)

    def calculateMAE(self):
        self.mae = np.mean(np.abs(self.error))
    
    def calculateR2(self, y_real):
        self.r2 = 1 - (np.var(self.error) / np.var(y_real))

    def updateLoss(self, epoch):
        self.loss_df = self.loss_df.append({
            'epoch': epoch + 1,
            'mse': self.mse
        }, ignore_index=True)
        self.loss_df['epoch'] = self.loss_df['epoch'].astype('int64') # this is to prevent from getting epoch as a float

    def exportLoss(self):
        self.loss_df.to_csv(LOSS_FILE, index=None)

    def exportMetrics(self):
        data = [self.mae, self.rmse, self.r2]
        self.metrics_df = pd.DataFrame(data=data)
        self.metrics_df.to_csv(METRICS_FILE, header=None, index=None)

# Hyperparameters and constants

CONFIG_FILE = 'config.csv'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
LOSS_FILE = 'costo.csv'
METRICS_FILE = 'metrica.csv'
DEV_WEIGHT_1 = 'weight_1.csv'
DEV_WEIGHT_2 = 'weight_2.csv'
WEIGHTS_FILE = 'weights.npx'

a = 0.01
b = 0.99

hidden_nodes, iterations, lr  = loadConfig(CONFIG_FILE) # if we can't use os module we can't use folders and a better file management
output_nodes = 1
