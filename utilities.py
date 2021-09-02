# Mauricio Abarca J.
# 19.319.550-4

import pandas as pd
import numpy as np

# Hyperparameters and constants
METRICS_FILE = 'metrica.csv'
OUTPUT_NODES = 1

# Init.weights of the DL
def iniW(hidden_nodes, input_nodes, output_nodes):
    w1 = np.random.randn(hidden_nodes, input_nodes)
    w2 = np.random.randn(output_nodes, hidden_nodes)
    return (w1, w2)

# STEP 1: Feed-forward of DAE
def forward(x, w1, w2):

    # Hidden Layer
    z1 = w1.dot(x)
    a1 = act_sigmoid(z1)

    # Output Layer
    z2 = w2.dot(a1)
    a2 = act_sigmoid(z2)

    return a1, a2

# STEP 2: Gradient via BackPropagation
def grad_bp(a1, a2, x, error, w1, w2, lr):

    # Backward Pass Output
    deriv2 = deriva_sigmoid(a2)
    dZ2 = np.multiply(error, deriv2)
    dW2 = dZ2.dot(a1.T)

    # Backward Pass Hidden
    deriv1 = deriva_sigmoid(a1)
    dZ1 = np.multiply(w2.T.dot(dZ2), deriv1)
    dW1 = dZ1.dot(x.T)

    W1, W2 = updW(w1, dW1, w2, dW2, lr)
    return W1, W2

# Update SNN's Weight
def updW(w1, dW1, w2, dW2, lr):
    W1 = w1 - (lr * dW1)
    W2 = w2 - (lr * dW2)
    return W1, W2

# Activation Function
def act_sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# Derivate of the Activation function    
def deriva_sigmoid(a):
    return (a * (1 - a))

def saveMetric(mae, rmse, r2):
    data = [mae, rmse, r2]
    metrics_df = pd.DataFrame(data=data)
    metrics_df.to_csv(METRICS_FILE, header=None, index=None)
    print('File: {} Generated!'.format(METRICS_FILE))

# Metrica
def metrica(y_pred, y_real):
    error = y_pred - y_real
    mse = np.power(error, 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(error).mean()
    r2 = 1 - (np.var(error) / np.var(y_real))    
    saveMetric(mae, rmse, r2)
    return 

def r2(y_pred, y_real):
    r2 = 1 - (np.var(y_pred - y_real) / np.var(y_real))
    return r2

def mse(error):
    return np.power(error, 2).mean()

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------

# Load Config function
#   Input:
#       fname: String | file name (with extension)
#   Output:
#       hidden_nodes: Number of hidden nodes given in the config file
#       iterations: Number of iterations given in the config file
#       lr: Learning Rate given in the config file
#       (all the above inside a list)
#
def load_config(fname):
    config = pd.read_csv(fname, header=None)
    hidden_nodes = int(config[0][0])
    iterations = int(config[0][1])
    lr = config[0][2]
    return hidden_nodes, iterations, lr


# Normalize Function
#   Input:
#       df: DataFrame object | The csv file previously loaded with pandas
#   Output:
#       A DataFrame object normalized with the min max function and other hyperparameters (this was given as a requirement)
#
def normalize(df):
    a = 0.01  # move later to config or something, as this is a hyperparameter
    b = 0.99
    return ((df - df.min()) / (df.max() - df.min())) * (b - a) + a


# Load data 
def load_data(fname):
    data = pd.read_csv(fname, header=None)
    X = normalize(data.drop(5, axis=1))
    y = normalize(data[[5]])

    X = X.to_numpy()
    y = y.to_numpy()

    np.random.shuffle(X)
    np.random.shuffle(y)

    return X, y

#save weights of SNN in numpy format
def save_w(w1, w2, weight_fname, cost, cost_fname):    
    np.savez(weight_fname, w1, w2)
    pd.DataFrame(data=cost).to_csv(cost_fname, header=['mse'], index=None)
    print('Files Saved!')
    
#load weight of SNN in numpy format
def load_w(fname):
    weights = np.load(fname)
    w1 = weights['arr_0']
    # print(weights['arr_0'])
    w2 = weights['arr_1']
    # print(weights['arr_1'])
    return(w1,w2)      
#