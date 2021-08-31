# Mauricio Abarca J.
# 19.319.550-4

from utilities import METRICS_FILE
import pandas as pd
import numpy as np

# Hyperparameters and constants
METRICS_FILE = 'metrica.csv'

# Init.weights of the DL
def iniW(...):
    # Init. w1 & w2
    # Complete code
    return (w1, w2)

# STEP 1: Feed-forward of DAE
def forward(x, w1, w2):
    #complete code
    return

# STEP 2: Gradient via BackPropagation
def grad_bp(...):
    #complete code
    return

# Update SNN's Weight
def updW(...):
    #complete code
    return

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
def metrica(y_real, y_pred):
    error = y_pred - y_real
    mse = np.power(error, 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(error).mean()
    r2 = 1 - (np.var(error) / np.var(y_real))    
    saveMetric(mae, rmse, r2)
    return 

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
    return list(hidden_nodes, iterations, lr)


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
def save_w(...):    
    #completar code    
    
#load weight of SNN in numpy format
def load_w(fname):
    #completar code
    # return pd.read_csv(fname, header=None) # this is for 1 weight
    return(w1,w2)      
#