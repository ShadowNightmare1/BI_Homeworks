# Mauricio Abarca J.
# 19.319.550-4

import pandas as pd
import numpy as np

# Hyperparameters and constants
METRICS_FILE = 'metrica.csv'
OUTPUT_NODES = 1

# Init.weights of the DL
def iniW(hidden_nodes, input_nodes, output_nodes):
    # Normal Initialization
    # w1 = np.random.rand(hidden_nodes, input_nodes)
    # w2 = np.random.randn(output_nodes, hidden_nodes)

    # He Initialization (most recommended for relu btw)
    # w1 = np.random.randn(hidden_nodes, input_nodes) * np.sqrt(2 / 1875) # size of previous layer (X)
    # w2 = np.random.randn(output_nodes, hidden_nodes) * np.sqrt(2 / 7500) # size of previous layer (A1


    # Xavier Initialization (Caso 2 ppt)
    # w1 = np.random.randn(hidden_nodes, input_nodes) * np.sqrt(1 / (hidden_nodes + input_nodes)) 
    # w2 = np.random.randn(output_nodes, hidden_nodes) * np.sqrt(1 / (output_nodes + hidden_nodes)) 

    # Caso 1 ppt
    r1 = np.sqrt(6 / (hidden_nodes + input_nodes))
    r2 = np.sqrt(6 / (output_nodes + hidden_nodes))

    w1 = np.random.randn(hidden_nodes, input_nodes) * 2 * r1 - r1
    w2 = np.random.randn(output_nodes, hidden_nodes) * 2 * r2 - r2

    return (w1, w2)

# STEP 1: Feed-forward of DAE
def forward(x, w1, w2):

    # Hidden Layer
    z1 = np.dot(w1, x) # w1.dot(x)
    a1 = act_sigmoid(z1)

    # Output Layer
    z2 = np.dot(w2, a1) # w2.dot(a1)
    a2 = act_sigmoid(z2)

    return a1, a2

# STEP 2: Gradient via BackPropagation
def grad_bp(a1, a2, x, error, w1, w2, lr):

    # Backward Pass Output
    deriv2 = deriva_sigmoid(a2)
    dZ2 = error * deriv2
    dW2 = np.dot(dZ2, a1.T)# dZ2.dot(a1.T)

    # Backward Pass Hidden
    deriv1 = deriva_sigmoid(a1)
    err1 = np.dot(w2.T, dZ2)
    dZ1 = err1 * deriv1
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
    r2Value = r2(y_pred, y_real)
    saveMetric(mae, rmse, r2Value)
    return 

def r2(y_pred, y_real):
    # ------------------------------------

    # ref: https://www.geeksforgeeks.org/python-coefficient-of-determination-r2-score/
    # indica que r2 = 1 - e1 ^2 / e2^2 
    # con e1 = y_real - y_pred
    # y e2 = y_real - .mean()  | que es la definicion de np.var() (en parte)

    # error = y_pred - y_real
    # error = np.power(error, 2) 
    # r2 = 1 - (np.var(error) / np.var(y_real)) # y así nos da valores más cercanos a 1  
    # r2 = 1 - (error.mean() / np.var(y_real))

    # ------------------------------------

    # https://medium.com/geekculture/linear-regression-from-scratch-in-python-without-scikit-learn-a06efe5dedb6 -> r2 = (pred -  mean())^2 / (real -mean() ^2)
    # sería var(a2) / var(y_real)
    # r2 = np.var(y_pred) / np.var(y_real) # da valores bien pequeños y segun esa formula, esto estaría mal (los valores se alejan de la recta)
    # r2 = 1 - (np.var(y_pred) / np.var(y_real)) # y ocupando esa formula nos daría algo más cercano a 1

    # ------------------------------------

    # another way: https://www.kite.com/python/answers/how-to-calculate-r-squared-with-numpy-in-python
    # corr_mat = np.corrcoef(y_real, y_pred)
    # corr_xy = corr_mat[0, 1]
    # r2 = corr_xy ** 2 # but it keeps giving bad values

    # ------------------------------------
    
    # https://www.colorado.edu/amath/sites/default/files/attached-files/ch12_0.pdf
    # r2 = 1 - SSE / SST = (SST - SSE)/SST = SSR/SST
    # SST -> np.var(y_real) | as seen aboveon other ref
    # SSE (Error of Sum Squares) -> sum((y_pred - y_real)**2)
    # error = y_real - y_pred
    # SSE = np.sum(error ** 2) 
    # SST = np.var(y_real)
    # r2 = 1 - SSE / SST  # da muy malos resultados
    
    # ----------------------------------------------

    # Según lo entregado por el profe
    
    err =  np.abs(y_pred - y_real)
    r2 = 1 - (np.var(err) / np.var(y_real))
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
    return ((df - df.min()) / (df.max() - df.min())) * (b - a) + a # this already do column wise operation (gets the min max of the column and when substracting substract by column)
                                                                   # in other words the error is not here (and we already check that min max of the result gives a,b values)

                                                                   # (this normalize is similar to the scikitlearn.preprocessing.MinMaxScaler one)

# Load data 
def load_data(fname):
    data = pd.read_csv(fname, header=None)
    X = normalize(data.drop(5, axis=1))
    y = normalize(data[[5]])

    X = X.to_numpy()
    y = y.to_numpy()

    # np.random.shuffle(X) # for some reason, without the shuffle it gives values range from 80 to 90 % 
    # np.random.shuffle(y)
        
    return X, y

#save weights of SNN in numpy format
def save_w(w1, w2, weight_fname, cost, cost_fname):    
    np.savez(weight_fname, w1, w2)
    pd.DataFrame(data=cost).to_csv(cost_fname, header=['mse'], index=None)
    print('Files: {} & {} were Saved!!'.format(weight_fname, cost_fname))
    
#load weight of SNN in numpy format
def load_w(fname):
    weights = np.load(fname)
    w1 = weights['arr_0']
    # print(weights['arr_0'])
    w2 = weights['arr_1']
    # print(weights['arr_1'])
    return(w1,w2)      
#