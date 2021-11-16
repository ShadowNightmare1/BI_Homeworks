# Mauricio Abarca J.
# 19.319.550-4

from numpy.ma import identity
import pandas as pd
import numpy  as np


# Hyperparameters and constants
BETA = 0.9
EPSILON = 10 ** -8
METRICS_FILE = 'metrica.csv'
CFG_SAE_FILE = 'cnf_sae.csv'
CFG_SOFTMAX_FILE = 'cnf_softmax.csv'
DL_WEIGHT_FILE = 'w_dl.npz'
COST_FILE = 'costo_softmax.csv'


# Initialize weights
def iniW(next, prev):
    r = np.sqrt(6 / (next + prev))
    w = np.random.rand(next, prev)
    w = w * 2 * r - r    
    return w


def ini_WV(next, prev):
    W = iniW(next, prev)
    V = np.zeros((next, prev))
    return W, V


# STEP 1: Feed-forward of AE
# def forward_ae(x,w1,w2):
    # complete code
    # return(a)    

def forward_ae(x, w):
    z = w.dot(x)
    a = act_sigmoid(z)
    return a

#Activation function
def act_sigmoid(z):
    return(1 / (1 + np.exp( - z)))   


# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a * (1 - a))


# Calculate Pseudo-inverse
def pinv_ae(x, w1, C):     
    a1 = forward_ae(x, w1)
    identity = np.identity(x.shape[1]) # not sure if it's the correct size
    w2 = ( np.dot(x, a1.T) ) * (( np.dot(a1, a1.T) + (identity / C)) ** -1)
    return w2


# STEP 2: Feed-Backward for AE
def gradW1(a1, w2, error, a2, x):  
    deriv2 = a2 # debiera ser la derivada de la identidad :/
    deriv1 = deriva_sigmoid(a1)
    dZ1 = error * deriv2
    gW1 = np.dot(w2.T, dZ1) * (np.dot(deriv1, x.T))
    cost = np.sum(a2 - x).mean() * 0.5
    return gW1, cost


# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    v = BETA * v + (1 - BETA) * (gw ** 2) # try dot later
    grms = (mu / np.sqrt(v + EPSILON)) * gw
    w = w - grms
    return w, v


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w,v,gw,mu):
    #complete code       
    return(w,v)


# Softmax's gradient
def gradW_softmax(x,y,a):
    #complete code           
    return(gW,Cost)


# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# Métrica
def metricas(x,y):
    #complete code           
    return(...)
    

#Confusion matrix
def confusion_matrix(y,z):
    #complete code               
    return(cm)


def mse(error):
    return np.power(error, 2).mean()


#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      

    # SAE CONFIG
    config_sae = pd.read_csv(CFG_SAE_FILE, header=None)

    penalty = float(config_sae[0][0])       # Penalty 
    batch_size = int(config_sae[0][1])      # Batch size
    sae_max_iter = int(config_sae[0][2])    # MaxIter
    lr = float(config_sae[0][3])            # Learn rate    
    
    
    # Best Approach for now
    hidden_nodes = list()
    for i in range(4, len(config_sae)):
        hidden_nodes.append(int(config_sae[0][i])) # Hidden Nodes

    # SOFTMAX CONFIG
    config_softmax = pd.read_csv(CFG_SOFTMAX_FILE, header=None)
    sftmx_max_iter = int(config_softmax[0][0])  #MaxIters
    mu = float(config_softmax[0][1])            #Learning     

    # RETURNING VALUES
    par_sae = [penalty, batch_size, sae_max_iter, lr, hidden_nodes]
    par_sft = [sftmx_max_iter, mu]

    return par_sae, par_sft


# Load data 
def load_data_csv(fname):
    aux = pd.read_csv(fname, header = None)
    y_aux = aux.T[[256]]
    y_aux.columns = ['Class']
    y = pd.get_dummies(y_aux.Class, prefix='Class')
    y = np.array(y)

    x = aux.T
    x = x.drop([256], axis=1)
    x = np.array(x)

    return  x.T, y.T    # (sample x class)


# Save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):   
    #comple code
    return

# Load weight of the DL in numpy format
def load_w_dl():
    #complete code   
    return(W)   
    