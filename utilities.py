# Mauricio Abarca J.
# 19.319.550-4

import pandas as pd
import numpy  as np

# Hyperparameters and constants
METRICS_FILE = 'metrica.csv'
CFG_SAE_FILE = 'cnf_sae.csv'
CFG_SOFTMAX_FILE = 'cnf_softmax.csv'

# Init.weight of the DL
def iniW(next_nodes, prev_nodes):
    # Weight init formula
    r = np.sqrt(6 / (next_nodes + prev_nodes))
    w = np.random.rand(next_nodes, prev_nodes) * 2 * r - r
    return w
    

# STEP 1: Feed-forward of AE
def forward_ae(x,w1,w2):	
    #complete code
	return(...)
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(a,x,w1,w2):    
    #complete code
    return(gW1,gW2)    

# Update W of the AE
def updW_ae(w1,w2,gW1,gW2,mu):
    w1-= mu*gW1
    w2-= mu*gW2
    return(w1,w2)

# Softmax's gradient
def grad_softmax(x,y,w,lambW):    
    #complete code    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
    #complete code          
    return(...)

# Métrica
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    #complete code              
    return(...)
    
#Confusuon matrix
def confusion_matrix(x,y):
    #complete code              
    return(cm)

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      

    # SAE CONFIG
    config_sae = pd.read_csv(CFG_SAE_FILE, header=None)
    sae_max_iter = int(config_sae[0][0])
    lr = float(config_sae[0][1]) # just for ensure float number
    # make this more dynamic later
    hidden_nodes_1 = int(config_sae[0][2])
    hidden_nodes_2 = int(config_sae[0][3])
    hidden_nodes_3 = int(config_sae[0][4])

    # SOFTMAX CONFIG
    config_softmax = pd.read_csv(CFG_SOFTMAX_FILE, header=None)
    sftmx_max_iter = int(config_softmax[0][0])
    mu = float(config_softmax[0][1]) # this is the learning rate
    lambda_softmax = float(config_softmax[0][2]) # penalty

    # RETURNING VALUES
    params_sae = [sae_max_iter, lr, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3]
    params_softmax = [sftmx_max_iter, mu, lambda_softmax]

    return params_sae, params_softmax


# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    # x = x.to_numpy() # maybe this added the error to our previous work
    return(x)

# save costo of Softmax and weights SAE 
def save_w_dl(W,Ws,cost):    
    #complete code
   
#load weight of the DL 
def load_w_dl():
    #complete code    
    return(W)    