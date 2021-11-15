# Mauricio Abarca J.
# 19.319.550-4

import pandas as pd
import numpy  as np


# Hyperparameters and constants
BETA = 0.9
METRICS_FILE = 'metrica.csv'
CFG_SAE_FILE = 'cnf_sae.csv'
CFG_SOFTMAX_FILE = 'cnf_softmax.csv'
DL_WEIGHT_FILE = 'w_dl.npz'
COST_FILE = 'costo_softmax.csv'


def getBatchSize():
    # SAE CONFIG
    config_sae = pd.read_csv(CFG_SAE_FILE, header=None)
    batch_size = int(config_sae[0][0])
    return batch_size

SOFTMAX_BATCH_SIZE = getBatchSize()

# Initialize weights
def iniWs(prev,next):
    W = iniW(next, prev)
    V = np.zeros((next, prev))
    return W, V


# Initialize Matrix's weight    
def iniW(next, prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w *2 * r - r    
    return w

    
# STEP 1: Feed-forward of AE
def forward_ae(x, w):
    z = w.dot(x)
    a = act_sigmoid(z)
    return a   

def forward_softmax(x, w):
    z = np.dot(w, x)
    a = softmax(z)
    return a

#Activation function
def act_sigmoid(z):
    return(1 / (1 + np.exp(- z)))   


# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))


# STEP 2: Feed-Backward for AE
def gradW_ae(a1, a2 , x, w2, error):    

    # Backward Pass Output
    deriv2 = deriva_sigmoid(a2)
    dZ2 = error * deriv2
    dW2 = np.dot(dZ2, a1.T)

    # Backward Pass Hidden
    deriv1 = deriva_sigmoid(a1)
    err1 = np.dot(w2.T, dZ2)
    dZ1 = err1 * deriv1
    dW1 = np.dot(dZ1, x.T) 

    return dW1, dW2



# Update AE's weights via SGD Momentum
def updW_ae_sgd(w, v, gw, mu):
    v = BETA * v + mu * gw
    w -= v
    return w, v    


# Softmax's gradient
def gradW_softmax(x, y, a):   
    gW = (- 1 / y.shape[1]) * np.dot((y - a), x.T)
    Cost = cross_entropy(y, a, x)
    return gW, Cost


def cross_entropy(y_real, y_pred, x):
    # print(y_real.shape[1]) # -> 25
    val = - 1 / y_real.shape[1]
    ce = np.multiply(y_real, np.log(y_pred)) # usar doble sum (ver ppt!!)
    cost = val * np.dot(ce, x.T)
    return cost.mean() # OJO!

# Update Softmax's weights via SGD Momentum
def updW_sft_sgd(w, v, gw, mu):
    v = BETA * v  + mu * gw # se estabiliza
    w = w - v
    
    return w, v


# Calculate Softmax
def softmax(z):
   # axis 0  -> columns | axis 1 -> rows
   exp_values = np.exp(z - np.max(z, axis=0, keepdims=True)) # this is to prevent overflow 
   probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
   return probabilities
   # return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)


# Métrica
def metricas(y_real, y_pred): # yv, zv # it works (i think)
    cm = confusion_matrix(y_real, y_pred)
    col = np.arange(1, cm.shape[0] + 1)
    col = list(col)

    prec = precission(cm)
   
    rec = recall(cm)

    fscore = fScore(prec, rec)

    avgFScore = np.sum(fscore) / cm.shape[0]

    conf_csv = pd.DataFrame(cm, columns=col)
    conf_csv.to_csv('confusion_matrix.csv', index=None)

    
    col.extend(['Avg F-Score'])
    newData = list(fscore)
    newData.extend([avgFScore])
    newData = np.array(newData)
    newData = np.expand_dims(newData, axis=0)

    metrica_csv = pd.DataFrame(newData, columns=col, index=['F-Score'])
    metrica_csv.to_csv(METRICS_FILE)

    return 


def precission(cm):
    diagonal = cm.diagonal()
    denom = np.sum(cm, axis=1)
    prec = diagonal / denom
    return prec


def recall(cm):
    diagonal = cm.diagonal()
    denom = np.sum(cm, axis=0)
    rec = diagonal / denom
    return rec


def fScore(prec, rec):
    num = prec * rec
    denom = prec + rec
    fscore = 2 * (num / denom)
    return fscore

def mse(error):
    return np.power(error, 2).mean()


#Confusion matrix
def confusion_matrix(y_real, y_pred): # yv, zv  # Funciona, con los datos del profe (de test) dio la misma matriz
    cm = np.zeros((y_real.shape[0], y_real.shape[0]))
    i = np.argmax(y_pred, axis=0)
    j = np.argmax(y_real, axis=0)
    
    for k in range(y_real.shape[1]):
        cm[i[k], j[k]] += 1
    return cm


# Mini batch function
def get_miniBatch(i, x, bsize):
    z = x[:, i * bsize : ( i + 1) * bsize]
    return z
    

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------

# Configuration of the SNN
def load_config():      
    
    # SAE CONFIG
    config_sae = pd.read_csv(CFG_SAE_FILE, header=None)
    batch_size = int(config_sae[0][0])
    sae_max_iter = int(config_sae[0][1])
    lr = float(config_sae[0][2])

    # Best Approach for now
    hidden_nodes = list()
    for i in range(3, len(config_sae)):
        hidden_nodes.append(int(config_sae[0][i]))

    # SOFTMAX CONFIG
    config_softmax = pd.read_csv(CFG_SOFTMAX_FILE, header=None)
    sftmx_max_iter = int(config_softmax[0][0])
    mu = float(config_softmax[0][1])

    # RETURNING VALUES
    params_sae = [batch_size, sae_max_iter, lr, hidden_nodes]
    params_softmax = [sftmx_max_iter, mu]

    return params_sae, params_softmax


# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)


# save weights SAE and costo of Softmax
def save_w_dl(W, Ws, cost):    
    # print(len(W))
    np.savez(DL_WEIGHT_FILE, wAE1 = W[0], 
                            wAE2 = W[1],
                            wAE3 = W[2],
                            #wAE4 = W[3],
                            wSoftMax = Ws)
    pd.DataFrame(data = cost).to_csv(COST_FILE, header=['Softmax Cost'],
                                                 index=None)
    
    print('Files: {} & {} were created and saved!'.format(DL_WEIGHT_FILE,
                                                          COST_FILE)) 


#load weight of the DL 
def load_w_dl():
    weights = np.load(DL_WEIGHT_FILE)
    W = [weights['wAE1'], 
        weights['wAE2'], 
        weights['wAE3'],
        #weights['wAE4'] ,
        weights['wSoftMax']] 
    weights.close()
    return W    


# save weights in numpy format
# def save_w_npy(w1, w2, mse):  
#     #complete code
#     return
