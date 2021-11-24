# Mauricio Abarca J.
# 19.319.550-4

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


def forward_ae_out(x, w):
    z = w.dot(x)
    return z


def forward_softmax(x, w):
    z = w.dot(x)
    a = softmax(z)
    return a


def forward_dl(x, W):
    for key in W.keys():
        if key == 'Ws':
            break
        if key == 'WAE1':
            input = x
        else:
            input = a

        a = forward_ae(input, W[key])

    aS = forward_softmax(a, W['Ws'])

    return aS

#Activation function
def act_sigmoid(z):
    return np.exp(np.fmin(z, 0)) / (1 + np.exp(- np.abs(z))) # to avoid overflow (as far as i seen on stackoverflow this replaces the commented above)
                                                            # and works for cases when the overflow appeared


# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a * (1 - a))


# Calculate Pseudo-inverse
def pinv_ae(x, w1, C):     
    a1 = forward_ae(x, w1) # (m x n) | nodos x muestra
    identity = np.identity(a1.shape[0]) 
    w2 = np.dot((np.dot(x, a1.T)), np.linalg.inv(( np.dot(a1, a1.T) + (identity / C))))
    return w2


# STEP 2: Feed-Backward for AE
def gradW1(a1, w2, error, x):
    Z2 = forward_ae(a1, w2)
    dZ2 = np.multiply(error, Z2)

    error1 = np.dot(w2.T, dZ2)
    dZ1 = np.multiply(error1, deriva_sigmoid(a1))
    dW1 = np.dot(dZ1, x.T)
    return dW1


# Update AE's weight via RMSprop
def updW1_rmsprop(w, v, gw, mu):
    v = np.dot(BETA, v) + np.dot(1 - BETA, gw ** 2)
    grms = np.multiply(mu / np.sqrt(v + EPSILON), gw)
    w -= grms
    return w, v


# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w, v, gw, mu):
    v = np.dot(BETA, v) + np.dot(1 - BETA, gw ** 2)
    grms = np.multiply(mu / np.sqrt(v + EPSILON), gw)
    w -= grms
    return w, v


# Softmax's gradient
def gradW_softmax(x, y, a, batchSize):
    Cost = cross_entropy(y, a, batchSize)
    error = np.dot(y - a, x.T)
    dWs = (- 1 / batchSize) * error
    return dWs, Cost


def cross_entropy(y_real, y_pred, batchSize):
    val = y_real * np.log(y_pred)
    inner_sum = np.sum(val, axis=0, keepdims=True)
    outer_sum = np.sum(inner_sum, axis=1)
    cost = (- 1 / batchSize) * outer_sum
    return cost

# Calculate Softmax
def softmax(z):
   exp_values = np.exp(z - np.max(z, axis=0, keepdims=True)) # this is to prevent overflow 
   probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
   return probabilities


# Métrica
def metricas(y_real, y_pred): 
    cm = confusion_matrix(y_real, y_pred)
    col = np.arange(1, cm.shape[0] + 1)
    col = list(col)
    print(cm)
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

    print('File: {}, genereated succesfully'.format(METRICS_FILE))
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
def save_w_dl(W, Ws, cost):   
    W.update({'Ws': Ws})
    np.savez(DL_WEIGHT_FILE, **W)
    pd.DataFrame(cost).to_csv(COST_FILE, header=['Softmax Cost'], index=None)

    print('Files: {} & {} were created and saved!'.format(DL_WEIGHT_FILE,
                                                          COST_FILE)) 

# Load weight of the DL in numpy format
def load_w_dl():
    weights = np.load(DL_WEIGHT_FILE)
    W = dict()
    print(weights.files)
    for wf in weights.files:
        W.update({wf: weights[wf]})

    weights.close()
    return W


def save_for_test(x, w, y):
    np.savez('train_weight', **w)
    pd.DataFrame(x).to_csv('Xr.csv', index=None, header=None)
    pd.DataFrame(y).to_csv('y.csv', index=None, header=None)
    return

def load_for_test():
    y = pd.read_csv('y.csv', header=None)
    y = np.array(y)

    Xr = pd.read_csv('Xr.csv', header=None)
    Xr = np.array(Xr)
    return Xr, y