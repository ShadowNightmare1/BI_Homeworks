# Mauricio Abarca J.
# 19.319.550-4

import pandas as pd
import numpy  as np

# Hyperparameters and constants
METRICS_FILE = 'metrica.csv'
CFG_SAE_FILE = 'cnf_sae.csv'
CFG_SOFTMAX_FILE = 'cnf_softmax.csv'
CODE = 10
DL_WEIGHT_FILE = 'w_dl.npz'
COST_FILE = 'costo_softmax.csv'
EPSILON = 1e-12
N = 100
CLASSES = 256


# Init.weight of the DL
def iniW(prev_nodes, next_nodes):

    # Weight Initialization (Case 2 ppt)
    r = np.sqrt(6. / (next_nodes + prev_nodes))
    w = np.random.rand(next_nodes, prev_nodes) * 2. * r - r
    return w
    

def generator():
    X = np.zeros((CLASSES, N))
    Y = np.zeros((CLASSES, N))

    for i in range(CLASSES):
        for j in range(N):
            newValX = np.random.uniform(0, 1)
            newValY = np.random.uniform(0, 1)
            X[i][j] = newValX
            Y[i][j] = newValY
    
    pd.DataFrame(X).to_csv('gen_test_x.csv', header=None, index=None)
    pd.DataFrame(Y).to_csv('gen_test_y.csv', header=None, index=None)

    return X, Y
# STEP 1: Feed-forward of AE
def forward_ae(x, w1, w2):	
    
    # Hidden Layer
    z1 = np.dot(w1, x)
    a1 = act_sigmoid(z1)

    # Output Layer
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)
	
    return a1, a2

def forward_softmax(x, w):
    z = w.dot(x)
    a = softmax(z)
    return a

#Activation function
def act_sigmoid(z):
    # return(1. / (1. + np.exp(-z))) # It avoids overflow and problems using integers
    return(1 / (1 + np.exp(-z)))   
    # return np.exp(np.fmin(z, 0)) / (1 + np.exp(- np.abs(z))) # to avoid overflow (as far as i seen on stackoverflow this replaces the commented above)
                                                            # and works for cases when the overflow appeared
    

# Derivate of the activation funciton
def deriva_sigmoid(a):
    # return(a * (1. - a))
    return a * (1 - a)

# STEP 2: Feed-Backward
def gradW_ae(a1, a2 , x, w1, w2, error):    

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

# Update W of the AE
def updW_ae(w1,w2,gW1,gW2,mu):
    w1 -= mu * gW1
    w2 -= mu * gW2
    return w1, w2


# Up´date W of the Softmax
def updW_softmax(w, dWs, mu):
    w = w - mu * dWs
    return w

# Softmax's gradient
def grad_softmax(x, y, w, lambW, a): 
    error = y - a 
    dZs = error.dot(x.T)
    dWs = - (dZs / y.shape[1]) + (lambW * w) # np.dot(lambW, w)
    cost = cross_entropy(y, a, lambW, w)
    # cost = categorical_cross_entropy(y, a)
    return dWs, cost


# Cross Entropy Function
def cross_entropy(y_real, y_pred, penalty, w):
    N = y_real.shape[1]
    ce = np.multiply(y_real, np.log(y_pred))
    cost = - (1 / N) * np.sum(ce)
    newCost = cost + (penalty / 2) * np.linalg.norm(w, ord=2)**2
    return newCost


def categorical_cross_entropy(y_real, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # to avoid inf
    correct_confidences = np.sum(y_pred_clipped * y_real, axis=0)
    negative_log_likelihoods = - np.log(correct_confidences)
    data_loss = np.mean(negative_log_likelihoods)
    return data_loss


# Calculate Softmax
def softmax(z):
   # axis 0  -> columns | axis 1 -> rows
   exp_values = np.exp(z - np.max(z, axis=0, keepdims=True)) # this is to prevent overflow 
   probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
   return probabilities

# Métrica
def metricas(x,y): # yv, zv # TODO Revisar!
    # cm = confusion_matrix(x,y)
    cm = confusion_matrix(x, y)
    print(cm)
    col = np.arange(1, cm.shape[0] + 1)
    col = list(col)
    diagonal = np.diag(cm)
    # print(diagonal)
    precissionVal = precission(diagonal, cm)
    recallVal = recall(diagonal, cm)
    fScoreVal = fscore(precissionVal, recallVal)
    # print(recallVal)
    # print(precissionVal)

    avgFScore = np.sum(fScoreVal) / cm.shape[0]
    # avgFscore = list(avgFScore)
    # avgCSVScore = np.zeros((1, cm.shape[0]))
    # avgCSVScore[0, 0] = avgFScore

    pd.DataFrame(cm, columns=col).to_csv('confusion.csv', index=None)

    col.extend(['Avg F-Score'])

    newData = list(fScoreVal)
    newData.extend([avgFScore])
    newData = np.array(newData)
    newData = np.expand_dims(newData, axis=0)
    # print(newData, newData.shape)
    # print(newData, len(newData))
    metrica = pd.DataFrame(data=newData, columns=col, index=['F-Score'])
    # metrica.round(3) # no funca
    metrica.to_csv(METRICS_FILE)
    return 

def precission(diagonal, cm):
    denom = np.sum(cm, axis=1)
    # print('prec: {}'.format(denom))
    value = diagonal / denom
    return value

def recall(diagonal, cm):
    denom = np.sum(cm, axis=0)
    # print('recall: {}'.format(denom))
    value = diagonal / denom
    return value

def fscore(precissionVal, recallVal):
    num = precissionVal * recallVal
    denom = precissionVal + recallVal
    division = num / denom
    value = 2 * division
    value = np.nan_to_num(value)
    return value
    
def mse(error):
    return np.power(error, 2).mean()

#Confusuon matrix
def confusion_matrix(x, y): # yv, xv
    j = np.argmax(x, axis=0)
    i = np.argmax(y, axis=0)
    cm = np.zeros((x.shape[0],x.shape[0]))
    for k in range(x.shape[1]):
        cm[i[k], j[k]] += 1
    return cm



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
    # hidden_nodes_1 = int(config_sae[0][2])
    # hidden_nodes_2 = int(config_sae[0][3])
    # hidden_nodes_3 = int(config_sae[0][4])

    # Best approach for now
    hidden_nodes = list()
    for i in range(2, len(config_sae)):
        hidden_nodes.append(int(config_sae[0][i]))
    
    # SOFTMAX CONFIG
    config_softmax = pd.read_csv(CFG_SOFTMAX_FILE, header=None)
    sftmx_max_iter = int(config_softmax[0][0])
    mu = float(config_softmax[0][1]) # this is the learning rate
    lambda_softmax = float(config_softmax[0][2]) # penalty

    # RETURNING VALUES
    # params_sae = [sae_max_iter, lr, hidden_nodes_1, hidden_nodes_2]# , hidden_nodes_3]
    params_sae = [sae_max_iter, lr, hidden_nodes]
    params_softmax = [sftmx_max_iter, mu, lambda_softmax]

    return params_sae, params_softmax


# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    # x = x.to_numpy() # maybe this added the error to our previous work
    return x

# save costo of Softmax and weights SAE 
def save_w_dl(W,Ws,cost):    
    # print(len(W))
    np.savez(DL_WEIGHT_FILE, wAE1 = W[0], 
                              wAE2 = W[1],
    #                          wAE3 = W[2],
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
        # weights['wAE3'], 
        weights['wSoftMax']] 
    weights.close()
    return (W)    