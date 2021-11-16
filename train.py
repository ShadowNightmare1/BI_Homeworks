# Mauricio Abarca J.
# 19.319.550-4

#Training SAE via RMSprop+Pseudo-inversa

import utilities as ut
import pandas as pd	
import numpy as np


# Minibatch Function
def get_miniBatch(i, x, bsize):
    z = x[:, i * bsize : ( i + 1) * bsize]
    return z


# Random Permutation / Reordering
def reordena_rand(x, y):
    idx = np.random.permutation(x.shape[1])
    new_x = x[:, idx]
    new_y = y[:, idx]
    return new_x, new_y


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, numBatch, BatchSize, mu):
    costo = []    
    for i in range(numBatch):   
        xe, ye = get_miniBatch(...)        
        #complete code...        
    return W, V, costo


# Softmax's training via RMSprop
def train_softmax(x, y, par1, par2):
    W, V        = ut.ini_WV(y.shape[0], x.shape[0])    
    numBatch   = np.int16(np.floor(x.shape[1] / par2[0]))    
    Costo = []
    for Iter in range(1, par1[0]):                
        xe, ye = reordena_rand(x, y)         
        W, V, c = train_sft_batch(xe, ye, W, V, numBatch, par2[0], par1[1])        
        Costo.append(np.mean(c))         
    return W, Costo

 
# AE's Training with miniBatch
def train_ae_batch(x, w1, v, w2, param):
    numBatch = np.int16(np.floor(x.shape[1] / param[1]))    
    cost= [] 
    for i in range(numBatch):                
        X    = get_miniBatch(i, x, param[1])
        
        # Forward Pass batch
        A1 = ut.forward_ae(X, w1)
        A2 = ut.forward_ae(A1, w2)

        # Calculate MSE
        error = A2 - X
        mse = ut.mse(error)

        # Backward Pass Batch
        w2 = ut.pinv_ae(X, w1, param[0])
        dW1, costo = ut.gradW1(A1, w2)
        cost.append(costo)

        w1, v = ut.updW1_rmsprop(w, v, dW1, param[3])
        
    return w1, v, cost


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, hn, param):        
    w1, v = ut.ini_WV(param[hn], x.shape[0])            
    w2   = ut.pinv_ae(x, w1, param[3]) # que hace y como funciona ?
    cost = []
    for Iter in range(1, param[1]):        
        xe     = x[:, np.random.permutation(x.shape[1])]                
        w1, v, c = train_ae_batch(xe, w1, v, w2, param)                  
        cost.append(np.mean(c))                

        if Iter % 10 == 0:
            print('Epoch: {} | MSE (mean of the batch): {}'.format(Iter, np.mean(c)))
    return w2.T 


#SAE's Training 
def train_sae(x, param):
    # param: 0 -> penalty 
    #        1 -> batch size
    #        2 -> max iter
    #        3 -> learning rate
    #        4: -> hidden nodes by layer

    W = {} # Diccionario?

    for hn in range(len(param[4])):
        print('AE={} Hnode={}'.format(hn + 1, param[4][hn]))

        if hn == 0:
            input = x
        
        else:
            input = a
        w2, a = train_ae(input, hn, param)  
    return W, x 


# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()    
    x, y         = ut.load_data_csv('train.csv')    
    W, Xr        = train_sae(x, p_sae)         
    # Ws, cost    = train_softmax(Xr, y, p_sft, p_sae)
    # ut.save_w_dl(W, Ws, cost)
       
if __name__ == '__main__':   
	 main()
