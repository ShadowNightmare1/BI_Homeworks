# Mauricio Abarca J.
# 19.319.550-4

# Training SAE via SGD with Momentum


import utilities as ut
import pandas as pd	
import numpy as np

	
# Softmax's training
def train_softmax(x, y, param):
    W, V = ut.iniWs(x.shape[0], y.shape[0])
    costo = []

    for iter in range(param[0]):        
        idx = np.random.permutation(x.shape[1])
        xe  = x[:, idx]
        ye = y[:, idx] # nope
        W, V, cost = softmax_batch(xe, ye, W, V, [ut.SOFTMAX_BATCH_SIZE, param[1]])              
        costo.append(cost)

        # Epoch log
        if iter % 100 == 0:
            print('Epoch {} | Cross-Entropy Loss: {}'.format(iter, cost))     
    return W, costo


# Training Softmax miniBatch SGD
def softmax_batch(x, y, W, V, param):
    numBatch = np.int16(np.floor(x.shape[1] / param[0]))    
    
    for i in range(numBatch):
        X = ut.get_miniBatch(i, x, param[0])
        Y = ut.get_miniBatch(i, y, param[0])

        # Forward Pass Batch
        A1 = ut.forward_softmax(X, W)

        # Backward Pass
        dWs, loss = ut.gradW_softmax(X, Y, A1)
        W, V = ut.updW_sft_sgd(W, V, dWs, param[1])
    

    return W, V, loss

# Training AE miniBatch SGD
def train_batch(x, W, V, param):
    numBatch = np.int16(np.floor(x.shape[1] / param[0]))    

    for i in range(numBatch):                
        X = ut.get_miniBatch(i, x, param[0])
        W2 = ut.iniW(W.shape[1], W.shape[0])
        
        # Forward Pass Batch
        A1 = ut.forward_ae(X, W)
        A2 = ut.forward_ae(A1, W2)

        # Calculate Cost Batch
        error = A2 - X
        mse = ut.mse(error)

        # Backward Pass Batch
        dW1, _ = ut.gradW_ae(A1, A2, X, W2, error)
        W, V = ut.updW_ae_sgd(W, V, dW1, param[2])

    return W, V, mse


#Training AE by use SGD
def train_ae(x, hn, param):    
    W, V    = ut.iniWs(x.shape[0], hn)            
    for Iter in range(1, param[1]):        
        xe  = x[:, np.random.permutation(x.shape[1])]        
        W, V, cost = train_batch(xe, W, V, param)
        A = ut.forward_ae(xe, W)

        if Iter % 10 == 0:
            print('Epoch: {} | MSE (last batch value): {}'.format(Iter, cost))
        
    return W, A


#SAE's Training 
def train_sae(x,param):    
    W = list()
    hidden_nodes_layers = param[-1]
    batch_size = param[0]
    max_iter = param[1]
    lr = param[2]
    for hn in range(len(hidden_nodes_layers)):
        print('AE={} Hnode={}'.format(hn + 1, hidden_nodes_layers[hn]))

        if hn == 0:
            input = x

        else:
            input = a

        w1, a = train_ae(input, hidden_nodes_layers[hn], [batch_size, max_iter, lr])
        W.append(w1)

    return W, a


# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()    
    xe          = ut.load_data_csv('train_x.csv')
    ye          = ut.load_data_csv('train_y.csv')
    W, Xr       = train_sae(xe, p_sae)     
    Ws, cost    = train_softmax(Xr, ye, p_sft)
    ut.save_w_dl(W, Ws, cost)
       

if __name__ == '__main__':   
	 main()
