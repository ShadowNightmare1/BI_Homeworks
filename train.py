# Mauricio Abarca J.
# 19.319.550-4

#Training SAE via RMSprop+Pseudo-inversa

import utilities as ut
import numpy as np


# Minibatch Function
def get_miniBatch(i, x, bsize):
    z = x[:, i * bsize : ( i + 1) * bsize]
    return z


# Random Permutation / Reordering
def reordena_rand(x, y):
    # print(x.shape)
    # print(y.shape)
    idx = np.random.permutation(x.shape[1])
    new_x = x[:, idx]
    new_y = y[:, idx]
    return new_x, new_y


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, numBatch, BatchSize, mu):
    costo = []    
    for i in range(numBatch):   
        X = get_miniBatch(i, x, BatchSize)        
        Y = get_miniBatch(i, y, BatchSize)

        # Forward Pass Batch
        A = ut.forward_softmax(X, W)

        # Backward Pass
        dWs, loss = ut.gradW_softmax(X, Y, A, BatchSize)
        costo.append(loss)
        W, V = ut.updW_sft_rmsprop(W, V, dWs, mu)
        
    return W, V, costo


# Softmax's training via RMSprop
def train_softmax(x, y, par1, par2):
    W, V       = ut.ini_WV(y.shape[0],x.shape[0])    
    numBatch   = np.int16(np.floor(x.shape[1] / par2[1]))    
    Costo = []
    for Iter in range(1, par1[0]):                
        xe,ye = reordena_rand(x, y)         
        W, V, c = train_sft_batch(xe, ye, W, V, numBatch, par2[1], par1[1])        
        Costo.append(np.mean(c))         

        if Iter % 10 == 0:
            print('Softmax Epoch: {} | Batch Mean Cost: {}'.format(Iter, np.mean(c)))
            
    return W, Costo

 
# AE's Training with miniBatch
def train_ae_batch(x, w1, v, w2, param):
    numBatch = np.int16(np.floor(x.shape[1] / param[1]))    
    cost= [] 
    for i in range(numBatch):                
        X    = get_miniBatch(i, x, param[1])
        
        # Forward Pass Batch
        A1 = ut.forward_ae(X, w1)
        A2 = ut.forward_ae(A1, w2)

        # Calculate Error
        error = A2 - X
        inner_val = np.power(error, 2)
        outer_val = 1 / (2 * param[1])
        first_sum = np.sum(inner_val, axis=0, keepdims=True)
        second_sum = np.sum(first_sum, axis=1)
        costo = outer_val * second_sum
        cost.append(costo)
        
        # Backward Pass Batch
        # w2 = ut.pinv_ae(X, w1, param[0])
        # dW2 = ut.gradW1()
        dW1 = ut.gradW1(A1, w2, error, X)
        w1, v = ut.updW1_rmsprop(w1, v, dW1, param[3])
        
    return w1, v, cost


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, hn, param):        
    w1, v = ut.ini_WV(param[4][hn], x.shape[0])            
    w2   = ut.iniW(x.shape[0], param[4][hn])
    cost = []
    
    for Iter in range(1, param[2]):        
        xe     = x[:, np.random.permutation(x.shape[1])]
        w1, v, c = train_ae_batch(xe, w1, v, w2, param)                  
        # A = ut.forward_ae(xe, w1)
        cost.append(np.mean(c))                

        if Iter % 10 == 0:
            print('Epoch: {} | Batch Mean Cost: {}'.format(Iter, np.mean(c)))
    
    return w2.T# , A


#SAE's Training 
def train_sae(x, param):
    # param: 0      -> penalty 
    #        1      -> batch size
    #        2      -> max iter
    #        3      -> learning rate
    #        4 ...  -> hidden nodes by layer

    W = {} 
    
    for hn in range(len(param[4])):
        print('AE={} Hnode={}'.format(hn + 1, param[4][hn]))

        w1 = train_ae(x, hn, param)  
        W.update({'WAE' + str(hn + 1): w1})
    A = ut.forward_ae(x, w1) 

    # así como está ahora da un costo de softmax de 0.2 (buenardo)
    # pero se cae en el test, revisar para ver si conseguimos mejorar esta cosa

    return W, A


# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()    
    x, y         = ut.load_data_csv('train.csv')    
    W, Xr        = train_sae(x, p_sae)         
    print('Xr: {}'.format(Xr.shape)) 
    Ws, cost    = train_softmax(Xr, y, p_sft, p_sae)
    ut.save_w_dl(W, Ws, cost)
       
if __name__ == '__main__':   
	main()
    
    
    
    