# Mauricio Abarca J.
# 19.319.550-4
# Aim : Deep-Learning: Training via BP+GD

import utilities as ut
import pandas as pd	
import numpy as np


# Softmax's training
def train_softmax(x,y,param): # i'm afraid this could be wrong
    w = ut.iniW(x.shape[0], y.shape[0])
    costo = []
    print('Softmax  | Max iter: {}'.format(param[0]))
    for iter in range(param[0]):        
        # Forward Pass
        a = ut.forward_softmax(x, w)

        # Calculate Cost
        # crossEntropy = ut.cross_entropy(y, a, param[2], w)
        # costo.append(crossEntropy)
        error = y - a
        # error = a - y
        mseVal = ut.mse(error)

        # Backward Pass
        dWs, loss = ut.grad_softmax(x, y, w, param[2], a)
        costo.append(loss)
        w = ut.updW_softmax(w, dWs, param[1])

        
        # Epoch log
        if iter % 10 == 0:
            print('Epoch {} | MSE: {} | Cross-Entropy Loss: {}'.format(iter, mseVal,loss))

    pd.DataFrame(data=a).to_csv('act_softmax.csv', header=None, index=None)   
    ut.metricas(a, y)
    return w, costo


# AE's Training 
def train_ae(x,hnode,MaxIter,mu):
    w1 = ut.iniW(x.shape[0], hnode[0])            
    w2 = ut.iniW(hnode[0], x.shape[0])
    # cost = list()

    for iter in range(MaxIter):
        # Forward Pass
        a1, a2 = ut.forward_ae(x, w1, w2)

        # Calculate Cost
        error = a2 - x 
        # error = x - a2
        mse = ut.mse(error)
        # cost.append(mse)

        # Backward Pass
        dW1, dW2 = ut.gradW_ae(a1, a2, x, w1, w2, error)
        w1, w2 = ut.updW_ae(w1, w2, dW1, dW2, mu)

        # Epoch Log
        if iter % 10 == 0:
            print('Epoch: {} | MSE: {}'.format(iter, mse))

    return w1, a1


#SAE's Training 
def train_sae(x,param):
    W = list()
    
    for hn in range(2,len(param)):
        print('AE={} Hnode={}'.format(hn - 1,param[hn]))
        if hn + 1 >= len(param):
            next_hn = ut.CODE
        else:
            next_hn = param[hn + 1]

        if hn - 1 == 1:
            input = x

        else:
            input = a

        w1, a = train_ae(input, [param[hn], next_hn], param[0], param[1])
        W.append(w1)

    return W, a                        
    
   
# Beginning ...
def main():
    param_sae, param_sft = ut.load_config()    
    xe = ut.load_data_csv('train_x.csv')
    ye = ut.load_data_csv('train_y.csv')
    # np.random.shuffle(xe)
    # np.random.shuffle(ye)
    W, Xr = train_sae(xe,param_sae) 
    Ws, cost = train_softmax(Xr,ye,param_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()
