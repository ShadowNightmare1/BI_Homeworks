# Mauricio Abarca J.
# 19.319.550-4


# SNN: Training via BackPropagation (Pseudocodigo del profe)

import pandas as pd
import numpy as np
# import utilities as ut
import ref_utilities as ut


# Training of SNN

def train_snn(X, y, param):
    _, n_features = X.shape
    hidden_nodes = param[0]
    lr = param[2]
    X_train = X.T
    y_train = y.T

    w1,w2 = ut.iniW(hidden_nodes, n_features, ut.OUTPUT_NODES)
    cost = []

    
    for iter in range(param[1]):
        # Forward Pass
        a1, a2 = ut.forward(X_train, w1, w2)

        # Calculate Cost
        error = a2 - y_train
        mse = ut.mse(error)
        cost.append(mse)

        # Backward Pass
        w1, w2 = ut.grad_bp(a1, a2, X_train, error, w1, w2, lr)
        
        # Epoch Log
        r2 = ut.r2(y_train, a2)
        if iter % 10 == 0:
            print('Epoch: {} | R2: {}'.format(iter, r2))

    
    return (w1, w2, cost)

# Beginning ...

def main():
    par_snn = ut.load_config('config.csv')
    xe, ye = ut.load_data('train.csv')
    w1, w2, cost = train_snn(xe, ye, par_snn)
    ut.save_w(w1, w2, 'w_snn.npz', cost, 'costo.csv')
    # pd.DataFrame(data=w1).to_csv('peso1.csv', header=None, index=None)    
    # pd.DataFrame(data=w2).to_csv('peso2.csv', header=None, index=None)    

if __name__ == '__main__':
    main()