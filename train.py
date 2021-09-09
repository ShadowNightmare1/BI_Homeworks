# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut
import numpy as np


# Training of SNN

def train_snn(X, y, param):
    _, n_features = X.shape
    hidden_nodes = param[0]
    lr = param[2]
    X_train = X.T
    y_train = y.T

    np.random.shuffle(X_train) # ok, it gives better results with the shuffle here
    np.random.shuffle(y_train)

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
        r2 = ut.r2(a2, y_train)

        if iter % 100 == 0:
            print('Epoch: {} | R2: {} | MSE: {}'.format(iter, r2, mse))
                  
    return (w1, w2, cost)

# Beginning ...

def main():
    par_snn = ut.load_config('config.csv')
    xe, ye = ut.load_data('train.csv')
    # print(xe.min()) # tanto min como max de xe e ye dan los valores correspondientes (a y b de la normalizacion)
    w1, w2, cost = train_snn(xe, ye, par_snn)
    ut.save_w(w1, w2, 'w_snn.npz', cost, 'costo.csv')


if __name__ == '__main__':
    main()
