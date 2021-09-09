# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut
import numpy as np

def main():
    xv, yv = ut.load_data('test.csv')
    X_test = xv.T
    y_test = yv.T
    np.random.shuffle(X_test) # in train gives better results but it keeps sending bad values here (at least this are 3 out of 6)
    np.random.shuffle(y_test)
    w1, w2 = ut.load_w('w_snn.npz')
    _, zv = ut.forward(X_test, w1, w2)
    ut.metrica(zv, y_test)
    # print(r2)

if __name__ == '__main__':
    main()