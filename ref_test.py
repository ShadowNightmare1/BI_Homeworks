# Mauricio Abarca J.
# 19.319.550-4

import ref_utilities as ut

def main():
    xv, yv = ut.load_data('test.csv')
    X_test = xv.T
    y_test = yv.T
    w1, w2 = ut.load_w('w_snn.npz')
    _, zv = ut.forward(X_test, w1, w2)
    ut.metrica(y_test, zv)

if __name__ == '__main__':
    main()