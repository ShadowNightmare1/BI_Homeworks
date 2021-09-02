# Mauricio Abarca J.
# 19.319.550-4

import utilities as ut

def main():
    xv, yv = ut.load_data('test.csv')
    X_test = xv.T
    y_test = yv.T
    w1, w2 = ut.load_w('w_snn.npz')
    _, zv = ut.forward(X_test, w1, w2)
    r2 = ut.r2(zv, y_test)
    ut.metrica(zv, y_test)
    print(r2)

if __name__ == '__main__':
    main()