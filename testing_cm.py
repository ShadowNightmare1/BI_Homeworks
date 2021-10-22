import pandas as pd
import utilities as ut
import numpy as np

# zv, yv = ut.loadTest()

# print(zv.shape)

# metrica = ut.metricas(yv, zv)

zv = pd.read_csv('test_zv.csv', header=None)
zv = np.array(zv)

yv = pd.read_csv('test_y.csv', header=None)
yv = np.array(yv)

cm = ut.confusion_matrix(yv, zv)
print(cm)

