# Mauricio Abarca J.
# 19.319.550-4

import train
import test

if __name__ == '__main__':
    print("Init Routine")
    train.main()
    # we could put a system wait here
    test.main()
    print("End of Routine")

    # 'reglas' para las neuronas ocultas: 
    # 1) N° neuronas ocultas debiera estar entre el tamaño del input y el del output (en este caso entre 5 y 1)
    # 2) N° neuronas debiera ser 2/3 el tamaño de la capa de entrada + el tamaño de la capa de salida
    # 3) N° neuronas no debiera ser más del doble del tamaño de la capa de entrada

    # 20,50000, 0.01 -> test -0.1