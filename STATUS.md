# STATUS Tarea 3 (branch tarea4)

- Funciona, pero no arroja una matriz de confusión correcta (Sabemos que el algoritmo de métricas funciona, por testing previos)
- train_ae entrega valores correctos de w.shape
- train_softmax entrega valores correctos de w.shape

# POR HACER
- Con los arreglos del profe (dejar de calcular pinv para w2, y dejar de devolver A en el batch), pasamos a tener un costo softmax de 0.2 y fracción, pero se cae en el test por tamaño de matrices, habrá que ver que onda

# OJO
- lr en autoencoder -> 0.01 (mejor resultado al final del AE)