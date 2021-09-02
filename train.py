# Mauricio Abarca J.
# 19.319.550-4

from utilities import *


def main():

    # Load Training Data 
    X, y = loadData(TRAIN_FILE)
    n_samples, n_caract = X.shape
    X_train = X.T
    y_train = y.T

    
    # Create First Hidden Layer
    dense1 = DenseLayer(hidden_nodes, n_caract)
    activation1 = ActivationLayer()

    # Create Output Layer
    dense2 = DenseLayer(output_nodes, hidden_nodes)
    activation2 = ActivationLayer()

    # Create Metrics object for the model
    metrics = Metrics()

    # Start the training epochs
    for i in range(iterations):

        # Forward Pass Hidden Layer
        dense1.forward(X_train) 
        activation1.sigmoid(dense1.output) # => A1

        # Forward Pass Output Layer
        dense2.forward(activation1.output)
        activation2.sigmoid(dense2.output) # => A2

        # Calculate metrics
        metrics.metrics(y_train, activation2.output, i)

        # Backward Pass Output Layer
        dE2 = metrics.error
        dense2.backward(dE2, activation2.sigmoidPrime, activation1.output)

    
        # Backward Pass Hidden Layer
        # dE1 = np.dot(dense2.weights.T, dense2.dZ)
        dE1 = dense2.weights.T @ dense2.dZ
        dense1.backward(dE1, activation1.sigmoidPrime, X_train)

        if (i % 10 == 0):
            print("Epoch: {} | MAE: {:.12f}".format(i + 1, metrics.mae))
        
    

    # Export the Loss of the training cycle
    metrics.exportLoss()
    dense1.saveWeights(1)
    dense2.saveWeights(2)

    pd.DataFrame(data=[dense1.weights, dense2.weights]).to_csv('weights.csv', header=None, index=None)
    # export weights 1 and 2.
   

if __name__ == '__main__':
    main()
    # x = np.array([[1, 2], 
    #               [3, 4]])
# 
    # y = np.array([[1, 2], 
    #               [3, 4]])
# 
    # print(np.multiply(x, y))
    # print(x.dot(y))