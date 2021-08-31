# Mauricio Abarca J.
# 19.319.550-4

from utilities import *

def main():
    # Load Testing Data
    X, y = loadData(TEST_FILE)
    n_samples, n_caract = X.shape
    X_test = X.T
    y_test = y.T

    # Create First Hidden Layer
    dense1 = DenseLayer(hidden_nodes, n_caract)
    activation1 = ActivationLayer()

    # Load Weights for Hidden Layer
    weight1 = weightsFromCSV(DEV_WEIGHT_1)
    dense1.loadWeights(weight1)

    # Create Output Layer
    dense2 = DenseLayer(output_nodes, hidden_nodes)
    activation2 = ActivationLayer()

    # Load Weights for Output Layer
    weight2 = weightsFromCSV(DEV_WEIGHT_2)
    dense2.loadWeights(weight2)

    # Create Metrics object for the model
    metrics = Metrics()

    # Forward Pass Hidden Layer
    dense1.forward(X_test)
    activation1.sigmoid(dense1.output)

    # Forward Pass Output Layer
    dense2.forward(activation1.output)
    activation2.sigmoid(dense2.output)

    # Calculate Metrics
    metrics.metricsTest(y_test, activation2.output)

    print("TEST | MAE: {}".format(metrics.mae))
    metrics.exportMetrics()

if __name__ == '__main__':
    main()