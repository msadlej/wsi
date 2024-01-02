import numpy as np


class Layer:
    """
    Class Layer.
    """

    def __init__(self, input_size, output_size):
        self._weights = np.zeros((input_size, output_size))
        self._bias = np.zeros((1, output_size))

        self._input = None
        self._output = None

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    def forwardPropagation(self, input_data):
        self.input = input_data

        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self._sigmoid(self.output)
        return self.output

    def backwardPropagation(self, dE_dY, alfa):
        dE_dX = np.dot(dE_dY, self.weights.T)
        dE_dW = np.dot(self.input.T, dE_dY)
        self.weights -= alfa * dE_dW
        self.bias -= alfa * dE_dY

        return self._dSigmoid(dE_dX) * dE_dY

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoidDerivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))


class NeuralNetwork:
    """
    Class NeuralNetwork.
    """

    def __init__(self, layers):
        self._layers = layers
        self._errors = []

    @property
    def layers(self):
        return self._layers

    @property
    def errors(self):
        return self._errors

    def predict(self, input_data):
        samples = input_data.shape[0]
        result = []

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forwardPropagation(output)

            result.append(np.argmax(output) + 3)

        return result

    def train(self, x, y, n_epochs, alfa=0.1):
        samples = x.shape[0]
        self.err_log = np.zeros(n_epochs)

        for _ in range(n_epochs):
            loss = 0

            for i in range(samples):
                output = x[i]

                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                loss += self._loss(y[i], output)
                dE_dY = self._lossDerivative(y[i], output)

                for layer in reversed(self.layers):
                    dE_dY = layer.backwardPropagation(dE_dY, alfa)

            loss /= samples
            self.errors[i] = loss

    def _loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _lossDerivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
