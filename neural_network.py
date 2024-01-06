import numpy as np
from typing import List


class NeuronLayer:
    """
    Class NeuronLayer. Represents a single layer of a neural network.

    Attributes:
    ----------
    weights : numpy.ndarray
        Weights of the layer.
    bias : numpy.ndarray
        Bias of the layer.
    input : numpy.ndarray
        Input of the layer.
    output : numpy.ndarray
        Output of the layer.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Parameters:
        ----------
        input_size : int
            Size of the input.
        output_size : int
            Size of the output.
        """

        self._weights = np.random.randn(input_size, output_size)
        self._bias = np.random.randn(1, output_size)

        self._input = None
        self._output = None

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @property
    def input(self) -> np.ndarray:
        return self._input

    @property
    def output(self) -> np.ndarray:
        return self._output

    def forwardPropagation(self, input: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation of the layer.

        Parameters:
        ----------
        input : numpy.ndarray
            Input of the layer.

        Returns:
        -------
        numpy.ndarray
            Output of the layer.
        """

        self._input = input
        self._output = self._sigmoid(np.dot(self.input, self.weights) + self.bias)

        return self.output

    def backwardPropagation(self, dE_dY: np.ndarray, alfa: float) -> np.ndarray:
        """
        Performs backward propagation of the layer.

        Parameters:
        ----------
        dE_dY : numpy.ndarray
            Gradient of the error with respect to the output.
        alfa : float
            Learning rate.

        Returns:
        -------
        numpy.ndarray
            Gradient of the error with respect to the input.
        """

        dE_dX = np.dot(dE_dY, self.weights.T)
        dE_dW = np.dot(self.input.T, dE_dY)
        self._weights -= alfa * dE_dW
        self._bias -= alfa * dE_dY

        return self._sigmoidDerivative(self.input) * dE_dX

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _sigmoidDerivative(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x) * (1 - self._sigmoid(x))


class NeuralNetwork:
    """
    Class NeuralNetwork. Represents a neural network.

    Attributes:
    ----------
    layers : list[NeuronLayer]
        List of layers of the neural network.
    errors : list[float]
        List of the training errors.
    """

    def __init__(self, layers: List[NeuronLayer]) -> None:
        """
        Parameters:
        ----------
        layers : list[NeuronLayer]
            List of layers of the neural network.
        """

        self._layers = layers
        self._errors = None

    @property
    def layers(self) -> List[NeuronLayer]:
        return self._layers

    @property
    def errors(self) -> List[float]:
        return self._errors

    def predict(self, X: np.ndarray) -> List[int]:
        """
        Predicts the output for the given input data.

        Parameters:
        ----------
        X : numpy.ndarray
            Input data.

        Returns:
        -------
        list[int]
            List of the predicted values.
        """

        samples = X.shape[0]
        result = []

        for i in range(samples):
            output = X[i]

            for layer in self.layers:
                output = layer.forwardPropagation(output)

            result.append(np.argmax(output))

        return result

    def train(
        self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100, alfa=0.1
    ) -> None:
        """
        Trains the neural network.

        Parameters:
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output data.
        n_epochs : int
            Number of epochs.
        alfa : float
            Learning rate.

        Returns:
        -------
        None
        """

        n_samples = X.shape[0]
        self._errors = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            cost = 0

            for i in range(n_samples):
                output = X[i]

                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                cost += self._cost(y[i], output)
                dE_dY = self._costDerivative(y[i], output)

                for layer in reversed(self.layers):
                    dE_dY = layer.backwardPropagation(dE_dY, alfa)

            self.errors[epoch] = cost / n_samples

    def _cost(self, y: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean((y - prediction) ** 2)

    def _costDerivative(self, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return 2 * (prediction - y) / y.size
