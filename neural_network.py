import numpy as np
from typing import List


class Layer:
    """
    Class Layer. Represents a single layer of a neural network.

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
        self._weights = np.zeros((input_size, output_size))
        self._bias = np.zeros((1, output_size))

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

    def forwardPropagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation of the layer.

        Parameters:
        ----------
        input_data : numpy.ndarray
            Input of the layer.

        Returns:
        -------
        numpy.ndarray
            Output of the layer.
        """

        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = self._sigmoid(self.output)

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
        self.weights -= alfa * dE_dW
        self.bias -= alfa * dE_dY

        return self._dSigmoid(dE_dX) * dE_dY

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _sigmoidDerivative(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x) * (1 - self._sigmoid(x))


class NeuralNetwork:
    """
    Class NeuralNetwork. Represents a neural network.

    Attributes:
    ----------
    layers : list[Layer]
        List of layers of the neural network.
    errors : list[float]
        List of the training errors.
    """

    def __init__(self, layers: List[Layer]) -> None:
        self._layers = layers
        self._errors = None

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def errors(self) -> List[float]:
        return self._errors

    def predict(self, input_data: np.ndarray) -> List[int]:
        """
        Predicts the output for the given input data.

        Parameters:
        ----------
        input_data : numpy.ndarray
            Input data.

        Returns:
        -------
        list[int]
            List of the predicted values.
        """

        samples = input_data.shape[0]
        result = []

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forwardPropagation(output)

            result.append(np.argmax(output) + 3)

        return result

    def train(
        self, x: np.ndarray, y: np.ndarray, n_epochs: int = 100, alfa=0.1
    ) -> None:
        """
        Trains the neural network.

        Parameters:
        ----------
        x : numpy.ndarray
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

        n_samples = x.shape[0]
        self.errors = np.zeros(n_epochs)

        for _ in range(n_epochs):
            cost = 0

            for i in range(n_samples):
                output = x[i]

                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                cost += self._cost(y[i], output)
                dE_dY = self._costDerivative(y[i], output)

                for layer in reversed(self.layers):
                    dE_dY = layer.backwardPropagation(dE_dY, alfa)

            self.errors[i] = cost / n_samples

    def _cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def _costDerivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
