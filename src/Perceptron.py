import numpy as np
import pandas as pd

class Perceptron:
    """
    A simple implementation of a single-layer Perceptron for binary and multiclass classification.
    """
    def __init__(self, input_size: int, output_size: int = 1, learning_rate: float = 0.01, epochs: int = 1000, activation: str = "step", loss: str = "mse", random_seed: int = None) -> None:
        """
        Initializes the perceptron with random coefficients (weights and biases combined).

        Parameters
        ----------
        input_size : int
            Number of features in the input.
        output_size : int, optional
            Number of output classes (default is 1 for binary classification).
        learning_rate : float, optional
            Step size for weight updates, by default 0.01.
        epochs : int, optional
            Number of iterations for training, by default 1000.
        activation : str, optional
            Activation function to use ("step", "sigmoid", or "softmax"), by default "step".
        loss : str, optional
            Loss function to use ("mse" or "binary_cross_entropy"), by default "mse".
        """
        self.input_size = input_size + 1  # Adding one for the bias term
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation
        self.loss_function = loss
        if random_seed is not None:
            np.random.seed(random_seed)
        self.coeffs = np.random.randn(self.input_size, output_size)
        self.y_prob = None  # Stores predicted probabilities
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the perceptron using the specified loss function and activation function.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features.
        y : np.ndarray or pd.Series
            Target labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        y = np.eye(self.output_size)[y] if self.output_size > 1 else y.reshape(-1, 1)
        
        for epoch in range(self.epochs):
            linear_output = np.dot(X, self.coeffs)
            predictions = self._activate(linear_output)
            loss_gradient = self._compute_loss_derivative(y, predictions)
            
            self.coeffs -= self.learning_rate * np.dot(X.T, loss_gradient)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on input data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        linear_output = np.dot(X, self.coeffs)
        self.y_prob = self._activate(linear_output)
        
        if self.activation_function == "softmax":
            return np.argmax(self.y_prob, axis=1)
        elif self.activation_function == "sigmoid":
            return (self.y_prob >= 0.5).astype(int)
        else:
            return self.y_prob
    
    def _step_activation(self, x: np.ndarray) -> np.ndarray:
        """Applies step activation function."""
        return np.where(x >= 0, 1, 0)
    
    def _sigmoid_activation(self, x: np.ndarray) -> np.ndarray:
        """Applies sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def _softmax_activation(self, x: np.ndarray) -> np.ndarray:
        """Applies softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Selects and applies the appropriate activation function."""
        if self.activation_function == "step":
            return self._step_activation(x)
        elif self.activation_function == "sigmoid":
            return self._sigmoid_activation(x)
        elif self.activation_function == "softmax":
            return self._softmax_activation(x)
        else:
            raise ValueError("Invalid activation function. Choose from 'step', 'sigmoid', or 'softmax'.")
    
    def _mean_squared_error_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes derivative of mean squared error loss."""
        return (y_pred - y_true) / y_true.shape[0]
    
    def _binary_cross_entropy_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes derivative of binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return (y_pred - y_true) / y_true.shape[0]
    
    def _categorical_cross_entropy_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes derivative of categorical cross-entropy loss."""
        return (y_pred - y_true) / y_true.shape[0]
    
    def _compute_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Selects and computes the appropriate loss derivative."""
        if self.loss_function == "mse":
            return self._mean_squared_error_derivative(y_true, y_pred)
        elif self.loss_function == "binary_cross_entropy":
            return self._binary_cross_entropy_derivative(y_true, y_pred)
        elif self.loss_function == "categorical_cross_entropy":
            return self._categorical_cross_entropy_derivative(y_true, y_pred)
        else:
            raise ValueError("Invalid loss function. Choose from 'mse', 'binary_cross_entropy', or 'categorical_cross_entropy'.")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluates model accuracy.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features.
        y : np.ndarray or pd.Series
            True labels.

        Returns
        -------
        float
            Accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions.flatten() == y)

