import numpy as np
import pandas as pd

class GDSVC:
    """
    Performs Support Vector Classification using gradient descent

    Parameters
    ----------
    C: float, default = 0
        Regularization coefficient

    learning_rate: float, default: 0.001
        learning rate for the gradient descent

    max_iter: float, default = 1000
        maximum number of iterations for the gradient descent

    tol: float, default = 1e-5
        Convergence criterion for gradient descent

    random_seed: int, default = None
        Random seed for random initialization

    Attributes
    ----------
    weights: np.ndarray
        Model weights

    bias: float
        Model bias

    """
    
    def __init__(self, C: float = 0, learning_rate: float = 0.001, max_iter: float = 1000, tol: float = 1e-5, random_seed: int = None) -> None:
        self.C = C
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.weights = None
        self.bias = None

    def fit(self, X, y) -> None:
        """
        Compute weights and biases for the linear SVC

        Parameters
        -----------
        X: np.ndarray
            Training data
        y: np.ndarray
            Training Targets
        """
        # Check if input is in pandas format and transform to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        n_samples, n_features = X.shape
        # sign variable (-1 when y = 0 and 1 when y = 1)
        y_sign = np.where(y==0,-1,1)
        #Initialize weights and biases
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = np.random.normal(loc=1, size=n_features)
        self.bias = 0
        
        # Gradient Descent loop
        iter_count = 0
        
        while (iter_count < self.max_iter):
            constraint, loss = self._hinge_loss(X, y_sign, self.C, self.weights, self.bias)
            if loss < self.tol:
                print("Gradient descent converged")
                break
            # Compute derivatives
            mask = (constraint>0).astype(int)
            w_grad = self.weights + self.C*np.dot(-mask*y_sign,X)
            b_grad = self.C*np.dot(mask,-y_sign)
            # Update weights and bias
            self.weights -= self.lr*w_grad
            self.bias -= self.lr*b_grad
            
            iter_count += 1
    
    def _hinge_loss(self, X, y_sign, C, w, b):
        """
        Compute hinge loss

        Parameters
        ----------
        X: np.ndarray
            Inpute data
        
        y_sign: np.ndarray
            Binary labels in form of -1 and 1
        
        C: float
            Regularization factor

        w: np.ndarray
            model weights

        b: float
            model bias

        Returns
        --------
        A tuple of soft-margin constraint and linear SVC loss/cost function

        """
        constraint = 1-y_sign*(np.dot(X,w)+b)
        return (constraint, 0.5*np.dot(w,w) + C*np.dot((constraint>0),constraint))
        
        

    def predict(self, X):
        """
        Perform classification

        Parameters
        ----------
        X: np.ndarray
            Test data

        Returns
        -------
        Predicted labels

        """
        return ((np.dot(X,self.weights)+self.bias)>0).astype(int)

