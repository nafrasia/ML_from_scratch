import numpy as np
import numpy.linalg as LA

class LinearRegression:
    '''
        Performs Linear Regression

        LinearRegression fits a linear model with coefficients b = (b1, …, bp) to minimize the residual sum of squares between 
        the actual label values in the dataset, and the predicted values by the linear approximation.

        Parameters
        ------------
        optimizer: string, default: least_squares
            Determines the method used for fitting

        learning_rate: float, default: 0.001
            The learning rate for the gradient descent method

        maxIter: float, default: 5000
            Maximum number iteration for the gradient descent method

        tolerance: float, default: 1e-8
            The criteron for convergence of teh grdient descent method

        Attributes
        -----------
        coeffs: np.array of floats
            Linear equation coefficients
        
        n_features: int
            Number of features in the dataset

        n_points: int
            Number of data points in the dataset

        lr: float
            Learning rate for the gradient descent method

        tol: float
            Tolerance for convergence of gradient descent
    '''

    def __init__(self, optimizer='least_squares', learning_rate=0.001, maxIter=5000, tolerance=10**(-8)):
        self.optimizer = optimizer
        self.coeffs = None
        self.n_features = None
        self.n_points = None
        self.lr = learning_rate
        self.maxIter = maxIter
        self.tol = tolerance
        self.RSS = None
        self.R2 = None

    def fit(self, X, y):
    '''

            Fit the linear model using the provided training data
            
            If least squares is picked as the optimizer, the
            numpy.linalg.lstsq() function is used to fit the data

            Parameters
            ----------
                X : numpy.ndarray
                    Training data features, shape (n_samples, n_features)
                y : numpy.ndarray
                    Training data labels, shape (n_samples,)

    '''
        
        Xfull = self._add_ones(X)
        
        if self.optimizer == 'least_squares':
            self.coeffs, self.RSS, rank, singular = LA.lstsq(Xfull, y, rcond=-1)
            self.R2 = 1 - self.RSS/np.sum((y-np.mean(y))**2)

        elif self.optimizer == 'gradient_descent':
            self.n_points, self.n_features = Xfull.shape
            
            if self.n_points != len(y):
                raise ValueError('Mismatch in the dimensions of X and y')

            y_pred = np.mean(y)*np.ones(self.n_points)
            self.coeffs = np.zeros(self.n_features)
            iterNum = 0
            while iterNum < self.maxIter:
                  grad = -2*np.matmul((y-y_pred), Xfull)
                  if np.sqrt(np.dot(grad, grad)) < self.tol:
                      print('gradient descent converged')
                      break
                  self.coeffs = self.coeffs - self.lr*grad
                  y_pred = np.matmul(Xfull,self.coeffs)
                  iterNum += 1
            
            if iterNum == self.maxIter:
                print('Maximum number of iteration reached')

            
            self.RSS = np.sum((y - y_pred)**2)
            self.R2 = 1 - self.RSS/np.sum((y-np.mean(y))**2)


    def predict(self, X):
    '''

        Predicts the target values for the provided test data.

        Parameters
        ----------
            X : numpy.ndarray : Test data features, shape (n_samples, n_features)

        Returns
        ----------
            numpy.ndarray : Predicted target values, shape (n_samples,)

    '''
        if self.optimizer == 'least_squares':
            return self._ls_predict(X)
        elif self.optimizer == 'gradient_descent':
            return self._gd_predict(X)
    
    def get_mse(self, y_true, y_pred):
    '''
        Returns the mean square error

        Parameters
        -----------
        y_true: numpy.ndarray
            Actual labels (ground truth)

        y_pred: numpy.ndarray
            Predicted labels

    '''
        return np.mean((y_true-y_pred)**2)

    def R2_score(self, X, y):
    '''
        Finds the R-squared score for the test data

        Parameters
        ----------
        X: numpy.ndarray
            Test data

        y: numpy.ndarray
            Test labels
    '''
        Xfull = self._add_ones(X)
        y_pred = np.sum(self.coeffs*Xfull, axis=1)
        RSS = np.sum((y_pred - y)**2)
        return (1 - RSS/(np.sum((y-np.mean(y))**2)))
    
    def _add_ones(self, X):
    '''
        This helper function adds a column of ones to the end of the input data.
        This extra column corresponds to the intercept of the linear equation

        Parameters
        -----------
        X: numpy.ndarrray
            Input data
    '''
        if X.ndim == 1:
            return np.append(np.vstack(X), np.vstack(np.ones(len(X))), axis=1)
        else:
            return np.append(X, np.vstack(np.ones(len(X))), axis=1)

    def _ls_predict(self, X):
    '''
        Performs prediction for least square option
        
        Parameters
        -----------
        X: numpy.ndarray
            Test data
    '''
        Xfull = self._add_ones(X)
        y_pred = np.matmul(Xfull,self.coeffs)
        return y_pred

    def _gd_predict(self, X):
    '''
        Performs prediction for gradient descent option

        Parameters
        -----------
        X: numpy.ndarray
            Test data
    '''
        Xfull = self._add_ones(X)
        y_pred = np.matmul(Xfull,self.coeffs)
        return y_pred

