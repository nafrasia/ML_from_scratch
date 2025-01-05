import numpy as np
from numpy.linalg import inv
from utility import sigmoid

class LogisticRegression:
    '''
    Performs Logistic Regression

    Parameters
    -----------
    penalty: str, default
        Regularization method

    solver: str, default: 'newton'
        Optimization algorithm to find minimum of the log-likelihood/log-loss
        Currently, 'newton' and 'gradient_descent' available

    max_iter: int, default:100
        Maximum number of iterations for the optimization alogrithm

    tol: float, default:0.0001
        Tolerance criteria for optimization algorithm

    learning_rate: float, default: 0.001
        (Only for gradient_descent) 

    Attributes
    -----------
    lr: float
        learning rate

    classes: np.ndarray
        Different classes of labels

    n_classes: int
        Number of classes (different labels)
    '''
    def __init__(self, penalty = None, solver='newton', max_iter=100, tol=0.0001, learning_rate=0.001):
        self.coeffs = None
        self.penalty = penalty #TODO(3) Introduce L1 and L2 regularizations
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.lr = learning_rate
        self.classes = None
        self.n_classes = None
    
    def fit(self, X, y):
        '''
        Fits the Logisitic model from training data

        Parameters
        ----------
        X: np.ndarray
            Training data

        y: np.ndarray
            Training labels

        '''
        #TODO(4) Adapt for pandas DataFrame
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        if n_samples != len(y):
            raise ValueError('Mismatch in the dimensions of X and y')

        if self.n_classes < 2:
            print('Warning: Only one class is found')

        elif self.n_classes < 3:
            self.coeffs = self._binary_classification(X,y)

        else:
            #TODO(1) Implement a 'one-vs-all' method for multiclass classification
            print('Multi-class classification. To be implemented...')
    
    def _binary_classification(self, X, y):
        '''
        This helper function performs binary classification
        '''
        n_samples, n_features = X.shape
        
        Xfull = self._add_ones(X)

        coeffs = np.zeros(n_features+1)
        
        if self.solver == 'newton':
            for iterNum in range(self.max_iter):
                y_pred = sigmoid(np.dot(Xfull,coeffs))
                score = self._calc_score(Xfull, y, y_pred) # The gradient is called score here to emphasize
                                                           # the fact that we are working with log(likelihood)
                if np.sqrt(np.dot(score, score)) < self.tol:
                    print('Newton method converged')
                    break
                hessian = self._calc_hessian(Xfull, y_pred, n_samples)
                coeffs = coeffs - np.dot(inv(hessian), score) # TODO(2) Address singular hessian

            if iterNum == self.max_iter:
                print('Maximum number of iteration reached')
        
        elif self.solver == 'gradient_descent':
            y_pred = sigmoid(np.dot(Xfull,coeffs))
            iterNum = 0
            while iterNum < self.max_iter:
                grad = -np.matmul((y-y_pred), Xfull)
                if np.sqrt(np.dot(grad, grad)) < self.tol:
                    print('gradient descent converged')
                    break
                coeffs = coeffs - self.lr*grad
                y_pred = sigmoid(np.matmul(Xfull,coeffs))
                iterNum += 1

            if iterNum == self.max_iter:
                print('Maximum number of iteration reached')


        return coeffs

    def _calc_score(self, X, y, yp):
        '''
        Calculating the gradient of the log-likelihood
        with respect to the linear model weights

        Parameters
        -----------
        X: np.ndarray
            input matrix
        y: np.ndarray
            target vector
        yp: np.ndarray
            prediction vector

        Return
        -----------
        gradient
        '''
        return np.dot(X.T,(y-yp))

    def _calc_hessian(self, X, yp, n):
        '''
        Calculating the Hessian of the log-likelihood
        with respect to the linear model weights (bias included)

        Parameters
        -----------
        X: np.ndarray
            input matrix
        y: np.ndarray
            target vector
        n: int
            number of samples

        Return
        -----------
        Hessian (second derivative matrix)
        '''
        W = np.eye(n)*(yp*(1-yp))
        hessian = -np.dot(np.dot(X.T, W), X)
        return hessian

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


    def predict(self, X):
        '''
        Predicting classes

        Parameters
        -----------
        X: np.ndarray
            Input data

        Return
        -----------
        Predicted target classes
        '''
        Xfull = self._add_ones(X)
        y_prob = sigmoid(np.dot(Xfull,self.coeffs))
        y_pred = np.array([])
        for y in y_prob:
            if y < 0.5:
                y_pred = np.append(y_pred, 0)
            else:
                y_pred = np.append(y_pred, 1)
        return y_pred
