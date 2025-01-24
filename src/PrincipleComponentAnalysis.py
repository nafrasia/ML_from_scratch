import numpy as np
from numpy.linalg import svd

class PrincipleComponentAnalysis:
    """
    Extract Priciple Axes of a dataset

    Parameters
    -----------
    n_components: int, default=None
        Number of priciple axes used to transform the data

    Attributes
    -----------
    n_components:
        Number of priciple axes used to transform the data

    p_components:
        The principle component matrix (typically shown as U)

    singular_values:
        The singular values obtained from Singular Value Decomposition(Eigenvalues of Covariance matrix)

    explained_variance:
        Variance along each priciple axis (singular_values**2/(n_samples-1))

    Vt:
        Transpose of the matrix with columns equal to right singular vectors

    """

    def __init__(self,n_components=None):
        """
        Class constructor
        """
        self.n_components = n_components
        self.p_components = None
        self.singular_values = None
        self.explained_variance = None
        self.Vt = None

    def fit(self,X):
        """
        Compute principle axes and singular values

        Parameters
        -----------
        X: np.ndarray
            Input data

        """
        n_samples, n_features = X.shape

        X_centred = X - np.mean(X, axis=0) #Centring the data

        if self.n_components is None:
            self.n_components = np.min(X.shape)

        U,S,Vt = svd(X_centred, full_matrices=False) #perform svd. full_matrices=False allows for reduced svd if needed.
        self.p_components = U
        self.singular_values = S
        self.explained_variance = S**2/(n_samples-1)
        self.Vt = Vt

    def transform(self,X):
        """
        Returns the data with contributions from n_components axes

        Parameters
        ----------
        X: np.ndarray
            Input data (data to reduce dimensionality of)

        Returns
        ---------
        Reduced data

        """
        means = np.mean(X, axis=0)
        X_centred = X - means

        S = self.singular_values
        if self.n_components < len(S):
            n = len(S)-self.n_components
            Sn = np.diag(S* np.append(np.ones(self.n_components), np.zeros(n))) #Axis switch. Use a diagonal matrix with ones only for first n_components
            return np.matmul(np.matmul(self.p_components, Sn),self.Vt)+means # Perform reverse SVD and add the mean to get the recuded data
        else:
            return np.matmul(np.matmul(self.p_components, np.diag(S)),self.Vt)+means
