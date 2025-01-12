import numpy as np

class NaiveBayesBase:
    def __init__(self, class_priors=None):
        """
        Base class for Naive Bayes Classifiers.

        Parameters
        ----------
        class_priors : dict or None, optional
            Prior probabilities for each class. If None, priors are computed from the data.
        """
        self.classes = None  # To store unique class labels
        self.class_priors = class_priors  # To store prior probabilities of each class

    def fit(self, X, y):
        """
        Train the Naive Bayes Classifier by computing necessary statistics.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        """
        self.classes = np.unique(y)  # Find unique class labels
        if self.class_priors is None:
            self.class_priors = self._compute_class_priors(y)  # Compute prior probabilities for each class
        self._fit(X, y)

    def _compute_class_priors(self, y):
        """
        Compute the prior probabilities for each class.

        Parameters
        ----------
        y : numpy.ndarray
            Target vector of shape (n_samples,).

        Returns
        -------
        dict
            A dictionary with class labels as keys and prior probabilities as values.
        """
        priors = {}
        n_samples = len(y)
        for cls in self.classes:
            priors[cls] = np.sum(y == cls) / n_samples
        return priors

    def _fit(self, X, y):
        """
        To be implemented by derived classes for specific model types.
        """
        raise NotImplementedError("_fit must be implemented in derived classes")

    def predict(self, X):
        """
        Predict class labels for given input data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted class labels of shape (n_samples,).
        """
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """
        Predict the class label for a single data point.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        int or str
            Predicted class label.
        """
        class_probs = {}
        for cls in self.classes:
            prior = np.log(self.class_priors[cls])  # Use log to prevent underflow
            likelihood = self._compute_likelihood(cls, x)
            class_probs[cls] = prior + likelihood

        return max(class_probs, key=class_probs.get)

    def _compute_likelihood(self, cls, x):
        """
        To be implemented by derived classes for specific model types.
        """
        raise NotImplementedError("_compute_likelihood must be implemented in derived classes")

class GaussianNaiveBayes(NaiveBayesBase):
    def __init__(self, class_priors=None):
        """
        Gaussian Naive Bayes Classifier.

        Parameters
        ----------
        class_priors : dict or None, optional
            Prior probabilities for each class. If None, priors are computed from the data.
        """
        super().__init__(class_priors)
        self.class_stats = {}  # To store mean and variance for each feature per class

    def _fit(self, X, y):
        """
        Compute mean and variance for each feature per class.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        """
        for cls in self.classes:
            X_class = X[y == cls]
            self.class_stats[cls] = {
                "mean": X_class.mean(axis=0),
                "var": X_class.var(axis=0)
            }

    def _compute_likelihood(self, cls, x):
        """
        Compute the log-likelihood of the data given a class using Gaussian distribution.

        Parameters
        ----------
        cls : int or str
            The class label.
        x : numpy.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        float
            Log-likelihood value.
        """
        mean = self.class_stats[cls]["mean"]
        var = self.class_stats[cls]["var"]
        
        # Compute the Gaussian probability density function
        numerator = -0.5 * ((x - mean) ** 2 / (var + 1e-9))  # Add a small value to variance to prevent division by zero
        denominator = -0.5 * np.log(2 * np.pi * var + 1e-9)
        return np.sum(numerator + denominator)  # summation used as we are working with log-likelihood

class MultinomialNaiveBayes(NaiveBayesBase):
    def __init__(self, class_priors=None):
        """
        Multinomial Naive Bayes Classifier.

        Parameters
        ----------
        class_priors : dict or None, optional
            Prior probabilities for each class. If None, priors are computed from the data.
        """
        super().__init__(class_priors)
        self.feature_counts = {}  # To store feature counts per class
        self.feature_totals = {}  # To store total feature counts per class

    def _fit(self, X, y):
        """
        Compute feature counts and totals for each class.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        """
        for cls in self.classes:
            X_class = X[y == cls]
            self.feature_counts[cls] = X_class.sum(axis=0)
            self.feature_totals[cls] = X_class.sum()

    def _compute_likelihood(self, cls, x):
        """
        Compute the log-likelihood of the data given a class using Multinomial distribution.

        Parameters
        ----------
        cls : int or str
            The class label.
        x : numpy.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        float
            Log-likelihood value.
        """
        counts = self.feature_counts[cls]
        total = self.feature_totals[cls]
        probabilities = (counts + 1) / (total + len(counts))  # Add 1 for Laplace smoothing
        return np.sum(x * np.log(probabilities))
