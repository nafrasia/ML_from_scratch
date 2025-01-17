import numpy as np
import pandas as pd
from DecisionTree import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Random Forest Classifier using an ensemble of decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the individual decision trees.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    n_features : int, default=None
        The number of features to consider when looking for the best split.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.

    criterion : str, default='gini'
        The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.

    Attributes
    ----------
    oob_score_ : float
        The out-of-bag score of the fitted random forest, if bootstrap is True.
    """

    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=2, n_features=None, bootstrap=True, criterion='gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.trees = []
        self.oob_score_ = None

    def fit(self, X, y):
        """
        Fit the random forest classifier on the training data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        y : numpy.ndarray
            Target vector of shape (n_samples,).
        """
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.DataFrame:
            y = y.to_numpy().flatten()
        
        self.trees = []
        n_samples, n_feats = X.shape
        oob_votes = np.zeros((n_samples, len(np.unique(y))))
        oob_counts = np.zeros(n_samples)

        # Set the number of features if not provided
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_feats))

        for _ in range(self.n_estimators):  # Iterate over the number of trees to create an ensemble
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                criterion=self.criterion
            )

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]

                # Track out-of-bag samples
                oob_indices = np.setdiff1d(np.arange(n_samples), indices)  # Find indices not included in the bootstrap sample
                if len(oob_indices) > 0:
                    tree.fit(X_sample, y_sample)
                    oob_predictions = tree.predict(X[oob_indices])
                    for i, pred in zip(oob_indices, oob_predictions):
                        oob_votes[i, pred] += 1
                        oob_counts[i] += 1
            else:
                X_sample, y_sample = X, y
                tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        if self.bootstrap:
            self.oob_score_ = self._calculate_oob_score(oob_votes, oob_counts, y)

    def _calculate_oob_score(self, oob_votes, oob_counts, y):
        """
        Calculate the out-of-bag score for the random forest.

        Parameters
        ----------
        oob_votes : numpy.ndarray
            Array of votes from out-of-bag samples, shape (n_samples, n_classes).

        oob_counts : numpy.ndarray
            Array of counts for how many times each sample was out-of-bag.

        y : numpy.ndarray
            True target values.

        Returns
        -------
        float
            The out-of-bag score.
        """
        valid_indices = oob_counts > 0
        oob_predictions = np.argmax(oob_votes[valid_indices], axis=1)
        return np.mean(oob_predictions == y[valid_indices])

    def predict(self, X):
        """
        Predict class labels for the given input data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted class labels of shape (n_samples,).
        """

        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(self._majority_vote, axis=0, arr=tree_predictions)

    def _majority_vote(self, predictions):
        """
        Perform majority voting among tree predictions.

        Parameters
        ----------
        predictions : numpy.ndarray
            Array of predictions from the trees.

        Returns
        -------
        int
            Predicted class label.
        """
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]
