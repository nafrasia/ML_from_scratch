import numpy as np
import pandas as pd

class Node:
    '''
    Contains properties of each node of a tree
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):#The astrisk forces value to be explicitly passed by name (Not positional)
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    '''
    Performs a Decision Tree Classification

    Parameters
    -----------
    criterion: str, default: 'gini'
        Method to calculate information gain

    min_samples_split: int, default: 2
        Minimum number of samples per node
        before calling it a leaf node
    
    max_depth: int, default: 20
        Maximum number of branching splits

    n_features: int, default: None
        Number of features to design the tree
        based of. It has to be less that total
        number of features in the data
    
    Attributes
    -----------
    criterion: str, default: 'gini'
        Method to calculate information gain

    min_samples_split: int, default: 2
        Minimum number of samples per node
        before calling it a leaf node
    
    max_depth: int, default: 20
        Maximum number of branching splits

    n_features: int, default: None
        Number of features to design the tree
        based of. It has to be less that total
        number of features in the data

    root: int, default: None
        The root of the tree

    '''

    def __init__(self, criterion='gini', min_samples_split=2, max_depth=20, n_features=None):
        self.criterion=criterion
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root= None

    def fit(self, X, y):
        '''
        Fits the model (creates a tree) based on training data

        Parameter
        ----------
        X: np.ndarray, pd.DataFrame
            Training data

        y: np.ndarray, pd.DataFrame
            Training labels

        '''
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = y.to_numpy()
        
        n_samples, n_feats = X.shape
        if self.n_features == None:
            self.n_features = n_feats
        else:
            if self.n_features > n_feats:
                raise ValueError(f'Number of features {self.n_features} specified is more than total number of features {features}')
        #debugging
        #print('before grow_tree')
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        '''
        Creates a tree

        Parameter
        ----------
        X: np.ndarray
            Training data

        y: np.ndarray
            Training labels

        depth: int, default: 0
            The depth of the current branch

        Return
        ---------
        Base of the branch node

        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check stopping criteria
        if depth > self.max_depth or n_samples < self.min_samples_split or n_labels==1:
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)
        
        #find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        best_feat, best_thr = self._best_split(X,y, feat_idxs)
        #debugging
        #print(best_feat, best_thr)
        Xleft = X[np.ravel(np.argwhere(X[:, best_feat]<best_thr)),:]
        Xright = X[np.ravel(np.argwhere(X[:, best_feat]>=best_thr)), :]
        yleft = y[np.argwhere(X[:, best_feat]<best_thr)]
        yright = y[np.argwhere(X[:, best_feat]>=best_thr)]

        left = self._grow_tree(Xleft, yleft, depth+1)
        right = self._grow_tree(Xright, yright, depth+1)

        return Node(best_feat, best_thr, left, right)

    def _most_common_label(self, y):
        '''
        Finds the most probable target value

        Parameter
        ----------
        y: np.ndarray
            The labels

        Return
        ---------
        The label with highest count

        '''
        labels, labelcount = np.unique(y, return_counts=True)
        return labels[np.argmax(labelcount)]
    
    def _best_split(self, X, y, feat_idxs):
        '''
        Finds the best feature and
        value of that feature to split 
        the tree on using best
        information gain
        
        Parameter
        ----------
        X: np.ndarray

        y: np.ndarray

        feat_idxs: list-like object

        Return
        ---------
        The best feature and best value
        of that feature to perform split on

        '''
        best_gain = -1
        
        for feat_idx in feat_idxs:
            #debugging
            #print(feat_idx)
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            #print(X_column.shape, y.shape)
            for thr in thresholds:
                if self.criterion == 'entropy':
                    gain = self._calc_IG(X_column, y, thr)
                elif self.criterion == 'gini':
                    gain = self._calc_GG(X_column, y, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _calc_IG(self, X, y, thr):
        '''
        Compute Entropy Information Gain

        Parameter
        ----------
        X: np.ndarray
            A column (feature) of the training data
        
        y: np.ndarray
            Training labels
        
        thr: float
            The value to split the branch on

        Return
        --------
        Entropy Information Gain

        '''
        pLabels, pCount = np.unique(y, return_counts=True)
        Eparent = -np.sum(pCount/np.sum(pCount)*np.log(pCount/np.sum(pCount)))

        yleft, yright = np.array([]),np.array([])
        for i in range(len(X)):
            if X[i] < thr:
                yleft = np.append(yleft, y[i])
            else:
                yright = np.append(yright, y[i])
        
        if len(yleft)==0 or len(yright)==0:
            return 0

        leftLabels, leftCount = np.unique(yleft, return_counts=True)
        rightLabels, rightCount = np.unique(yright, return_counts=True)
        Eleft =  -np.sum(leftCount/np.sum(leftCount)*np.log(leftCount/np.sum(leftCount)))
        Eright =  -np.sum(rightCount/np.sum(rightCount)*np.log(rightCount/np.sum(rightCount)))

        IG = Eparent - len(yleft)/len(y)*Eleft - len(yright)/len(y)*Eright

        return IG

    def _calc_GG(self, X, y, thr):
        '''
        Compute Gini Gain

        Parameter
        ----------
        X: np.ndarray
            A column (feature) of the training data
        
        y: np.ndarray
            Training labels
        
        thr: float
            The value to split the branch on

        Return
        --------
        Gini Information Gain

        '''
        pLabels, pCount = np.unique(y, return_counts=True)
        Gparent = 1-np.sum((pCount/np.sum(pCount))**2)
        
        yleft, yright = np.array([]),np.array([])
        
        for i in range(len(X)):
            if X[i] < thr:
                yleft = np.append(yleft, y[i])
            else:
                yright = np.append(yright, y[i])

        if len(yleft)==0 or len(yright)==0:
            return 0

        leftLabels, leftCount = np.unique(yleft, return_counts=True)
        rightLabels, rightCount = np.unique(yright, return_counts=True)
        Gleft =  1 - np.sum((leftCount/np.sum(leftCount))**2)
        Gright =  1 - np.sum((rightCount/np.sum(rightCount))**2)

        GG = Gparent - len(yleft)/len(y)*Gleft - len(yright)/len(y)*Gright

        return GG

    def predict(self, X):
        '''
        Performs prediction on test dataset

        Parameter
        ----------
        X: np.ndarray, pd.DataFrame
            The test data

        Return
        ---------
        Predicted labels/targets

        '''
        if type(X)==pd.DataFrame:
            X = X.to_numpy()

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, X, node):
        '''
        This helper function follows the tree for
        a specific test data point top to bottom
        till it finds a leaf node. The value of
        the leaf node is the prediction for
        that data point

        Parameter
        ----------
        X: np.ndarray
            The test dataset

        node: DecisionTree.Node
            The starting point

        Return
        ---------
        Predicted label/target
        '''
        if node.is_leaf():
            return node.value

        if X[node.feature] < node.threshold:
            return self._traverse_tree(X,node.left)
        else:
            return self._traverse_tree(X, node.right)


class DecisionTreeRegressor:
    '''
    Performs a Decision Tree Regression

    Parameters
    -----------
    criterion: str, default: 'squared-error'
        Method to calculate information gain

    min_samples_split: int, default: 2
        Minimum number of samples per node
        before calling it a leaf node
    
    max_depth: int, default: 20
        Maximum number of branching splits

    n_features: int, default: None
        Number of features to design the tree
        based of. It has to be less that total
        number of features in the data
    
    Attributes
    -----------
    criterion: str, default: 'gini'
        Method to calculate information gain

    min_samples_split: int, default: 2
        Minimum number of samples per node
        before calling it a leaf node
    
    max_depth: int, default: 20
        Maximum number of branching splits

    n_features: int, default: None
        Number of features to design the tree
        based of. It has to be less that total
        number of features in the data

    root: int, default: None
        The root of the tree

    '''

    def __init__(self, criterion='squared-error', min_samples_split=2, max_depth=20, n_features=None):
        self.criterion=criterion
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root= None

    def fit(self, X, y):
        '''
        Fits the model (creates a tree) based on training data

        Parameter
        ----------
        X: np.ndarray, pd.DataFrame
            Training data

        y: np.ndarray, pd.DataFrame
            Training labels

        '''
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = y.to_numpy()
        
        n_samples, n_feats = X.shape
        if self.n_features == None:
            self.n_features = n_feats
        else:
            if self.n_features > n_feats:
                raise ValueError(f'Number of features {self.n_features} specified is more than total number of features {features}')
        
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        '''
        Creates a tree

        Parameter
        ----------
        X: np.ndarray
            Training data

        y: np.ndarray
            Training labels

        depth: int, default: 0
            The depth of the current branch

        Return
        ---------
        Base of the branch node

        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check stopping criteria
        if depth > self.max_depth or n_samples < self.min_samples_split or n_labels==1:
            leaf_value = self._average_label(y)
            return Node(value= leaf_value)
        
        #find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        best_feat, best_thr = self._best_split(X,y, feat_idxs)
        #debugging
        #print(best_feat, best_thr)
        Xleft = X[np.ravel(np.argwhere(X[:, best_feat]<best_thr)),:]
        Xright = X[np.ravel(np.argwhere(X[:, best_feat]>=best_thr)), :]
        yleft = y[np.argwhere(X[:, best_feat]<best_thr)]
        yright = y[np.argwhere(X[:, best_feat]>=best_thr)]

        left = self._grow_tree(Xleft, yleft, depth+1)
        right = self._grow_tree(Xright, yright, depth+1)

        return Node(best_feat, best_thr, left, right)

    def _average_label(self, y):
        '''
        Returns the mean target values in a leaf

        Parameter
        ----------
        y: np.ndarray
            The labels

        Return
        ---------
        Average label value

        '''
        return np.mean(y)
    
    def _best_split(self, X, y, feat_idxs):
        '''
        Finds the best feature and
        value of that feature to split 
        the tree on using MSE
        
        Parameter
        ----------
        X: np.ndarray

        y: np.ndarray

        feat_idxs: list-like object

        Return
        ---------
        The best feature and best value
        of that feature to perform split on

        '''
        best_error = np.mean((np.mean(y)-y)**2)
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            #====================================================
            # The 2 criterion for differentiating categorical
            # and continuous features is arbitrary. I picked
            # it assuming all categorical data with more than 2
            # categories are OneHot-encoded
            #====================================================
            if len(thresholds) > 2: 
                thresholds = (thresholds[1:]+thresholds[:-1])/2
            
            for thr in thresholds:
                if self.criterion == 'squared-error':
                    squared_error = self._calc_MSE(X_column, y, thr)
                else:
                    raise ValueError("Invalid criterion! Currently, only 'squared-error' allowed")

                if squared_error < best_error:
                    best_error = squared_error
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _calc_MSE(self, X, y, thr):
        '''
        Compute Entropy Information Gain

        Parameter
        ----------
        X: np.ndarray
            A column (feature) of the training data
        
        y: np.ndarray
            Training labels
        
        thr: float
            The value to split the branch on

        Return
        --------
        Weighted mean-square-error

        '''

        yleft, yright = np.array([]),np.array([])
        for i in range(len(X)):
            if X[i] < thr:
                yleft = np.append(yleft, y[i])
            else:
                yright = np.append(yright, y[i])
        
        if len(yleft)==0 or len(yright)==0:
            return 0
        
        leftMSE = np.mean((np.mean(yleft)-yleft)**2)
        rightMSE = np.mean((np.mean(yright)-yright)**2)

        MSE = len(yleft)/len(y)*leftMSE + len(yright)/len(y)*rightMSE 

        return MSE

    def predict(self, X):
        '''
        Performs prediction on test dataset

        Parameter
        ----------
        X: np.ndarray, pd.DataFrame
            The test data

        Return
        ---------
        Predicted labels/targets

        '''
        if type(X)==pd.DataFrame:
            X = X.to_numpy()

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, X, node):
        '''
        This helper function follows the tree for
        a specific test data point top to bottom
        till it finds a leaf node. The value of
        the leaf node is the prediction for
        that data point

        Parameter
        ----------
        X: np.ndarray
            The test dataset

        node: DecisionTree.Node
            The starting point

        Return
        ---------
        Predicted label/target
        '''
        if node.is_leaf():
            return node.value

        if X[node.feature] < node.threshold:
            return self._traverse_tree(X,node.left)
        else:
            return self._traverse_tree(X, node.right)

