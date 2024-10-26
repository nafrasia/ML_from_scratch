import numpy as np

class KNearestNeighbor:
    '''
        Performs K-Nearest Neighbour

        Parameters
        ----------
            k: int, default=5
                Number of nearest neighbours

            p: int, default=2
                The value of p in p-norm (Minkowski distance)

            metric: string, default='minskowski'
                Type of distance used

            mode: string, default='classification'
                Type of model/problem

        Attributes
        ----------
            X_train: numpy.ndarray, pandas.DataFrame, pandas.Series
                training features

            y_train: numpy.ndarray, pandas.DataFrame, pandas.Series
                training labels

            categories: numpy.array
                list of features obsereved during fit/training


    '''
    def __init__(self, k=5, p=2, metric='minkowski', mode='classification'):
        self.k = k
        self.p = p
        self.metric = metric
        self.mode = mode
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        '''

            Fit the kNN model using the provided training data

            Parameters
            ----------
                X : numpy.ndarray 
                    Training data features, shape (n_samples, n_features)
                y : numpy.ndarray
                    Training data labels, shape (n_samples,)

        '''

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.mode == 'classification':
            self.categories = np.unique(self.y_train)

    def compute_distance(self, X, x0):
        '''
            Calculates p-norm distnace from x0 to X (single data point)

            Parameters
            ----------
                X: numpy.ndarray
                    a data point

                x0: numpy.ndarray
                    a data point

            Returns
            ---------
                float
                    distance
        '''

        X_np = np.array(X)
        x0_np = np.array(x0)

        distance = np.sum((X_np-x0_np)**self.p)**(1/self.p)
        return distance

    def _classifier(self, X):
        '''
            Predicts the target values for a single test data point
            using majority vote 

            Parameters
            ----------
                X : numpy.ndarray
                    Test data features, shape (n_samples, n_features)
        
            Returns
            ----------
                int
                    Predicted target class
        '''
        most_common = np.array([])
        for point in X:
            distance = np.array([])
            kNearestLabels= np.array([])
            for neighbor in self.X_train:
                distance = np.append(distance, self.compute_distance(neighbor, point))
            sortedIndex = np.argsort(distance)[:self.k]
            for index in sortedIndex:
                kNearestLabels = np.append(kNearestLabels, self.y_train[index])
            uniqueLabels, inverseMap, labelCounts = np.unique(kNearestLabels, return_counts=True, return_inverse=True)
            most_common = np.append(most_common, uniqueLabels[np.argmax(labelCounts)])
                # A failsafe needs to be implemented to inform user of potential tie of number of labels and suggest modifying k

        return most_common

    def _regressor(self, X):
        '''
        Predicts the target values for a single test data point
        using the mean value of the k nearest neighbour labels

        Parameters
        ----------
            X : numpy.ndarray
                Test data features, shape (n_samples, n_features)
        
        Returns
        ----------
             float
                Predicted target values

        '''

        kMean = np.array([])
        for point in X:
            distance = np.array([])
            kNearestLabels = np.array([])
            for neighbor in self.X_train:
                distance = np.append(distance, self.compute_distance(neighbor, point))
            
            sortedIndex = np.argsort(distance)[:self.k]
            for index in sortedIndex:
                kNearestLabels = np.append(kNearestLabels, self.y_train[index])
            
            kMean = np.append(kMean, np.mean(kNearestLabels))

        return kMean

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
        
        if self.mode == 'classification':
            y_pred = self._classifier(X)

        elif self.mode == 'regression':
            y_pred = self._regressor(X)
        
        else:
            raise ValueError('Invalid mode. mode should be either "classification" or "regression"')

        return y_pred


    def accuracy_score(self, y_true, y_pred):
        '''
            Calculates the accuracy of predictions vs. ground truth
            This is the accuracy for a classification and RMSE for a
            regression

            Parameters
            ----------
                y_true: numpy.ndarray
                    Actual labels of the test data

                y_pred: numpy.ndarray
                    Predicted labels of the test data

            Returns
            ---------
                float
                    accuracy or RMSE

        '''

        if self.mode == 'classification':
            comp, compCount = np.unique(y_true==y_pred, return_counts=True)
            if len(comp) == 1:
                acc = 1.0
            else:
                acc = compCount[1]/(compCount[0]+compCount[1])

            return acc

        elif self.mode == 'regression':
            print('Computing Root-Mean-Squared-Error (RMSE)...')
            rmse = np.sqrt(np.sum((y_true-y_pred)**2))
            
            return rmse

