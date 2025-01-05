import numpy as np

def quicksort(X):
    '''
    Performs quick sort

    Parameter
    ----------
    X: list-like object
        The array to sort

    Returns
    ----------
    Sorted array

    '''
    if len(X) <= 1:
        return X
    else:
        pivot = X[len(X) // 2]
        left = [x for x in X if x < pivot]
        middle = [pivot]
        right = [x for x in X if x > pivot]
        return quicksort(left) + middle + quicksort(right)

def sigmoid(X):
    return 1/(1+np.exp(-X))
