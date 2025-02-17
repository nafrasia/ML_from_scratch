import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
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
    '''
    Returns value of sigmoid function at X
    '''
    return 1/(1+np.exp(-X))

def softmax(X):
    ''' Returns softmax function of X'''
    return np.exp(X)/np.sum(np.exp(X))

