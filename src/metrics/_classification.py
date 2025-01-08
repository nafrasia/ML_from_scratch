''' Metrics to evaluate performance of classification tasks'''
#=============================================================
#importing modules
#=============================================================
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib import colormaps
#=============================================================
# Functions
#=============================================================
def accuracy_score(y_true, y_pred):
    ''' 
    Computing accuracy of a classification prediction
    
    Parameters
    ----------
    y_true: np.ndarray
        Ground truth

    y_pred: np.ndarray
        Predictions

    Return
    ---------
    Accuracy (percentage of correct predictions)
    '''

    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, visualize=False):
    '''
    Computing confusion matrix

    Parameters
    -----------
    y_true: np.ndarray
        Ground Truth labels

    y_pred: np.ndarray
        Predicted labels

    visualize: Boolean, default:False
        If a heatmap of confusion matrix should be shown

    Return
    ---------
    Confusion Matrix: np.ndarray

    '''
    if len(y_true) != len(y_pred):
        ValueError(f'Length of vectors {len(y_true)} and {len(y_pred)} are not the same')

    for i in randint(0,len(y_true), size=int(len(y_true)*0.2)): # Only testing 20% of the array to save computation
        if y_pred[i] > 0 and y_pred[i] < 1:
            ValueError('Predictions are presented in probability format')

    unique_labels = np.unique(y_pred)
    n_l = len(unique_labels)
    CM = np.zeros((n_l, n_l))

    for i in range(len(y_pred)):
        CM[np.argwhere(unique_labels==y_true[i]),np.argwhere(unique_labels==y_pred[i])] += 1

    if visualize:
        _plot_confusion_matrix(CM)

    return CM.astype(int)

def _plot_confusion_matrix(CM, labels=None):
    '''
    Plotting confusion matrix on a heatmap

    Parameters
    ----------
    CM: np.ndarray
        Confusion Matrix

    labels: list-like, default:None
        list of class names
    
    '''
    rows, cols = CM.shape
    
    fig, ax = plt.subplots(figsize=(1.5*rows,1.5*rows))

    ax.imshow(CM, cmap=colormaps['Blues'], alpha=0.5)

    # Loop over data dimensions and create text annotations.
    for i in range(rows):
        for j in range(cols):
            text = ax.text(j, i, f'{CM[i, j]}({CM[i, j]*100/np.sum(CM[i,:]):0.1f}%)',
                        ha="center", va="center", color="k")

    # Show all ticks and label them with the respective classes.
    # If no list of string class names are passed, the classes
    # are labeled 0 to n-1
    if labels == None:
        labels = np.arange(rows)

    ax.set_xticks(range(cols), labels=labels,
                ha="right", rotation_mode="anchor")
    ax.set_yticks(range(rows), labels=labels)

    ax.set_xlabel('Predictions')
    ax.set_ylabel('Ground Truth')

def recall(y_true, y_pred):
    '''
    Compute recall for a classification problem

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth

    y_pred: np.ndarray
        Predictions

    Return
    ---------
    Recall for a binary classification

    or

    An array of recalls for multiclass classification

    '''
    cm = confusion_matrix(y_true, y_pred, visualize=False)
    
    recall = np.diag(cm)/np.sum(cm,axis=1)
    
    if len(recall) == 2:
        return recall[1]
    else:
        return recall

def average_recall(y_true, y_pred):
    '''
    Compute Macro-averaged recall
    
    Parameters
    ----------
    y_true: np.ndarray
        Ground truth

    y_pred: np.ndarray
        Predictions

    Return
    ---------
    Average recall

    '''
    return np.mean(recall(y_true, y_pred))

def precision(y_true, y_pred):
    '''
    Compute precision for a classification task

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth

    y_pred: np.ndarray
        Predictions

    Return
    ---------
    Precision for a binary classification

    or

    An array of precisions for multiclass classification

    '''
    cm = confusion_matrix(y_true, y_pred, visualize=False)

    precision = np.diag(cm)/np.sum(cm,axis=0)
    
    if len(precision) == 2:
        return precision[1]
    else:
        return precision

def average_precision(y_true, y_pred):
    '''
    Compute Macro-averaged precision
    
    Parameters
    ----------
    y_true: np.ndarray
        Ground truth

    y_pred: np.ndarray
        Predictions

    Return
    ---------
    Average precision

    '''
    return np.mean(precision(y_true, y_pred))

def F1score(y_true, y_pred):
    R = recall(y_true, y_pred)
    P = precision(y_true, y_pred)
    return 2*P*R/(P+R)

#TODO(1) Add AUC-ROC
