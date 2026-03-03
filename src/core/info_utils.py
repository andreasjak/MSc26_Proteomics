"""
info_utils.py
-------------
Reusable information-theoretic utilities for the MSc26 proteomics project.

Provides mutual information and entropy computation functions.
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from npeet import entropy_estimators as ee

from itertools import combinations

def interaction_information(X, Y=None, k=3):
    """Compute the interaction information among features in X and potentially with binary target Y.
    
    Parameters:
    - X: array-like of shape (n_samples, n_features)
        The input samples.
    - Y: array-like of shape (n_samples,), optional
        The binary target variable, only 0 or 1.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if Y is not None: 
        I = 0
        for size in range(1, n_features + 1):
            sign = (-1) ** (size + 1)
            for subset in combinations(range(n_features), size):
                I += sign * ee.micd(X[:, subset], Y.reshape(-1, 1), k=k)

    else: 
        if n_features < 2:
            raise ValueError("At least two features are required to compute interaction information.")
        I = 0
        for size in range(1, n_features + 1):
            sign = (-1) ** (size - 1)
            for subset in combinations(range(n_features), size):
                I += sign * ee.entropy(X[:, subset], k=k)
    return I*np.log(2)

def interaction_information_ccd(X1, X2, Y, k=3, method='npeet', IX1Y=None, IX2Y=None):
    """Compute the 3-way interaction information among features in X1, X2 and binary target Y."""
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    Y = np.asarray(Y).reshape(-1, 1)
    
    if method == 'npeet':
        # Using NPEET
        idx = Y.ravel() == 1
        p = np.mean(idx)
        I = ee.mi(X1, X2, k=k) - p * ee.mi(X1[idx], X2[idx], k=k) - (1 - p) * ee.mi(X1[~idx], X2[~idx], k=k)
        return I*np.log(2)
    elif method == 'sklearn':
        # Using scikit-learn
        idx = Y.ravel() == 1
        p = np.mean(idx)
        I = mutual_info_regression(X1, X2.ravel(), n_neighbors=k)[0] \
            - p * mutual_info_regression(X1[idx], X2[idx].ravel(), n_neighbors=k)[0] \
            - (1 - p) * mutual_info_regression(X1[~idx], X2[~idx].ravel(), n_neighbors=k)[0]
        return I
    elif method == 'npeet_2':
        # Using NPEET with conditional mutual information
        I = ee.mi(X1, X2, k=k) - ee.cmi(X1, X2, Y, k=k)
        return I*np.log(2)
    elif method == 'npeet_3':
        # Using NPEET if I(X1;Y)=IX1Y and I(X2;Y)=IX2Y are both known
        I = IX1Y + IX2Y - ee.micd(np.c_[X1, X2], Y.reshape(-1, 1), k=k)*np.log(2)
        return I
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'sklearn', 'npeet_2', or 'npeet_3'.")
    
def interaction_information_ccc(X1, X2, X3, k=3, method='npeet', IX1X3=None, IX2X3=None):
    """Compute the 3-way interaction information among features in X1, X2, X3."""
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    X3 = np.asarray(X3).reshape(-1, 1)

    if method == 'npeet':
        # Using NPEET with conditional mutual information
        I = ee.mi(X1, X2, k=k) - ee.cmi(X1, X2, X3, k=k)
        return I*np.log(2)
    elif method == 'sklearn':
        # Using scikit-learn
        raise NotImplementedError("Scikit-learn does not support conditional mutual information for continuous variables.")
    elif method == 'npeet_2':
        # Using NPEET if I(X1;Y)=IX1Y and I(X2;Y)=IX2Y are both known
        I = IX1X3 + IX2X3 - ee.mi(np.c_[X1, X2], X3, k=k)*np.log(2)
        return I
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'npeet_2'.")




def mutual_information_cd(X, Y, k=3, method='npeet'):
    """Compute the mutual information between features in X and binary target Y."""
    X = np.asarray(X)
    n_features = X.shape[1]
    Y = np.asarray(Y).reshape(-1, 1)

    if method == 'npeet':
        MI = np.zeros(n_features)
        for i in range(n_features):
            MI[i] = ee.micd(X[:, i].reshape(-1, 1), Y, k=k)
        return MI*np.log(2)
    elif method == 'sklearn':
        return mutual_info_classif(X, Y.ravel(), n_neighbors=k)
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'sklearn'.")
    
def mutual_information_cc(X, X2, k=3, method='npeet'):
    """Compute the mutual information between features in X and feature X2."""
    X = np.asarray(X)
    n_features = X.shape[1]
    X2 = np.asarray(X2).reshape(-1, 1)

    if method == 'npeet':
        MI = np.zeros(n_features)
        for i in range(n_features):
            MI[i] = ee.mi(X[:, i].reshape(-1, 1), X2, k=k)
        return MI*np.log(2)
    elif method == 'sklearn':
        return mutual_info_regression(X, X2.ravel(), n_neighbors=k)
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'sklearn'.")



def joint_mutual_information_cd(X, Y, k=3, method='npeet'):
    """Compute the joint mutual information between feature X and binary target Y, I({X};Y)."""
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1, 1)

    if method == 'npeet':
        return ee.micd(X, Y, k=k)*np.log(2)
    elif method == 'sklearn':
        raise NotImplementedError("Scikit-learn does not support joint mutual information for continuous variables.") 
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'sklearn'.")
    
def joint_mutual_information_cc(X, X2, k=3, method='npeet'):
    """Compute the joint mutual information between features X and X2, I({X};X2)."""
    X = np.asarray(X)
    X2 = np.asarray(X2).reshape(-1, 1)

    if method == 'npeet':
        return ee.mi(X, X2, k=k)*np.log(2)
    elif method == 'sklearn':
        raise NotImplementedError("Scikit-learn does not support joint mutual information for continuous variables.")   
    else:
        raise ValueError("Invalid method specified. Choose from 'npeet', 'sklearn'.")