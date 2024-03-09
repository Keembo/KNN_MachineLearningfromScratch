"""Regressor with Nearest Neighbors"""

# Import libraries
import numpy as np

# Import created functions and classes
from._base import _get_weights
from._base import NearestNeighbor

class KNeighborsRegressor(NearestNeighbor):
    """
    A class for implementing k-nearest neighbors regression.

    Parameters
    ----------
    n_neighbors : int, default=5
        The number of neighbors to use for classification.
    weights : {'uniform', 'distance'}, default='uniform'
        If 'uniform', all points in each neighborhood are weighted equally.
        If 'distance', the weight of a point is proportional to the inverse
        of its distance to the nearest neighbor.
    p : int, default=2
        The power of the Minkowski metric to use when calculating distance.

    Attributes
    ----------
    n_neighbors : int
        The number of neighbors to use for classification.
    weights : {'uniform', 'distance'}
        The weighting scheme to use when calculating neighbor weights.
    p : int
        The power of the Minkowski metric to use when calculating distance.
    _y : array-like of shape (n_samples,)
        The labels associated with the training data.
    _X : array-like of shape (n_samples, n_features)
        The training data.

    Methods
    -------
    fit(X, y)
        Fits the model to the training data.
    predict(X)
        Returns the predicted labels for the provided data points.
    """
    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        p=2
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            p=p
        )
        self.weights = weights
        
    def predict(self, X):
        """
        Predict the labels for the provided data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data points for which to predict labels.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted labels for the data points.
        """ 
        X = np.array(X)
        
        # Calculate weights
        if self.weights == 'uniform':
            neigh_ind = self._kneighbors(X, return_distance = False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self._kneighbors(X)
            
        weights = _get_weights(neigh_dist, self.weights)
        
        # Get the prediction
        _y = self._y
        if self.weights == 'uniform':
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            num = np.sum(_y[neigh_ind]* weights, axis=1)
            denom = np.sum(weights, axis=1)
            y_pred = num/denom
            
        return y_pred