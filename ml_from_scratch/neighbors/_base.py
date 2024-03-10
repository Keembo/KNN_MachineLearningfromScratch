"""Nearest Neighbors"""

# Import libraries
import numpy as np

def _get_weights(dist, weights):
    """
    Get the weights from an array of distances

    Parameters
    ----------
    dist : array-like, shape (n_queries, n_samples)
        The distances between each query point and each sample.
    weights : str or callable
        The weight function used in prediction.

    Returns
    -------
    weights_arr : array-like, shape (n_queries, n_samples)
        The weights for each query point.

    """
    if weights == 'uniform':
        weights_arr = None
    else:
        weights_arr = 1.0/(dist**2)

    return weights_arr

class NearestNeighbor:
    """
    A class for computing the nearest neighbors of a dataset.

    Parameters
    ----------
    n_neighbors : int, default=5
        The number of neighbors to use for knn search.
    weights : str or callable, default='uniform'
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
            are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
        - [callable] : a user-defined function which accepts an array of
            distances as input and returns an array of the same shape as
            distances.

    p : int, default=2
        The power parameter for the Minkowski distance.  When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2.

    Attributes
    ----------
    n_neighbors : int
        The number of neighbors to use for knn search.
    weights : str or callable
        The weight function used in prediction.
    p : int
        The power parameter for the Minkowski distance.
    _X : array-like, shape (n_samples, n_features)
        The training data.
    _y : array-like, shape (n_samples,)
        The target values.

    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.
    kneighbors(X, return_distance=True)
        Find the nearest neighbors of each sample in X.

    """
    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        p=2
    ):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        
    def _compute_distance(self, x1, x2):
        """
        Compute the distance between two points.

        Parameters
        ----------
        x1 : array-like, shape (n_features,)
            The first point.
        x2 : array-like, shape (n_features,)
            The second point.

        Returns
        -------
        dist : float
            The distance between the two points.

        """
        abs_diff = np.abs(x1 - x2)
        sigma_diff = np.sum(np.power(abs_diff, self.p))
        dist = np.power(sigma_diff, 1./self.p)
        return dist
    
    def _kneighbors(self, X, return_distance=True):
        """
        Find the nearest neighbors of each sample in X.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            The query points.
        return_distance : bool, default=True
            Whether to return the distances between the query points and the
            neighbors.

        Returns
        -------
        neighbors : array-like, shape (n_queries, n_neighbors)
            The indices of the nearest neighbors for each query point.
        distances : array-like, shape (n_queries, n_neighbors)
            The distances between the query points and their neighbors, only
            returned if return_distance is True.

        """
        n_queries = X.shape[0]
        n_samples = self._X.shape[0]
        list_dist = np.empty((n_queries, n_samples))
        for i in range(n_queries):
            X_i = X[i]
            for j in range(n_samples):
                X_j = self._X[j]
                dist_ij = self._compute_distance(x1=X_i, x2=X_j)
                list_dist[i,j] = dist_ij
                
        # Sort the distance in ascending order
        neigh_ind = np.argsort(list_dist, axis=1)[:, :self.n_neighbors]
        
        if return_distance:
            neigh_dist = np.sort(list_dist, axis=1)[:, :self.n_neighbors]
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
    
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.
        y : array-like, shape (n_samples,)
            The target values.

        """
        self._X = np.array(X)
        self._y = np.array(y)