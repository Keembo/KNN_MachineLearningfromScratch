"""Classifier with Nearest Neighbors"""

# Import libraries
import numpy as np

# Import created functions and classes
from._base import _get_weights
from._base import NearestNeighbor

class KNeighborsClassifier(NearestNeighbor):
    """
    Classifier with Nearest Neighbors

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
        Other weighting schemes can be used by subclassing :class:`NearestNeighbor`.
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent to using
        manhattan_distance (l1), and euclidean_distance (l2) for p = 2.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Unique classes found in the training data.
    n_classes_ : int
        Number of unique classes found in the training data.

    Methods
    -------
    predict_proba(X)
        Predict the class probabilities for the samples in X.
    predict(X)
        Predict the labels for the samples in X.
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
        
    def predict_proba(self, X):
        """
        Predict the class probabilities for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        probas : array of shape (n_samples, n_classes)
            Class probabilities of shape (n_samples, n_classes). The columns correspond
            to the classes in sorted order, as they appear in the attribute `classes_`.
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
        neigh_y = _y[neigh_ind]
        n_queries = X.shape[0]

        self.classes_ = np.unique(neigh_y)
        n_classes = len(self.classes_)

        neigh_proba = np.empty((n_queries, n_classes))
        for i in range(n_queries):
            neigh_y_i = neigh_y[i]
            for j, class_ in enumerate(self.classes_):
                i_class = (neigh_y_i == class_).astype(int)
                if self.weights == 'uniform':
                    class_counts_ij = np.sum(i_class)
                else:
                    weights_i = weights[i]
                    class_counts_ij = np.dot(weights_i, i_class)
                neigh_proba[i, j] = class_counts_ij
        
    # Normalize counts (Get Probability)
    def predict(self, X):
        """
        Predict the labels for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted labels for the samples in X.
        """
        # Predict neighbor probability
        neigh_proba = self.predict_proba(X)

        # Predict y
        ind_max = np.argmax(neigh_proba, axis=1)
        y_pred = self.classes_[ind_max]

        return y_pred