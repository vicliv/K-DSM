import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, seed=42, model_name='KNN', n_neighbors=5, aggregation='kth'):
        assert aggregation in ('mean', 'median', 'kth'), "aggregation must be 'mean', 'median', or 'kth'"
        self.seed = seed
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation

    def fit(self, X_train, y_train=None):
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.model.fit(X_train)
        return self

    def predict_score(self, X):
        distances, _ = self.model.kneighbors(X)
        if self.aggregation == 'mean':
            return distances.mean(axis=1)
        elif self.aggregation == 'median':
            return np.median(distances, axis=1)
        else:  # kth
            return distances[:, -1]
