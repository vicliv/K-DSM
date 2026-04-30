import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOF:
    def __init__(self, seed=42, model_name='LOF', n_neighbors=20):
        self.seed = seed
        self.model_name = model_name
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train=None):
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        self.model.fit(X_train)
        return self

    def predict_score(self, X):
        # decision_function returns negative outlier factor; negate so higher = more anomalous
        return -self.model.decision_function(X)
