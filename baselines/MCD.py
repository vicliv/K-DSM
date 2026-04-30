import numpy as np
from sklearn.covariance import EllipticEnvelope


class MCD:
    def __init__(self, seed=42, model_name='MCD', contamination=0.1, support_fraction=None):
        self.seed = seed
        self.model_name = model_name
        self.contamination = contamination
        self.support_fraction = support_fraction

    def fit(self, X_train, y_train=None):
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.seed,
        )
        self.model.fit(X_train)
        return self

    def predict_score(self, X):
        # mahalanobis returns squared Mahalanobis distances; higher = more anomalous
        return self.model.mahalanobis(X)
