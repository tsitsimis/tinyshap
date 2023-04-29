import math
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def shap_kernel(coalition: List[int], dim: int) -> float:
    size = sum(coalition)
    return (dim - 1) / (math.comb(dim, size) * size * (dim - size))


class SHAPExplainer:
    def __init__(self, model: callable, X: pd.DataFrame, n_samples: int = None) -> None:
        """
        model : callable
            A callable that accepts a numpy Array or pandas Dataframe with features and returns predictions
            Usually the `.predict()` or `.predict_proba()` method of scikit-learn estimators but it can be
            any other function
        X : pandas.DataFrame
            Dataset for sampling random samples when estimating the SHAP values
        n_samples : int
            Number of random samples to estimate SHAP values
        """
        self.model = model
        self.X = X

        if n_samples is None:
            self.n_samples = 2 ** self.X.shape[1]
        else:
            self.n_samples = n_samples

    def _generate_coalitions(self) -> pd.DataFrame:
        """
        Generates random coalitions (array of 0s and 1s)

        It first generates random numbers between 0 and 2**dim-1, converts them to
        binary representation and then to binary arrays
        """
        dim = self.X.shape[1]

        random_numbers = np.random.choice(2**dim, size=self.n_samples, replace=True).tolist()
        binary_numbers = [np.binary_repr(x, width=dim) for x in random_numbers]
        binary_array = np.array([list(x) for x in binary_numbers], dtype=int)

        coalitions = pd.DataFrame(binary_array, columns=self.X.columns)

        # Coalitions where all values are random or exactly equal to the test sample
        # don't provide any meaningful information
        coalitions = coalitions.loc[(coalitions.sum(axis=1) > 0) & (coalitions.sum(axis=1) < dim)]
        return coalitions

    def _get_feature_values(self, coalitions: pd.DataFrame, x_test: np.ndarray) -> pd.DataFrame:
        """
        Converts coalitions to features by replacing 1s to actual feature values and 0s to random values
        """
        # Sample with replacement in case the number of samples exceeds the number of rows
        X_samples = self.X.sample(coalitions.shape[0], replace=True)
        feature_values = pd.DataFrame(np.where(coalitions, x_test, X_samples.values), columns=self.X.columns)
        return feature_values

    def _get_coalition_weights(self, coalitions: pd.DataFrame) -> np.ndarray:
        """
        Calculates the weight of each coalition
        """
        dim = self.X.shape[1]

        weights = coalitions.apply(lambda row: shap_kernel(coalition=row.tolist(), dim=dim), axis=1)
        weights = weights.values
        return weights

    def _explain_sample(self, x_test: np.ndarray) -> pd.Series:
        """
        Calculates SHAP values of a single sample using approximate KernelSHAP
        """
        assert x_test.ndim == 1

        coalitions = self._generate_coalitions()
        feature_values = self._get_feature_values(coalitions, x_test)
        predictions = self.model(feature_values)
        weights = self._get_coalition_weights(coalitions)

        # y_pred_mean = self.model(self.X.mean().to_frame().T.values)[0]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(coalitions, predictions, sample_weight=weights)  #  - y_pred_mean
        shap_values = pd.Series(data=lr.coef_, index=self.X.columns)
        shap_values["avg_prediction"] = lr.intercept_  # y_pred_mean
        return shap_values

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates SHAP values of multiple samples in `X`
        """
        shap_values = X.apply(self._explain_sample, axis=1)
        return shap_values
