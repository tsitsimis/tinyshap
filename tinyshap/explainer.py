from typing import Tuple

import numpy as np
import pandas as pd


class SHAPExplainer:
    def __init__(self, model: callable, X: pd.DataFrame, n_samples: int = 1000) -> None:
        """
        model : callable
            A callable that accepts a numpy Array or pandas Dataframe with features and returns predictions
            Usually the `.predict()` or `.predict_proba()` method of scikit-learn estimators but it can be
            any other function
        X : pandas.DataFrame
            Dataset for sampling random samples when estimating the SHAP values
        n_iter : int
            Number of random samples to estimate SHAP values
        """
        self.model = model
        self.X = X
        self.n_samples = n_samples

    def _generate_feature_combinations(
        self, x_test: np.ndarray, feature_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates random coalitions of features of a test sample `x_test` to estimate Shapley values
        for feature `feature_name`

        Returns 2 pandas DataFrames. The first has the values of x_test in the feature_name column and the other
        has a random value from the `self.X`
        """
        dim = self.X.shape[1]

        # Each feature's value either stays as is, or masked with a random value
        # This create a (n_iter, dim-1) DataFrame with 1 and 0. 1 means the feature value remains intact,
        # 0 means it is replaced with a random value
        mask_df = pd.DataFrame(
            np.random.choice([0, 1], size=(self.n_samples, dim - 1)), columns=self.X.drop(feature_name, axis=1).columns
        ).drop_duplicates()

        mask_df_with_feature = mask_df.copy()
        mask_df_with_feature[feature_name] = 1
        mask_df_with_feature = mask_df_with_feature[self.X.columns]

        mask_df_without_feature = mask_df.copy()
        mask_df_without_feature[feature_name] = 0
        mask_df_without_feature = mask_df_without_feature[self.X.columns]

        # Random values are sampled from the dataset
        X_samples = self.X.sample(mask_df.shape[0], replace=True)

        combinations_with_feature = pd.DataFrame(
            np.where(mask_df_with_feature, x_test, X_samples.values), columns=self.X.columns
        )
        combinations_without_feature = pd.DataFrame(
            np.where(mask_df_without_feature, x_test, X_samples.values), columns=self.X.columns
        )
        return combinations_with_feature, combinations_without_feature

    def _explain_single_feature(self, x_test: np.ndarray, feature_name: str) -> float:
        """
        Calculates Shapley values of a single sample and feature
        """
        assert x_test.ndim == 1

        combinations_with_feature, combinations_without_feature = self._generate_feature_combinations(
            x_test=x_test, feature_name=feature_name
        )
        marginal_contributions = self.model(combinations_with_feature) - self.model(combinations_without_feature)
        return np.mean(marginal_contributions)

    def _explain_all_features(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculates Shapley values of all features of a single sample
        """
        assert x_test.ndim == 1
        return np.array([self._explain_single_feature(x_test, feature_name=col) for col in self.X.columns])

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Shapley values of multiple samples in `X`
        """
        shap_values_df = X.apply(self._explain_all_features, axis=1).apply(pd.Series)
        shap_values_df.columns = self.X.columns
        return shap_values_df
