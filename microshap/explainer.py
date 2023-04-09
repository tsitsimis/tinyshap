import numpy as np
import pandas as pd


class SHAPExplainer:
    def __init__(self, predict_function: callable, X: pd.DataFrame, n_iter: int=1000) -> None:
        self.predict_function = predict_function
        self.X = X
        self.n_iter = n_iter

    def _generate_feature_combinations(self, x_test: np.ndarray, feature_name: str) -> np.ndarray:
        dim = self.X.shape[1]

        # Each feature's value either stays as is, or masked with a random value
        # This create a (n_iter, dim-1) DataFrame with 1 and 0. 1 means the feature value remains intact,
        # 0 means it is replaced with a random value
        mask_df = pd.DataFrame(
            np.random.choice([0, 1], size=(self.n_iter, dim-1)),
            columns=self.X.drop(feature_name, axis=1).columns
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
            np.where(mask_df_with_feature, x_test, X_samples.values),
            columns=self.X.columns
        )
        combinations_without_feature = pd.DataFrame(
            np.where(mask_df_without_feature, x_test, X_samples.values),
            columns=self.X.columns
        )
        return combinations_with_feature, combinations_without_feature

    def _explain_single_feature(self, x_test: np.ndarray, feature_name: str) -> float:
        assert x_test.ndim == 1
        
        combinations_with_feature, combinations_without_feature = (
            self._generate_feature_combinations(x_test=x_test, feature_name=feature_name)
        )
        marginal_contributions = (
            self.predict_function(combinations_with_feature) - self.predict_function(combinations_without_feature)
        )
        return np.mean(marginal_contributions)
    
    def _explain_all_features(self, x_test: np.ndarray) -> np.ndarray:
        assert x_test.ndim == 1
        return np.array([self._explain_single_feature(x_test, feature_name=col) for col in self.X.columns])
    
    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        shap_values_df = X.apply(self._explain_all_features, axis=1).apply(pd.Series)
        shap_values_df.columns = self.X.columns
        return shap_values_df
