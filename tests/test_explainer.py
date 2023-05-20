import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from tinyshap import SHAPExplainer

dataset = load_diabetes()
X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

X_train_summary = pd.DataFrame(KMeans(n_clusters=10, n_init="auto").fit(X_train).cluster_centers_, columns=X.columns)


def test__generate_coalitions():
    explainer = SHAPExplainer(model.predict, X=X_train_summary)
    coalitions = explainer._generate_coalitions()

    assert type(coalitions) == pd.DataFrame
    assert coalitions.min().min() == 0
    assert coalitions.max().max() == 1
    assert coalitions.shape[0] <= 2 ** X.shape[1]


def test__get_feature_values():
    x_test = X_test.iloc[0].values

    explainer = SHAPExplainer(model.predict, X=X_train_summary)
    coalitions = explainer._generate_coalitions()
    feature_values = explainer._get_feature_values(coalitions, x_test)

    assert type(feature_values) == pd.DataFrame
    assert feature_values.shape == coalitions.shape
    assert feature_values.columns.tolist() == coalitions.columns.tolist()


def test__get_coalition_weights():
    explainer = SHAPExplainer(model.predict, X=X_train_summary)
    coalitions = explainer._generate_coalitions()
    coalition_weights = explainer._get_coalition_weights(coalitions)

    assert type(coalition_weights) == np.ndarray
    assert coalition_weights.ndim == 1
    assert coalition_weights.shape[0] == coalitions.shape[0]


def test__explain_sample():
    x_test = X_test.iloc[0].values

    explainer = SHAPExplainer(model.predict, X=X_train_summary)
    shap_values = explainer._explain_sample(x_test)

    assert type(shap_values) == pd.Series
    assert shap_values.shape[0] == X.shape[1] + 1  # features + avg prediction
    assert np.allclose(
        float(shap_values.sum()), explainer.model(pd.DataFrame(x_test.reshape(1, -1), columns=X.columns)), rtol=0.01
    )


def test_shap_values():
    explainer = SHAPExplainer(model.predict, X=X_train_summary)
    shap_values = explainer.shap_values(X_test)

    assert shap_values.shape[0] == X_test.shape[0]
    assert shap_values.shape[1] == X_test.shape[1] + 1
    assert np.allclose(shap_values.sum(axis=1).values, explainer.model(X_test), rtol=0.01)
