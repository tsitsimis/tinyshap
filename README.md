# tinyshap
<a href="https://pypi.org/project/tinyshap" target="_blank">
    <img src="https://img.shields.io/pypi/v/tinyshap?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/tinyshap" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/tinyshap.svg?color=%2334D058" alt="Supported Python versions">
</a>

![](./assets/demo-dependency-plot.png)

A minimal implementation of the SHAP algorithm using the KernelSHAP method. In less then 100 lines of code, this repo serves as an educational resource to understand how SHAP works without all the complexities of a production-level package.

## Installation
```bash
pip install tinyshap
```

## Example usage
```python
from tinyshap import SHAPExplainer

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Explain predictions
explainer = SHAPExplainer(model.predict, X=X_train.mean().to_frame().T)
contributions = explainer.shap_values(X)
```

See complete [notebook](./notebooks/demo.ipynb)

## Resources
* [A Unified Approach to Interpreting Model Predictions (arXiv)](https://arxiv.org/abs/1705.07874)
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shap.html#kernelshap)


## Licence
MIT
 
