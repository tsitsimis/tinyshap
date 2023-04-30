# tinyshap
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
 
