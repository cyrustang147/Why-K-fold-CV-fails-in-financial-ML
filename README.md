# Why K-fold CV Fails in Financial ML
A compact, reproducible project that demonstrates how standard K-fold cross-validation can produce optimistic (misleading) performance estimates in financial machine learning. (Referencing chapter 7 of 'Advances in Financial Machine Learning' by Prado.)

The simulation uses a regime-switching AR(2) return generator and a small Keras model. The goal: show K-fold (contiguous blocks) appears to perform well **in-sample** while true forward (time-aware) test performance is much worse.

## Introduction

López de Prado points out two main reasons K-fold fails in finance:

1. **Non-IID data (autocorrelation/regime shifts).**
2. **Repeated testing/model selection bias** (re-using the same set of data in both training and validation).

* Our return generator is based on an AR(2) process whose parameters switch slowly and randomly over time. As these parameters remain locally stable, the process is predictable in the short term (near the training data) but becomes unpredictable in the long term (far into the future). This captures the non-IID nature of financial data.
* In practice, financial markets often behave the same way. For example, through regime shifts such as bull and bear markets, where dynamics are stable within a regime but change abruptly when the regime shifts.
* Our Keras model trained with access to *both past and future* return data, causing some training data to be reused in validation. This demonstrates the second point above.

## Project structure

### `model.py`

* `arreturn()` — regime-switching AR(2) simulator.
* `features_label()` - builder that turns returns → lag features and labels.
* `flawed_model()` - A neural network to model future returns using past returns. The default model has 225 parameters, intentionally to overfit.
* `kfold_cv()` - An implementation of k-fold CV for the flawed_model

### `model_demos.ipynb` 

* A notebook that generates returns, runs in-sample training, K-fold CV, and out-of-sample evaluation; also generates plots.

### `demo_results.md`

* A markdown that stores and explains all the outputs from the demo notebook.

## Usage

* Here are short, runnable snippets.

```py
from model import (
    arreturn,
    features_label,
    flawed_model,
    kfold_cv
)
rng = np.random.default_rng(76)
returns, alphas = arreturn(rng)

X, Y = features_label(returns)

full_model = flawed_model(X.shape[1])

val_losses = kfold_cv(X, Y)

print("\nMean CV loss:", np.mean(val_losses))
```


## Typical Results

Typical behaviour you should observe:

<img src="https://github.com/cyrustang147/Why-K-fold-CV-fails-in-financial-ML/blob/main/plots/table1.png" width="480">

* **Low training loss** (model can fit in-sample).
* **K-fold mean CV loss** often low (appears optimistic).
* **True out-of-sample (holdout) loss** is substantially higher.
* **Interpretation:** Because K-fold contiguous blocks allow training folds to include data both before and after a validation block, the training set often contains samples from the *same local regime* as the validation block (i.e., the model effectively learned that regime). In real deployment, you do not have those future samples — hence, real OOS performance degrades.

## Parameters Tuning

* `avg_regime_length` (in simulator): increase → model memorises local regimes more (more divergence between K-fold CV and OOS).
* `q` (number of lags in features): increase q to give the model more history, causing more overfitting.
* `n_splits` in K-fold: fewer splits → larger validation blocks; effect changes how much future information is inside training folds.
* Optimiser/model capacity/epochs: increasing capacity + epochs tends to make the model memorise local regimes (more divergence between K-fold CV and OOS).


## References

Marcos Lopez De Prado, Advances in Financial Machine Learning (New Jersey: Wiley, 2018), 103-105

## License

MIT — feel free to reuse & adapt for research/teaching. 

## Contact/Author

Cyrus Tang — Github: cyrustang147


