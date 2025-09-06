import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Tuple
from sklearn.model_selection import KFold

# -----------------------------
# Regime-switching AR(2) sim
# -----------------------------
def arreturn(rng: np.random._generator.Generator, n: int = 10000, avg_regime_length: float = 500.0):
    """
    Simulate an AR(2) time series with occasional regime switches.
    - rng: numpy Generator
    - n: length of series
    - avg_regime_length: expected number of steps between regime switches
    Returns:
      r: array shape (n,) of returns
      alphalist: array shape (n,2) of (alpha1, alpha2) used at each time
    """
    e = rng.normal(scale=0.03, size=n)              # i.i.d. noise
    r = np.zeros(n)                                 # return series
    # list of candidate AR(2) coefficient pairs 
    choices = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]

    alphas = choices[rng.integers(len(choices))]    # initial regime (random)
    alphalist = [alphas, alphas]                    # seed list; will append one per time step

    p_switch = 1.0 / avg_regime_length              # probability of switching at each step

    for t in range(2, n):
        # decide if we change regime at this time step
        if rng.random() < p_switch:
            # choose a new regime different from current
            possible = [a for a in choices if a != alphas]
            alphas = possible[rng.integers(len(possible))]
        alpha1, alpha2 = alphas
        # AR(2) update: r[t] = alpha1*r[t-1] + alpha2*r[t-2] + noise
        r[t] = alpha1 * r[t-1] + alpha2 * r[t-2] + e[t]
        alphalist.append(alphas)

    return r, np.array(alphalist)


# -----------------------------
# Build lag-features & labels
# -----------------------------
def features_label(returns: np.ndarray, q: int = 4):
    """
    Produce X, Y for supervised learning, q is the lag in features.
    - For sample i: X[i] = [r[i], r[i+1], ..., r[i+q-1]]
      Y[i] = r[i+q]
    """
    n = len(returns) - q
    X = np.zeros((n, q))
    Y = np.zeros(n)

    for i in range(n):
        for j in range(q):
            X[i, j] = returns[i + j]
        Y[i] = returns[i + q]
    return X, Y


# -----------------------------
# Keras model factory
# -----------------------------
def flawed_model(input_dim: int, learning_rate: float = 1e-4):
    """
    Returns a freshly-compiled Keras model. 
    - input_dim : dimension of the features
    - learning_rate: learning rate used in Adam
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    return model


# -----------------------------
# K-fold cross-validation (contiguous blocks)
# -----------------------------

def kfold_cv(X, Y, n_splits: int = 2):
    """
    Implement a k-fold CV for our flawed_model.
    - n_splits: number of groups (k)
    Return a list of the fold's loss
    """

    kf = KFold(n_splits=n_splits, shuffle=False)  # contiguous blocks (no shuffle)

    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        # clear session to avoid leftover graph/weights in some TF versions
        keras.backend.clear_session()

        # fresh model each fold
        model = flawed_model(X.shape[1])

        # fit on fold's training set
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            verbose=0
        )

        # Evaluate on validation set
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        val_losses.append(val_loss)
    return val_losses
