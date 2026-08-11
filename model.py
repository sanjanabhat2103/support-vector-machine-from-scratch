"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    x = np.asarray(x, dtype = float)
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    std[std == 0.0] = 1.0
    return (x - mean) / std

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    return {'w': np.zeros(n_features), 'b': 0.0}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    weights = np.asarray(params['w'], dtype = float)
    bias = params['b']
    return np.asarray(x @ weights + bias, dtype = float)

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return max(0, 1 - y * score)

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    w = params["w"]
    scores = compute_scores(x, params)
    hinge = np.maximum(0.0, 1.0 - y * scores)
    obj_fun = np.mean(hinge) + reg_lambda * np.dot(w, w)
    return float(obj_fun)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    scores = compute_scores(x, params)
    margins = 1 - y * scores
    active = margins > 0
    n = x.shape[0]
    dw = -x[active].T @ y[active] / n + 2 * reg_lambda * params["w"]
    db = -np.sum(y[active]) / n
    return {'dw': dw, 'db': float(db)}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    w = params["w"]
    b = params["b"]
    dw = grads["dw"]
    db = grads["db"]
    new_w = w - learning_rate * dw
    new_b = b - learning_rate * db
    return {'w': new_w, 'b': new_b}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    scores = compute_scores(x, params)
    preds = np.where(scores >= 0, 1, -1)
    return np.asarray(preds)

# Step 11 - accuracy_score
import numpy as np

def accuracy_score(y_pred, y_true):
    return float(np.mean(y_true == y_pred))

