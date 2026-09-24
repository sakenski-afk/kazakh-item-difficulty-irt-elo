import numpy as np
from scipy.stats import pearsonr, spearmanr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def regression_report(y_true_b, y_pred_b, p_true):
    """Metrics on logit-difficulty scale plus RMSE on proportion-correct scale."""
    y_true_b, y_pred_b = np.asarray(y_true_b), np.asarray(y_pred_b)
    p_pred = sigmoid(-y_pred_b)
    return {
        "RMSE_b": float(np.sqrt(np.mean((y_true_b - y_pred_b) ** 2))),
        "MAE_b": float(np.mean(np.abs(y_true_b - y_pred_b))),
        "RMSE_p": float(np.sqrt(np.mean((np.asarray(p_true) - p_pred) ** 2))),
        "Pearson": float(pearsonr(y_true_b, y_pred_b)[0]) if np.std(y_pred_b) > 1e-9 else 0.0,
        "Spearman": float(spearmanr(y_true_b, y_pred_b)[0]) if np.std(y_pred_b) > 1e-9 else 0.0,
    }
