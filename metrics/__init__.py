from .simple_metrics import accuracy, mse, mae, r2
from .binary_confusion_matrix import *
metrics_by_name = {
    "accuracy": accuracy,
    "mse": mse,
    "mae": mae,
    "r2": r2,
    "tp": true_positive,
    "fn": false_negative,
    "fp": false_positive,
    "tn": true_negative,
    "binary_confusion_matrix": binary_confusion_matrix,
    "precision": precision,
    "recall": recall,
    "f_score": f_score
}


def get_metrics_by_name(metrics):
    if isinstance(metrics, type):
        return metrics()
    elif isinstance(metrics, str):
        if metrics in metrics_by_name.keys():
            return metrics_by_name[metrics]
        else:
            raise Exception("cannot find metrics %s" % metrics)
    else:
        return metrics