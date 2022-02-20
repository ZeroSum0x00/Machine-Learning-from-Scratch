from .simple_metrics import accuracy, mse, mae, r2

metrics_by_name = {
    "accuracy": accuracy,
    "mse": mse,
    "mae": mae,
    "r2": r2
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

