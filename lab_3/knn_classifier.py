import numpy as np


def knn_predict(X_train, y_train, x, ks=(1, 3, 5), seed=2026, batch_size=256):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    order = np.random.default_rng(seed).permutation(len(X_train))
    train = ((X_train - mean) / std)[order]
    labels = y_train[order]
    x = (x - mean) / std
    predictions = {k: np.empty(len(x), dtype=np.int8) for k in ks}

    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        distances = np.sum((batch[:, None, :] - train[None, :, :]) ** 2, axis=2)
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, :max(ks)]
        neighbor_labels = labels[neighbors]
        for k in ks:
            predictions[k][start:start + len(batch)] = neighbor_labels[:, :k].sum(axis=1) > k / 2
    return predictions
