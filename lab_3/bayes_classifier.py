import numpy as np


def fit_bayes(X_train, y_train):
    means, covariances = [], []
    for label in (0, 1):
        x = X_train[y_train == label]
        mean = x.mean(axis=0)
        centered = x - mean
        covariance = centered.T @ centered / len(x)
        ridge = max(1e-3 * np.trace(covariance) / 2, 1e-6)
        means.append(mean)
        covariances.append(covariance + ridge * np.eye(2))
    return np.array(means), np.array(covariances)


def gaussian_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def bayes_predict(x, means, covariances, priors):
    scores = np.empty((len(x), 2))
    for label in (0, 1):
        delta = x - means[label]
        covariance = covariances[label]
        _, logdet = np.linalg.slogdet(covariance)
        solved = np.linalg.solve(covariance, delta.T).T
        distance = np.sum(delta * solved, axis=1)
        scores[:, label] = np.log(priors[label]) - 0.5 * (2 * np.log(2 * np.pi) + logdet + distance)
    return scores.argmax(axis=1)
