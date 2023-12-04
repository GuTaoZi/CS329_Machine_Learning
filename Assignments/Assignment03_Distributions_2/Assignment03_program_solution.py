import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def predict(features, weights):
    return softmax(np.dot(features, weights))


def train(features, targets, epochs, learning_rate):
    D = features.shape[1]
    C = targets.shape[1]
    weights = np.zeros(shape=(D + 1, C))
    for _ in range(epochs):
        features_bias = np.c_[features, np.ones(features.shape[0])]
        weights -= (
            learning_rate
            * np.dot(features_bias.T, (predict(features_bias, weights) - targets))
            / features.shape[0]
        )
    return weights.flatten()


if __name__ == "__main__":
    il = input().split()
    N, D, C, E = map(int, il[:-1])
    L = float(il[-1])
    features = []
    for _ in range(N):
        features.append(list(map(float, input().split())))
    targets = []
    for _ in range(N):
        targets.append(list(map(int, input().split())))
    features = np.array(features)
    targets = np.array(targets)
    weights = train(features, targets, E, L)
    print("\n".join([f"{w:.3f}" for w in weights]))
