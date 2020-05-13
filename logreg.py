import numpy as np
from scipy.special import expit
from descent import gradient_descent


class LogisticRegression:
    def __init__(self, alpha=1e-3, compute_alpha=False, eps=1e-6, l2=None,
                 verbose=False, method='line'):
        self.alpha = alpha
        self.compute_alpha = compute_alpha
        self.method = method
        self.eps = eps
        self.l2 = l2
        self.verbose = verbose
        self.W = None

    def fit(self, X, y):

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        W = np.random.normal(size=(X.shape[1], 1))

        if self.l2 is not None:
            Q = lambda W: np.mean(np.logaddexp(0, -y * X.dot(W))) + \
                          self.l2 / 2 * np.linalg.norm(W)
            Q_grad = lambda W: (-np.mean(X * y * expit(-X.dot(W) * y), axis=0,
                                         keepdims=True).T + \
                                self.l2 * W)
        else:
            Q = lambda W: np.mean(np.logaddexp(0, -y * X.dot(W)))
            Q_grad = lambda W: -np.mean((X * y).T.dot(expit(-X.dot(W) * y)),
                                        axis=0)

        self.W = gradient_descent(Q, Q_grad, W,
                                  eps=self.eps,
                                  alpha=self.alpha,
                                  compute_alpha=self.compute_alpha,
                                  verbose=self.verbose,
                                  method=self.method)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.sign(X.dot(self.W))