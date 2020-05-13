import numpy as np
from scipy.special import expit
from descent import gradient_descent, newton


class LogisticRegression:
    def __init__(self, alpha=1e-3, compute_alpha=False, eps=1e-6, l2=0,
                 verbose=False, solver='gd', method='line'):
        assert solver in ['gd', 'newton']
        self.alpha = alpha
        self.compute_alpha = compute_alpha
        self.method = method
        self.eps = eps
        self.l2 = l2
        self.verbose = verbose
        self.solver = solver
        self.W = None

    def fit(self, X, y):

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        W = np.random.normal(size=(X.shape[1], 1))

        Q = lambda W: np.mean(np.logaddexp(0, -y * X.dot(W))) + \
                      self.l2 / 2 * np.linalg.norm(W)
        Q_grad = lambda W: -(X * y).T.dot(expit(-X.dot(W) * y)) / X.shape[0] + \
                           self.l2 * W
        Q_hess = lambda W: (X * expit(-X.dot(W) * y)).T.dot(
            X * expit(X.dot(W) * y)) / X.shape[0] + \
                           self.l2 * np.eye(X.shape[1])

        if self.solver == 'gd':
            self.W = gradient_descent(Q, Q_grad, W,
                                      eps=self.eps,
                                      alpha=self.alpha,
                                      compute_alpha=self.compute_alpha,
                                      verbose=self.verbose,
                                      method=self.method)
        if self.solver == 'newton':
            self.W = newton(Q_grad, Q_hess, W,
                            eps=self.eps,
                            verbose=self.verbose)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.sign(X.dot(self.W))