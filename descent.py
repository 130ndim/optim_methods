import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvals
from unidimensional import UnidimSearch

def line_search(fn, x0, step=0.01, multiplier=1.5):
    curr = fn(x0)
    if curr < fn(x0 + step) or fn(x0 - step) < curr:
        step *= -1

    while curr > fn(x0 + step):
        step *= multiplier
        x0 += step
        curr = fn(x0)

    return x0


def gradient_descent(fn, fn_grad, x0, alpha=1e-3, eps=1e-6, compute_alpha=False,
                     verbose=False, method='line', return_trajectory=False, **kwargs):
    curr_x = x0
    crit = np.linalg.norm(fn_grad(curr_x)) * eps
    i = 0
    trajectory = [curr_x]
    while True:
        i += 1
        curr_grad = fn_grad(curr_x)
        grad_norm = np.linalg.norm(curr_grad)
        if verbose and i % verbose == 0:
            print(
                f'Iteration {i}, Grad norm: {grad_norm:.6f}, '
                f'Left: {grad_norm - crit:.6f}')
        if compute_alpha:
            alpha = optimize_step(fn, curr_grad, curr_x, method=method, **kwargs)
        if np.abs(grad_norm) < crit:
            if return_trajectory:
                return trajectory
            else:
                return curr_x
        curr_x = curr_x - alpha * curr_grad
        trajectory.append(curr_x)


def optimize_step(fn, grad, x0, method='line', confidence=1e-6, **kwargs):
    objective = lambda x: fn(x0 - x * grad)
    if method == 'line':
        return line_search(objective, np.random.rand(), **kwargs)
    else:
        return UnidimSearch(method)(objective, 0, 1, eps=confidence, **kwargs)['min']


def newton(fn_grad, fn_hess, x0, eps=1e-6, verbose=False, return_trajectory=False):
    curr_x = x0
    i = 0
    crit = np.linalg.norm(fn_grad(curr_x)) * eps
    trajectory = [curr_x]
    while True:
        i += 1
        curr_grad = fn_grad(curr_x)
        curr_hess = fn_hess(curr_x)

        # if Hessian is positive definite use Cholesky decomposition
        if np.all(eigvals(curr_hess) > 0):
            d = cho_solve(cho_factor(curr_hess), curr_grad)
        else:
            d = np.linalg.inv(curr_hess).dot(curr_grad)

        grad_norm = np.linalg.norm(curr_grad)

        if verbose and i % verbose == 0:
            print(
                f'Iteration {i}, Grad norm: {grad_norm:.6f}, Left: {max(0, grad_norm - eps):.6f}')
        if grad_norm < crit:
            if return_trajectory:
                return trajectory
            else:
                return curr_x
        curr_x = curr_x - d
        trajectory.append(curr_x)

