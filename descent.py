import numpy as np
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
                     verbose=False, method='line', **kwargs):
    curr_x = x0
    curr_grad = fn_grad(x0)
    prev_norm = np.linalg.norm(curr_grad)
    i = 0
    while True:
        i += 1
        if verbose and i % verbose == 0:
            print(
                f'Iteration {i}, Norm difference = {np.linalg.norm(curr_grad):.6f}')
        if compute_alpha:
            alpha = optimize_step(fn, curr_grad, curr_x, method=method, **kwargs)
        curr_x = curr_x - alpha * curr_grad
        curr_grad = fn_grad(curr_x)
        curr_norm = np.linalg.norm(curr_grad)
        if np.abs(prev_norm - curr_norm) < eps:
            break
        else:
            prev_norm = curr_norm
    return curr_x


def optimize_step(fn, grad, x0, method='line', **kwargs):
    objective = lambda x: fn(x0 - x * grad)
    if method == 'line':
        return line_search(objective, np.random.rand(), **kwargs)
    else:
        return UnidimSearch(method)(objective, 0, 1, **kwargs)['min']



