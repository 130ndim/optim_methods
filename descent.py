import numpy as np


def line_search(fn, x0, step=0.01, multiplier=1.5):
    curr = fn(x0)
    if curr < fn(x0 + step) or fn(x0 - step) < curr:
        step *= -1

    while curr > fn(x0 + step):
        step *= multiplier
        x0 += step
        curr = fn(x0)

    return x0


def gradient_descent(fn, fn_grad, x0, alpha=1e-3, eps=1e-6,
                     compute_alpha=False, verbose=False, **kwargs):
    curr_x = x0
    curr_grad = fn_grad(x0)
    start_norm = np.linalg.norm(curr_grad)
    i = 1
    while np.linalg.norm(curr_grad) > eps * start_norm:
        if verbose:
            print(
                f'Iteration {i}, Norm difference = {np.linalg.norm(curr_grad) - eps * start_norm:.6f}')
        if compute_alpha:
            alpha = optimize_step(fn, curr_grad, curr_x, **kwargs)
        curr_x -= alpha * curr_grad
        curr_grad = fn_grad(curr_x)
        i += 1
    return curr_x


def optimize_step(fn, grad, x0, **kwargs):
    return line_search(lambda x: fn(x0 - x * grad), np.random.rand(), **kwargs)

