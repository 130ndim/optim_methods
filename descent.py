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