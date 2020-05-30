import numpy as np
import numpy.linalg as la

def matrix_given_condn(condition_number, n, random_state=777):
    r = np.sqrt(condition_number)
    np.random.seed(random_state)
    A = np.random.randn(n, n)
    u, s, v = la.svd(A)
    h, l = np.max(s), np.min(s)  # highest and lowest eigenvalues (h / l = current cond number)

    # linear stretch: f(x) = a * x + b, f(h) = h, f(l) = h/r, cond number = h / (h/r) = r
    def f(x):
        return h * (1 - ((r - 1) / r) / (h - l) * (h - x))

    new_s = f(s)
    new_A = (u * new_s).dot(v.T)  # make inverse transformation (here cond number is sqrt(k))
    new_A = new_A.dot(new_A.T)  # make matrix symmetric and positive semi-definite (cond number is just k)
    return new_A