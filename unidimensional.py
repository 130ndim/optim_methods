import numpy as np
from utils import count


def _dichotomy_search(fn, a, b, eps=1e-6):
    intervals = [np.abs(b - a)]
    it = 0
    while np.abs(b - a) > eps:
        it += 1
        x1 = (a + b) / 2 - eps / 2.5
        x2 = (a + b) / 2 + eps / 2.5
        if fn(x1) < fn(x2):
            b = x2
        elif fn(x1) > fn(x2):
            a = x1
        else:
            a = x1
            b = x2
            break
        intervals.append(np.abs(b - a))

    return {'iterations': it,
            'min': (a + b) / 2,
            'intervals': np.array(intervals)}


def _gss(fn, a, b, eps=1e-6):
    intervals = [np.abs(b - a)]
    it = 0
    inv = (3 - np.sqrt(5)) / 2
    x1 = a + (b - a) * inv
    x2 = b - (b - a) * inv
    f1 = fn(x1)
    f2 = fn(x2)
    while np.abs(b - a) > eps:
        it += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * inv
            f1 = fn(x1)
        elif f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - (b - a) * inv
            f2 = fn(x2)
        else:
            intervals.append(np.abs(b - a))
            break
        intervals.append(np.abs(b - a))
    return {'iterations': it,
            'min': (a + b) / 2,
            'intervals': np.array(intervals)}


def _fibonacci(fn, a, b, eps=1e-6):
    intervals = [np.abs(b - a)]

    def genfib(l, r, eps):
        a, b = 0., 1.
        while a <= (r - l) / eps:
            yield a
            a, b = b, a + b
        yield a

    it = 0
    fibs = list(genfib(a, b, eps))
    n = len(fibs) - 2
    x1 = a + fibs[-3] / fibs[-1] * (b - a)
    x2 = a + fibs[-2] / fibs[-1] * (b - a)
    f1 = fn(x1)
    f2 = fn(x2)
    for i in range(1, n):
        it += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + fibs[-3-i] / fibs[-1-i] * (b - a)
            f1 = fn(x1)
        elif f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + fibs[-2-i] / fibs[-1-i] * (b - a)
            f2 = fn(x2)
        else:
            intervals.append(np.abs(x2 - x1))
            break
        intervals.append(np.abs(b - a))

    return {'iterations': n,
            'min': (a + b) / 2,
            'intervals': np.array(intervals)}



class UnidimSearch:
    def __init__(self, method: str):
        assert method in ['dichotomy', 'golden', 'fibonacci']
        self.method = method

    def __call__(self, fn, a, b, eps):

        if self.method == 'dichotomy':
            algo = _dichotomy_search
        elif self.method == 'golden':
            algo = _gss
        elif self.method == 'fibonacci':
            algo = _fibonacci
        fn = count(fn)
        res = algo(fn, a, b, eps)
        res.update({'call_count': fn.count})
        return res

    def __repr__(self):
        return f'{self.__class__.__name__}(method={self.method})'


