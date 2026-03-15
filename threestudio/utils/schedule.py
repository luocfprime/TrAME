import numpy as np

import threestudio


def piecewise_linear_schedule(pts):
    """
    Return a piecewise linear schedule.
    pts: list of tuples [(x1, y1), (x2, y2), ...]
    """
    x, y = zip(*pts)
    if not all(x[i] <= x[i + 1] for i in range(len(x) - 1)):  # if x not in ascending order, warn it
        threestudio.warn("x values are not in ascending order")
    x = np.array(x)
    y = np.array(y)
    return lambda t: np.interp(t, x, y)


def constant_schedule(value):
    return lambda x: value


schedule_functions = {
    "constant": constant_schedule,  # args: value
    "piecewise_linear": piecewise_linear_schedule,  # args: pts
}
