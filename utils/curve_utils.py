import numpy as np


def fit_curve(t, values):
    # parabola when we have enough points, straight line otherwise
    degree = 2 if len(t) >= 5 else 1
    return np.polyfit(t, values, degree)


def fit_error(coeffs, t, values):
    # sum of squared distances between the curve and the real points
    return ((np.polyval(coeffs, t) - values) ** 2).sum()


def curve_speed(coeffs, t):
    # speed of the curve at time t (px/frame)
    return np.polyval(np.polyder(coeffs), t)