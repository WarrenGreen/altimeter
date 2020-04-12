from math import *


def gaussian_f(mu, sigma2, x):
    """
    Gaussian likelihood with a mean and variance at input, `x`
    Args:
        mu: mean
        sigma2: variance
        x:

    Returns:
        (float) probability
    """
    coefficient = 1.0 / sqrt(2.0 * pi * sigma2)
    exponential = exp(-0.5 * (x - mu) ** 2 / sigma2)
    return coefficient * exponential


def update(mean1, var1, mean2, var2):
    """
    Takes in two means and two squared variance terms, and
    returns updated gaussian parameters.
    Args:
        mean1:
        var1:
        mean2:
        var2:

    Returns:
        (Tuple[float, float]): mean and variance
    """
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)

    return new_mean, new_var


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    """
    Takes in two means and two squared variance terms, and
    returns updated gaussian parameters, after motion.
    Args:
        mean1:
        var1:
        mean2:
        var2:

    Returns:
        (Tuple[float, float]): mean and variance
    """
    new_mean = mean1 + mean2
    new_var = var1 + var2

    return new_mean, new_var
