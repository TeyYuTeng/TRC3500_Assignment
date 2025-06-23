import numpy as np
from scipy.optimize import curve_fit

test = [25, 25.2, 25.4, 25.6, 25.8, 26]
delta = [i - 25 for i in test]
performance = [0.53, 0.56, 0.65, 0.84, 0.95, 0.99]

# Logistic function
def logistic(x, x0, k):
    return (0.5 / (1 + np.exp(-k * (x - x0)))) + 0.5

# Fit the curve
params, _ = curve_fit(logistic, delta, performance, p0=[0.5, 5.0])
x0, k = params

print(f"50% detection threshold (JND): {x0:.3f}")