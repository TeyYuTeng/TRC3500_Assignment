import numpy as np
from scipy.optimize import curve_fit

# Data
delta_h = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
alert_rate = np.array([5, 25, 45, 75, 85, 95])

# Logistic function
def logistic(x, x0, k):
    return 100 / (1 + np.exp(-k * (x - x0)))

# Fit the curve
params, _ = curve_fit(logistic, delta_h, alert_rate, p0=[1.0, 5.0])
x0, k = params

print(f"50% detection threshold (JND): {x0:.2f} mm")