import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def linear_model(x, a, b):
    return a * x + b

def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Load CSV file
data = pd.read_csv("Data/Asg3/wk3_assignment.csv", names=["X", "Y"], header=None)


# Assuming the CSV has two columns: 'X' and 'Y'
x = data['X'].values
y = data['Y'].values

# Scatter plot
plt.scatter(x, y, label="Data", color='blue')

# Fit linear model
popt_lin, _ = curve_fit(linear_model, x, y)
y_pred_lin = linear_model(x, *popt_lin)

# Fit exponential model
popt_exp, _ = curve_fit(exponential_model, x, y, maxfev=10000)
y_pred_exp = exponential_model(x, *popt_exp)

# Compute R²
r2_lin = r2_score(y, y_pred_lin)
r2_exp = r2_score(y, y_pred_exp)

# Compute SSE and SST
sse_lin = np.sum((y - y_pred_lin) ** 2)
sse_exp = np.sum((y - y_pred_exp) ** 2)
sst = np.sum((y - np.mean(y)) ** 2)

# Plot fitted curves
plt.plot(x, y_pred_lin, label=f"Linear Fit (R²={r2_lin:.4f})", color='red')
plt.plot(x, y_pred_exp, label=f"Exponential Fit (R²={r2_exp:.4f})", color='green')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Scatter Plot with Linear and Exponential Fits")
plt.show()

# Print statistics
print("SST: {:.4f}".format(sst))
print("Linear Fit: a = {:.4f}, b = {:.4f}, R² = {:.4f}, SSE = {:.4f}, SST = {:.4f}".format(*popt_lin, r2_lin, sse_lin, sst))
print("Exponential Fit: a = {:.4f}, b = {:.4f}, R² = {:.4f}, SSE = {:.4f}, SST = {:.4f}".format(*popt_exp, r2_exp, sse_exp, sst))
