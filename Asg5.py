import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset
data = pd.read_csv('data/Asg5/melbDailyTmps.csv', header = None)  # Replace with your file path
data.columns = ['Product', 'Station', 'Year', 'Month', 'Day', 'MaxTemp', 'Period', 'Quality']

# Preprocess data: Filter out missing values (-10°C) and invalid rows
data = data[data['MaxTemp'] != -10]
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data['DayOfYear'] = data['Date'].dt.dayofyear  # Convert date to day of year (1-365)

# Extract clean data
x_data = data['DayOfYear'].values
y_data = data['MaxTemp'].values

# Function to predict temperature for any given date
def predict_temp(year, month, day):
    date = datetime(year=year, month=month, day=day)
    day_of_year = date.timetuple().tm_yday  # Convert date to day of year (1-366)
    predicted_temp = sine_model(day_of_year, A_fit, f_fit, phi_fit, C_fit)
    return predicted_temp

# Define the sine wave model
def sine_model(x, A, f, phi, C):
    return A * np.sin(f * x + phi) + C

# Initial parameter guesses
A_guess = 10  # Amplitude (half of typical seasonal variation)
f_guess = (2 * np.pi) / 365  # Yearly frequency
phi_guess = -np.pi / 2  # Phase shift to align peak with summer (Jan)
C_guess = 15  # Average temperature
initial_guess = [A_guess, f_guess, phi_guess, C_guess]

# Fit the model
params, _ = curve_fit(sine_model, x_data, y_data, p0=initial_guess)
A_fit, f_fit, phi_fit, C_fit = params

# Estimate temperature for 17/05/2023 (DayOfYear = 137)
target_day = 137  # 17 May is the 137th day of the year
predicted_temp = sine_model(target_day, A_fit, f_fit, phi_fit, C_fit)

print(f"Predicted temperature for 17/05/2023: {predicted_temp:.1f}°C")

# Estimate temperature for 30/04/2025
print(f"Predicted temperature for 30/04/2025: {predict_temp(2025, 4, 30):.2f}°C")

# (Optional) Plot the fitted model
plt.scatter(x_data, y_data, s=1, label='Actual Data')
x_fit = np.linspace(1, 365, 365)
plt.plot(x_fit, sine_model(x_fit, A_fit, f_fit, phi_fit, C_fit), 'r-', label='Fitted Sine Wave')
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()