import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
data = pd.read_csv('Data/Asg8/wk8_sensor_data.csv')  # Replace with your actual file path

# 2. Preprocess the data
# Calculate average temperature across all sensors
data['Average_Temp'] = data[['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']].mean(axis=1)

# Apply moving average smoothing
window_size = 15  # Adjust this based on your noise characteristics
data['Smoothed_Temp'] = data['Average_Temp'].rolling(window=window_size, center=True, min_periods=1).mean()

# 3. Find peak temperature
peak_temp = data['Smoothed_Temp'].max()
peak_time = data.loc[data['Smoothed_Temp'].idxmax(), 'Time']

# 4. Validate results
print(f"Estimated peak temperature: {peak_temp:.2f}°C at time {peak_time:.2f}s")

# 5. Plot results
plt.figure(figsize=(12, 6))

# Plot individual sensors with transparency
for i in range(1, 5):
    plt.plot(data['Time'], data[f'Sensor{i}'], alpha=0.2, label=f'Sensor {i}')

# Plot average and smoothed curves
plt.plot(data['Time'], data['Average_Temp'], 'k-', alpha=0.4, linewidth=1, label='Raw Average')
plt.plot(data['Time'], data['Smoothed_Temp'], 'r-', linewidth=2, label=f'Moving Average (window={window_size})')

# Mark the peak
plt.scatter(peak_time, peak_temp, color='red', s=100, zorder=5, 
            label=f'Peak: {peak_temp:.2f}°C')

plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Sensor Data Analysis with Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()