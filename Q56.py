import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image in grayscale
image = cv2.imread('Image3.png', cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform
f_transform = np.fft.fft2(image)
f_shifted = np.fft.fftshift(f_transform)  # Shift low frequencies to center
magnitude_spectrum = np.abs(f_shifted)

# Apply log transform for visualization
log_amplitude = np.log1p(magnitude_spectrum)

# Display the original and the log amplitude spectrum
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Log Amplitude Spectrum")
plt.imshow(log_amplitude, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
