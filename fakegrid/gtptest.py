import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, Wavelet
from scipy.signal import morlet2

# Define parameters
srate = 1000  # Sampling rate (Hz)
duration = 2  # Duration (seconds)
freq = 50     # Frequency (Hz)

# Create sine wave signal
time = np.arange(0, duration, 1/srate)
sig = np.sin(2 * np.pi * freq * time)

# Calculate scales
dt = 1 / srate
dj = 0.1
s0 = 2 * dt
J = int(np.log2(sig.shape[0] * dt / s0) / dj)
scales = s0 * 2 ** (dj * np.arange(J))

# Calculate wavelet transform
cwt = ssq_cwt(sig, scales, wavelet='morlet')

# Plot results
plt.figure(figsize=(10, 5))
plt.imshow(np.abs(cwt), aspect='auto', cmap='jet', origin='lower',)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Transform of 50 Hz Sine Signal')
plt.colorbar()
plt.show()

