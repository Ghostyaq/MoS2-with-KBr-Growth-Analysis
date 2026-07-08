import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from numpy.fft import ifft

# Full generate_signal() function 
def generate_signal(n=1500, x_start=0, x_end=1200, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    x = np.linspace(x_start, x_end, n)
    signal = np.zeros(n)
    
    sharp_centers = []
    sharp_heights = []
    num_bumps = 1
    
    for _ in range(num_bumps):
        center = np.random.randint(0, n)
        width = np.random.randint(2, 5)
        height = np.random.uniform(300, 450)
        bump = height * gaussian(n, std=width)
        bump = np.roll(bump, center - n // 2)
        signal += bump
        sharp_centers.append(center)
        sharp_heights.append(height)
    
    forced_center = np.random.randint(5, 11)
    max_height = max(sharp_heights) if sharp_heights else 400
    forced_height = max_height * np.random.uniform(0.6, 0.8)
    forced_width = np.random.randint(2, 5)
    forced_peak = forced_height * gaussian(n, std=forced_width)
    forced_peak = np.roll(forced_peak, forced_center - n // 2)
    signal += forced_peak
    sharp_centers.append(forced_center)
    sharp_heights.append(forced_height)
    
    num_hills = np.random.randint(1, 3)
    min_distance = 200
    hill_attempts, added_hills, max_attempts = 0, 0, 20
    
    while added_hills < num_hills and hill_attempts < max_attempts:
        center = np.random.randint(0, n)
        if all(abs(center - c) > min_distance for c in sharp_centers):
            width = np.random.randint(25, 50)
            height = np.random.uniform(10, 50)
            hill = height * gaussian(n, std=width)
            hill = np.roll(hill, center - n // 2)
            signal += hill
            added_hills += 1
        hill_attempts += 1
    
    freq_noise = (np.random.randn(n) + 1j * np.random.randn(n))
    low_pass_window = np.exp(-np.linspace(0, 8, n))
    freq_noise *= low_pass_window
    time_noise = np.real(ifft(freq_noise))
    signal += 1 * time_noise
    signal += 1 * np.random.randn(n)
    
    return x, signal

# Use 3 base signals 
base_signals = []
x = None
for _ in range(3):
    x, base = generate_signal()
    base_signals.append(base)

# Function to create variations 
def create_variation(base_signal):
    amplitude = np.random.uniform(0.9, 1.1)
    shift = np.random.randint(-1, 1)
    noise = np.random.normal(0, 5, size=base_signal.shape)
    shifted = np.roll(base_signal, shift)
    return amplitude * shifted + noise

# Generate variations 
num_variations = 5000
all_signals = []

for _ in range(num_variations):
    base = base_signals[np.random.randint(0, 3)]  # Randomly choose base signal
    variant = create_variation(base)
    all_signals.append(variant)

df = pd.DataFrame(np.vstack(all_signals), columns=np.round(x, 2))

# Save (optional) 
# df.to_csv("signal_variations_from_3_real_bases.csv", index=False)

# Plot examples
plt.figure(figsize=(12, 6))
for i in range(50):
    plt.plot(x, df.iloc[i], label=f"Signal {i+1}")
plt.title("Variations from 3 Randomly Generated Base Signals")
plt.xlabel("x")
plt.ylabel("Signal Height")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df.T.to_csv("/Users/mitchellhung/Desktop/Mitchell folder/2025 Internship/made_up_data.csv", index=True)
