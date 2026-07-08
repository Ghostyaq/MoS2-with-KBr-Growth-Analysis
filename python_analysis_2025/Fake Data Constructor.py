#Description: Generates fake data?
#How to Use: I don't really see why anyone would want to use this... Anyways:
'''
Generates a 1D signal with:
    - 1 to 4 sharp Gaussian peaks at random locations
    - 1 forced sharp Gaussian peak near the start (index 5-10)
    - 1 to 2 broad Gaussian hills placed away from sharp peaks
    - Low frequency structured noise and white noise
    
    Parameters:
    - n: length of the signal array
    - x_start, x_end: range for x-axis values
    - random_seed: int or None for reproducibility
    
    Returns:
    - x: numpy array of x-axis values
    - signal: numpy array of the generated signal
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from numpy.fft import ifft
import time

def generate_signal(n=1500, x_start=0, x_end=1200, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    x = np.linspace(x_start, x_end, n)
    signal = np.zeros(n)
    
    # 1. Sharp Gaussian peaks
    sharp_centers = []
    sharp_heights = []
    num_bumps = np.random.randint(1, 5)
    
    for _ in range(num_bumps):
        center = np.random.randint(0, n)
        width = np.random.randint(2, 5)
        height = np.random.uniform(300, 450)
        
        bump = height * gaussian(n, std=width)
        bump = np.roll(bump, center - n // 2)
        signal += bump
        
        sharp_centers.append(center)
        sharp_heights.append(height)
    
    # 2. Forced sharp peak near start (index 5-10)
    forced_center = np.random.randint(5, 11)
    if sharp_heights:
        max_height = max(sharp_heights)
    else:
        max_height = 400
        
    forced_height = max_height * np.random.uniform(0.6, 0.8)
    forced_width = np.random.randint(2, 5)
    
    forced_peak = forced_height * gaussian(n, std=forced_width)
    forced_peak = np.roll(forced_peak, forced_center - n // 2)
    signal += forced_peak
    
    sharp_centers.append(forced_center)
    sharp_heights.append(forced_height)
    
    # 3. Broad hills away from sharp peaks
    num_hills = np.random.randint(1, 3)
    min_distance = 200
    hill_attempts = 0
    max_attempts = 20
    added_hills = 0
    
    while added_hills < num_hills and hill_attempts < max_attempts:
        center = np.random.randint(0, n)
        if all(abs(center - c) > min_distance for c in sharp_centers):
            width = np.random.randint(100, 300)
            height = np.random.uniform(40, 75)
            hill = height * gaussian(n, std=width)
            hill = np.roll(hill, center - n // 2)
            signal += hill
            added_hills += 1
        hill_attempts += 1
    
    # 4. Low-frequency structured noise
    freq_noise = (np.random.randn(n) + 1j * np.random.randn(n))
    low_pass_window = np.exp(-np.linspace(0, 8, n))
    freq_noise *= low_pass_window
    time_noise = np.real(ifft(freq_noise))
    
    # 5. Add noise to signal
    signal += 5 * time_noise
    signal += 5 * np.random.randn(n)
    
    return x, signal


# Example usage:
random_seed = int(time.time())
x, signal = generate_signal(random_seed=random_seed)
plt.figure(figsize=(12, 4))
plt.plot(x, signal, color='darkgreen')
plt.title("Random Data with Forced Sharp Peak and Conditional Broad Hills")
plt.xlabel("x")
plt.ylabel("Signal")
plt.grid(True)
plt.tight_layout()
plt.show()
