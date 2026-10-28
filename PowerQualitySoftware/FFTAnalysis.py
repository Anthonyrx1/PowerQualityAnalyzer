import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt


#Data reading
df = pd.read_csv('CreatedSyntheticData.csv', usecols=['t', 'combine']) 

#Extracting time and voltage values int array 
t = df['t'].values
combine = df['combine'].values

print ("Time array: ", t[:10])
print("Voltage array: ", combine[:10])


#time step voltage dt, sampling frequency computations
dt = t[1] - t[0]
fs = 1.0/dt

print("time step voltage dt:", dt)
print("Sampling frequency (Hz):", fs)

N = len(combine)

#Applying Fast Fourier Transform with frequency bins
fft_values = fft(combine) 
frequencies = fftfreq(N, dt)

print("Frequencies: ", frequencies[:10])
print("FFT Magnitudes: ", np.abs(fft_values[:10]))

#retrieval of positive frequency values
pos_freq_index = frequencies > 0
pos_frequencies = frequencies[pos_freq_index]
pos_magnitudes = np.abs(fft_values[pos_freq_index])
pos_mag_adj = pos_magnitudes / (N / 2)

#Optomization for plotting in Hz
max_freq = 1000
freq_mask = pos_frequencies <= max_freq



# Calculate RMS of voltage signal
voltage_rms = np.sqrt(np.mean(combine**2))
print(f"RMS Voltage: {voltage_rms:.2f} V")
print(df)


# Find fundamental (peak magnitude in positive frequencies)
fundamental_idx = np.argmax(pos_magnitudes)
fundamental_freq = pos_frequencies[fundamental_idx]
print(f"Detected Fundamental Frequency: {fundamental_freq:.2f} Hz")


#Calculation of Total Harmonic Disortion

harmonics = []
harmonicfreqs = []
harmonic_num = 50

fund_mag = pos_magnitudes[fundamental_idx]

#loop for finding the nearest frequency to THD for all samples
for i in range(2, harmonic_num + 1):
    targetfreq = i * fundamental_freq

    
    idx = np.argmin(np.abs(pos_frequencies - targetfreq))

    #Storing magnitude at selected frequency
    harmonics.append(pos_magnitudes[idx])
    harmonicfreqs.append(pos_frequencies[idx])

harmonics = np.array(harmonics)

#Calculation of THD
def THDcalculate(harmonics, fund_mag):
    THD_calc = 100 * np.sqrt(np.sum(harmonics**2))/ fund_mag

    return THD_calc
thd = THDcalculate(harmonics, fund_mag)
print(f"Total Harmonic Disortion(THD): {thd:.2f}%")



plt.figure(figsize=(10, 12))

#Time domain
plt.subplot(2, 1, 1) 
plt.plot(t[:1000], combine[:1000])
plt.title("Time-Domain Voltage")
plt.xlabel("Time (S)")
plt.ylabel("Voltage (V)")
plt.grid(True)

#Frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(pos_frequencies[freq_mask], pos_mag_adj[freq_mask])
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()