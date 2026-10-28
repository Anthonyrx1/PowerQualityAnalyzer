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

#Hann-windowed spectrum used ONLY for THD. The record is not an exact whole number of cycles,
#so with no window the peak height depends on where the tone falls between FFT bins (scalloping)
#and the fundamental/harmonics get under-read by different amounts.
win_mags = np.abs(fft(combine * np.hanning(N)))[pos_freq_index]

#Frequency resolution (Hz per FFT bin)
bin_res = fs / N

#Magnitude of a tone = energy summed over its main lobe (+/-2 bins), which doesn't depend
#on the tone's offset from the nearest bin
def band_mag(mags, center, k=2):
    lo = max(center - k, 0)
    return np.sqrt(np.sum(mags[lo:center + k + 1] ** 2))

fund_idx_w = np.argmax(win_mags)
fund_mag = band_mag(win_mags, fund_idx_w)

#loop for finding the peak near each harmonic of the fundamental
for i in range(2, harmonic_num + 1):
    targetfreq = i * fundamental_freq

    #Skip harmonics above the Nyquist frequency (they would alias)
    if targetfreq >= fs / 2:
        break

    #The fundamental was only located to within one FFT bin, and that error is multiplied
    #by the harmonic order, so search a window around the target and take the peak.
    #Window half-width grows with order but stays far below the 60 Hz harmonic spacing.
    half_width = 2 * bin_res + i * bin_res
    window = np.abs(pos_frequencies - targetfreq) <= half_width
    local_idx = np.argmax(np.where(window, win_mags, -np.inf))

    #Storing magnitude at the peak found near this harmonic
    harmonics.append(band_mag(win_mags, local_idx))
    harmonicfreqs.append(pos_frequencies[local_idx])

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