import pandas as pd

"df = pd.read_csv('egauge_9-Mains_Power.csv')"

"print(df.to_string())"




import sys
import scipy as sp
import numpy as np
import matplotlib as mpl
from scipy.fftpack import fft, fftfreq

# We're going to get the windows from the "signal" library of SciPy
from scipy import signal
import random

import matplotlib.pyplot as plt
#print(plt.style.available)
plt.style.use('classic')

# Frequencies & Length
f1 = 5
f2 = 4
duration = 2.0

# Sampling Rate
Fs = 20 * max(f1,f2)
Ts = 1/Fs

# Time vector and Sinusoids
t = np.arange(0, duration, Ts)
s1 = np.sin(f1 * 2 * sp.pi * t)
s2 = np.sin(f2 * 2 * sp.pi * t)

# Add them up
s3 = s1 + s2

plt.plot(t,s3)
plt.show()

various_windows = ['boxcar', 'triang', 'hann', 'hamming', 'blackman', 'bartlett', 'nuttall']

# Pick a random one from the list
window_type = random.choice(various_windows)
window_function = signal.windows.get_window(window_type, len(s3), True)

# Apply the window to the function
s3w = s3 * window_function

def get_Spectrum(signal, NFFT, Ts):
    CFFT = fft(signal,NFFT) / len(signal)
    FF = fftfreq(NFFT,Ts)
    SS = 20.0 * np.log10(np.abs(CFFT))
    return FF, SS

y = s3
yw = s3w

framepad = 4
NFFT = 2 ** int(np.ceil(np.log2(np.abs(len(y)*framepad))))

# We already know Ts from when the signal was created

F, Y = get_Spectrum(y, NFFT, Ts)
FW, YW = get_Spectrum(yw, NFFT, Ts)


fig = plt.figure(figsize=(12,10))
plt.suptitle('Windows', size='large')

# Top plot - time domain
ax1 = fig.add_subplot(2,1,1)
ax1.plot(y, color='blue', label='Original', linewidth=1)
ax1.plot(yw, color='red', label='Windowed', linewidth=1)
ax1.plot(window_function, color='green', label='Window ('+window_type+')', linewidth=1, linestyle='--')

ax1.set_title('Signal', fontsize='small')
ax1.set_xlabel('Time (sec)', fontsize='small')
ax1.set_ylabel('Amplitude', fontsize='small')
ax1.set_ylim([1.1*min(y), 1.1*max(y)])
#ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax1.legend()
ax1.grid()

# Bottom plot - frequency domain
ax2 = fig.add_subplot(2,1,2)
ax2.plot(F[0:NFFT//2], Y[0:NFFT//2], color='blue', label='Original', linewidth=1)
ax2.plot(FW[0:NFFT//2], YW[0:NFFT//2], color='red', label='Windowed ('+window_type+')', linewidth=1)

ax2.set_title('Spectrum', fontsize='small')
ax2.set_xlabel('Frequency (Hz)', fontsize='small')
ax2.set_ylabel('Magnitude (dB)', fontsize='small')
ax2.set_ylim([-100,10])
#ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax2.legend()
ax2.grid()

# Show it
plt.tight_layout()
plt.show()


ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t, s, color='blue', linewidth=1)
ax1.set_title('Signal', fontsize='small')
ax1.set_xlabel('Time (sec)', fontsize='small')
ax1.set_ylabel('Amplitude', fontsize='small')
ax1.set_ylim([1.1 * min(s), 1.1 * max(s)])
#ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax1.legend()
ax1.grid()

NFFT = 2 ** int(np.ceil(np.log2(len(s) * framepad)))

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(F[0:NFFT//2], Y[0:NFFT//2], color='blue', linewidth=1)
ax2.axvline(f1, color='red',   linestyle='--', linewidth=1, label=f'f1 = {f1} Hz')
ax2.axvline(f2, color='green', linestyle='--', linewidth=1, label=f'f2 = {f2} Hz')
ax2.set_title('Spectrum', fontsize='small')
ax2.set_xlabel('Frequency (Hz)', fontsize='small')
ax2.set_ylabel('Magnitude (dB)', fontsize='small')
ax2.set_ylim([-100, 10])
#ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax2.legend()
ax2.grid()



builtin_windows = {
    'Rectangular (Boxcar)' : signal.windows.boxcar(len(s)),
    'Hanning'              : signal.windows.hann(len(s)),
    'Hamming'              : signal.windows.hamming(len(s)),
    'Bartlett'             : signal.windows.bartlett(len(s)),
    'Nuttall'              : signal.windows.nuttall(len(s)),
}

various_windows = ['boxcar', 'triang', 'hann', 'hamming', 'blackman', 'bartlett', 'nuttall']

# Pick a random one from the list
window_type = random.choice(various_windows)
window_function = signal.windows.get_window(window_type, len(s3), True)

# Apply the window to the function
s3w = s3 * window_function

    # Time domain
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t, s3w)
    ax1.set_title("Windowed Signal")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")
    ax1.grid()

    # Frequency domain
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(F[0:NFFT//2], Y[0:NFFT//2])
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim([-100,10])
    ax2.grid()

    plt.tight_layout()
    plt.show()



for window_label, window_type in windows.items():
    window_function = signal.windows.get_window(window_type, len(s3), True)
    s3w = s3 * window_function
    NFFT = 2 ** int(np.ceil(np.log2(len(s3) * framepad)))
    F, Y = get_Spectrum(s3w, NFFT, Ts)
    fig = plt.figure(figsize=(12,10))
    plt.suptitle(window_label + " Window")
    # Time domain
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t, s3w)
    ax1.set_title("Windowed Signal")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")
    ax1.grid()
    # Frequency domain
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(F[0:NFFT//2], Y[0:NFFT//2])
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim([-100,10])
    ax2.grid()
    plt.tight_layout()
    plt.show()


    # Frequencies & Length
f1 = 5
f2 = 4
duration = 2.0

# Sampling Rate
Fs = 20 * max(f1,f2)
Ts = 1/Fs

framepad = 4

# Time vector and Sinusoids
t = np.arange(0, duration, Ts)
s1 = np.sin(f1 * 2 * np.pi * t)
s2 = np.sin(f2 * 2 * np.pi * t)

s3 = s1 + s2

#zeropadding by 4
NFFT = 2 ** int(np.ceil(np.log2(len(s3) * framepad)))
F, Y = get_Spectrum(y, NFFT, Ts)
fig = plt.figure(figsize=(12,10))

#time domain

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t, s3, color='blue', linewidth=1)
ax1.set_title('Signal', fontsize='small')
ax1.set_xlabel('Time (sec)', fontsize='small')
ax1.set_ylabel('Amplitude', fontsize='small')
ax1.set_ylim([1.1 * min(s3), 1.1 * max(s3)])
#ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax1.legend()
ax1.grid()

#Frequency Spectrum
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(F[0:NFFT//2], Y[0:NFFT//2], color='blue', linewidth=1)
ax2.axvline(f1, color='red',   linestyle='--', linewidth=1, label=f'f1 = {f1} Hz')
ax2.axvline(f2, color='green', linestyle='--', linewidth=1, label=f'f2 = {f2} Hz')
ax2.set_title('Spectrum', fontsize='small')
ax2.set_xlabel('Frequency (Hz)', fontsize='small')
ax2.set_ylabel('Magnitude (dB)', fontsize='small')


plt.plot(t,s3)
plt.show()

windows = {
    "Rectangular": "boxcar",
    "Hanning": "hann",
    "Hamming": "hamming",
    "Bartlett": "bartlett",
    "Nuttall": "nuttall",
    #Extra (other function)
    "Flat-top": "flattop"
    
}

for winLabel, winType in windows.items():
    winfunction = signal.windows.get_window(winType, len(s3), True)
    s3w = s3 * winfunction
    NFFT = 2 ** int(np.ceil(np.log2(len(s3) * framepad)))
    F, Y = get_Spectrum(s3w, NFFT, Ts)
    fig = plt.figure(figsize=(12,10))
    plt.suptitle(winLabel + " Window")
    # Time domain
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t, s3w)
    ax1.set_title("Windowed Signal")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")
    ax1.grid()
    # Frequency domain
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(F[0:NFFT//2], Y[0:NFFT//2])
    ax2.set_title("Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim([-100,10])
    ax2.grid()
    plt.tight_layout()
    plt.show()


# Energy before windowing
E_original = np.sum(s3**2) / len(s3)
print(f"{'Window':<20} {'E_windowed':>12} {'E_corrected':>13} {'Ratio':>8}")
print("-" * 55)

for winLabel, winType in windows.items():
    winfunction = signal.windows.get_window(winType, len(s3), True)
    s3w  = s3 * winfunction

    # Correction factor
    cf   = np.sqrt(len(winfunction) / np.sum(winfunction**2))
    s3wc = s3w * cf

    # Energy
    E_w  = np.sum(s3w**2)  / len(s3w)
    E_wc = np.sum(s3wc**2) / len(s3wc)
    print(f"{winLabel:<20} {E_w:>12.6f} {E_wc:>13.6f} {E_w/E_original:>8.4f}")

print(f"\n{'Original':<20} {E_original:>12.6f}")




for winLabel, winType in windows.items():
    winfunction = signal.windows.get_window(winType, len(s3), True)
    s3w  = s3 * winfunction

    # Correction factor
    cf   = np.sqrt(len(winfunction) / np.sum(winfunction**2))
    s3wc = s3w * cf

    # Energy
    E_w  = np.sum(s3w**2)  / len(s3w)
    E_wc = np.sum(s3wc**2) / len(s3wc)
    print(winLabel, "\t", round(E_w, 6), "\t", round(E_wc, 6), "\t", round(E_w/E_original, 4))

print("Original", "\t", round(E_original, 6))
#---------------------------------------------------
import sys
import scipy as sp
import numpy as np
import matplotlib as mpl
from scipy.fftpack import fft, fftfreq
from scipy import signal

# We're going to need this for the noise
import random

import matplotlib.pyplot as plt
#print(plt.style.available)
plt.style.use('classic')

# Length 
duration = 2.0

# Frequencies
f1 = 5
f2 = 4

# Sampling Rate
Fs = 20 * max(f1,f2)

# Amplitudes
s1_ampl = 1.0
s2_ampl = 1.5

n1_ampl = 0.1
n2_ampl = 0.5

# Time vector and Sinusoids
t = np.arange(0, duration, 1/Fs)
s1 = s1_ampl * np.sin(f1 * 2 * sp.pi * t)
s2 = s2_ampl * np.sin(f2 * 2 * sp.pi * t)

# Quick plot 
plt.figure(figsize=(5,4))

plt.plot(t, s1, color='blue', label='s1')
plt.plot(t, s2, color='green', label='s2')

mm = 1.1* max(np.maximum(s1,s2))
plt.ylim(-mm,mm) 
plt.xlim(0,duration)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

np.random.seed(12345)

n1 = n1_ampl * np.random.normal(0,1,len(t))
n2 = n2_ampl * np.random.normal(0,1,len(t))

# Quick plot 
plt.figure(figsize=(5,4))

plt.plot(t, n1, color='blue', label='n1')
plt.plot(t, n2, color='green', label='n2')

mm = 1.1* max(np.maximum(n1,n2))
plt.ylim(-mm,mm) 
plt.xlim(0,duration)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

s1n1 = s1+n1
s2n2 = s2+n2

# Quick plot 
plt.figure(figsize=(5,4))

plt.plot(t, s1n1, color='blue', label='s1+n1')
plt.plot(t, s2n2, color='green', label='s2+n2')

mm = 1.1* max(np.maximum(s1n1,s2n2))
plt.ylim(-mm,mm) 
plt.xlim(0,duration)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Inner product (numpy function assumes real values)
svar = np.inner(s1,s1)
nvar = np.inner(n1,n1)
print('Inner Product:')
print('\tSNR (s1,n1) = {:3.2f} dB'.format(10*np.log10(svar/nvar)))

# Sum of squares
svar = np.sum(np.square(s2))
nvar = np.sum(np.square(n2))
print('Sum of Squares:')
print('\tSNR (s2,n2) = {:3.2f} dB'.format(10*np.log10(svar/nvar)))

# Variance
svar = np.var(s1)
nvar = np.var(n1)
print('Variance:')
print('\tSNR (s1,n1) = {:3.2f} dB'.format(10*np.log10(svar/nvar)))

svar = np.var(s2)
nvar = np.var(n2)
print('\tSNR (s2,n2) = {:3.2f} dB'.format(10*np.log10(svar/nvar)))


def noiseAdd(s, snr_db, seed=None):
    spower = np.var(signal)
    noisepower = s / (10 ** (snr_db / 10))
    
    if seed is not None:
        np.random.seed(seed)
    noise = noise_std * np.random.normal(0, 1, len(signal))
    
    return signal + noise, noise, noise_std

def add_noise(signal, snr_db, seed=None):
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    
    if seed is not None:
        np.random.seed(seed)
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
    
    return signal + noise, noise


# Parameters
Fs = 2000          # sampling rate (samples/sec)
duration = 2.0    
t = np.linspace(0, duration, int(Fs * duration), endpoint=False)

# Two sinusoids at 200 Hz and 400 Hz
f1, f2 = 200, 400
sin1 = np.sin(2 * np.pi * f1 * t)
sin2 = np.sin(2 * np.pi * f2 * t)


sum = sin1 + sin2
# Write WAV file
wav.write('signals_200_400Hz.wav', Fs)







# Normalize and convert to int16 (same format as sin400.wav)
mixed_int16 = (signals / np.max(np.abs(signals)) * 32767 * 0.9).astype(np.int16)


def add_noise_db(signal, db):
    signal_power = np.mean(signal**2)
    noise_power = signal_power * (10 ** (db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

noisy = add_noise_db(mixed, -3)
wav.write('signals_noise_-3dB.wav', Fs, noisy)
Fs, noisy = wav.read('signals_noise_-3dB.wav')
noisy = add_noise_db(signals, -3)

