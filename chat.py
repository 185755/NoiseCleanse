import numpy as np
import scipy.io.wavfile as wav
import statsmodels.api as sm
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import soundfile as sf

# Funkcja autokorelacji z statsmodels
def estimate_ar_params(signal, r):
    if len(signal) <= r or np.var(signal) == 0:
        return np.zeros(r + 1)  # Zabezpieczenie przed zbyt krótkim sygnałem lub zerową wariancją
    autocorr = sm.tsa.acf(signal, nlags=r, fft=True)
    R = toeplitz(autocorr[:r])  # Macierz Toeplitza
    r_vec = autocorr[1:r+1]
    if R.shape[0] != R.shape[1] or R.shape[0] != len(r_vec):
        return np.zeros(r)  # Zabezpieczenie przed niezgodnymi wymiarami
    a = np.linalg.solve(R, r_vec)  # Rozwiązanie układu równań
    return np.insert(a, 0, 1)  # Dodanie a0 = 1

# Detekcja zakłóceń impulsowych
def detect_impulses(signal, theta, r, std_factor=3):
    impulses = np.zeros(len(signal), dtype=bool)
    std_dev = np.std(signal[:100])  # Początkowe odchylenie

    for k in range(r, len(signal)):
        phi = signal[k-1:k-r-1:-1]  # Poprzednie próbki
        if len(phi) != r:
            continue  # Pomijanie jeśli próbki są niepełne
        pred = np.dot(theta[:, k], phi)  # Predykcja AR
        error = np.abs(signal[k] - pred)

        if error > std_factor * std_dev:
            impulses[k] = True

        std_dev = 0.99 * std_dev + 0.01 * error

    return impulses

# Interpolacja liniowa
def interpolate(signal, impulses):
    clean_signal = np.copy(signal)
    for k in range(1, len(signal) - 1):
        if impulses[k]:
            clean_signal[k] = 0.5 * (signal[k-1] + signal[k+1])
    return clean_signal

# Wczytanie pliku
filename = '06.wav'
signal, samplerate = sf.read(filename)
signal = signal[:, 0] if len(signal.shape) > 1 else signal

r = 4
theta = np.array([estimate_ar_params(signal[max(0, i-100):i], r) for i in range(r, len(signal)) if len(signal[max(0, i-100):i]) > r]).T
impulses = detect_impulses(signal, theta, r)
clean_signal = interpolate(signal, impulses)

# Zapisanie odszumionego pliku
sf.write("cleaned.wav", clean_signal, samplerate)
print("Plik odszumiony zapisany jako cleaned.wav")

# Wizualizacja
plt.figure(figsize=(14, 6))
plt.plot(signal, label="Oryginalny sygnał", alpha=0.7)
plt.plot(impulses * max(signal), 'r', label="Zakłócenia")
plt.plot(clean_signal, label="Po odszumieniu", alpha=0.9)
plt.legend()
plt.title("Usuwanie zakłóceń impulsowych")
plt.xlabel("Próbki")
plt.ylabel("Amplituda")
plt.show()
