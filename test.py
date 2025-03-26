# importing python module named numpy
import numpy as np
from scipy.io import wavfile
 
def ewls(signal, r, lambda_ = 0.99):
    N = len(signal)
    theta = np.zeros((r,1))
    fi = np.zeros((r,1))
    print(theta.T)
    print(fi)
    thetaT = theta.T
    print(thetaT)
    x = np.matmul(fi, thetaT)
    print(x)
    x = np.matmul(thetaT, fi)
    print(x)
    x = (thetaT @ fi)
    print(x)
    for k in range(r, 100):
        for i in range(0, r):
            fi[i, 0] = signal[k-i]
        e = signal[k] - np.multiply(fi.T, theta)

samplerate, data = wavfile.read('06.wav')
print(ewls(data, 4))