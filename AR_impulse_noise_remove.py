from scipy.io import wavfile
import statsmodels.api as sm
from scipy.linalg import toeplitz
import numpy as np
DEBUG = True

def youl_walker(signal, r):
    nOfSamples = len(signal)
    #autocorlelationSignal = np.correlate(signal, signal)[nOfSamples-1:]
    autocorlelationSignal = sm.tsa.acf(signal, nlags = r, fft = True)
    R = toeplitz(autocorlelationSignal[:r])
    r_vec = autocorlelationSignal[1: r+1]
    a = np.linalg.solve(R, r_vec)
    a = np.insert(a, 0, 1)
    if(DEBUG):
        print(autocorlelationSignal)
        print(R)
        print(r_vec)
        print(a)
    return a
    
def ewls(signal, r, lambda_ = 0.99):
    N = len(signal)
    P = np.eye(r) * 1000
    theta = np.zeros((r, 1))
    theta_s = np.zeros((r, N))
    a = np.zeros(r)
    fi = np.zeros((r, 1))
    for t in range(r, N):
        for i in range(0, r):
            fi[i, 0] = signal[t-i]
        e = signal[t] - fi.T @ theta
        print(e)
        k = P @ fi / (lambda_ + fi.T @ P @ fi)
        theta = theta + k @ e
        P = (P - k @ fi.T @ P) / lambda_
        print(theta)
        theta_s[:, t] = theta[:,0]


    return theta

def main():
    samplerate, data = wavfile.read('06.wav')
    print(samplerate)
    youl_walker(data, 4)
    print(ewls(data, 4))
    

if __name__ == "__main__":
    main()