# Impulse Noise Removal from Music Recordings

## Project Overview

This project aims to implement a solution for removing impulse noise from music recordings. To achieve this, an **autoregressive model (AR)** was used to represent the local dynamics of the signal. The **Exponentially Weighted Least Squares (EW-LS) algorithm** was applied to estimate the model parameters. A noise detector based on **local standard deviation** was utilized to identify noise-corrupted samples. If the prediction error of the AR model exceeded three times the local standard deviation, the sample was considered noise and was replaced using **linear interpolation** of neighboring samples.

## Implementation Details

### 1. **Preprocessing of Input Signal**
To improve robustness against silent periods (zero-value samples), **low-variance white noise** was added to the input signal. This prevents instability in the parameter identification algorithm.

### 2. **Autoregressive Model (AR)**
A **4th-order autoregressive model** was used to estimate the local dynamics of the signal, described by the following equation:

\[
AR(r): y(t) = \sum_{i=1}^{r} a_i y(t - i) + n(t)
\]

where:
- \( r \) is the order of regression,
- \( a_i \) are the autoregression coefficients,
- \( n(t) \) is additive noise (zero mean, known variance).

### 3. **Exponentially Weighted Least Squares (EW-LS) Algorithm**
The AR model parameters were dynamically updated using the **EW-LS** algorithm. The key equations are:

\[
\epsilon(t) = y(t) - \phi^T (t) \hat{\theta}(t-1)
\]

\[
k(t) = \frac{P(t-1) \phi(t)}{\lambda + \phi^T (t) P(t-1) \phi(t)}
\]

\[
\hat{\theta}(t) = \hat{\theta}(t-1) + k(t) \epsilon(t)
\]

\[
P(t) = \frac{1}{\lambda} \left[ P(t-1) - \frac{P(t-1) \phi(t) \phi^T (t) P(t-1)}{\lambda + \phi^T (t) P(t-1) \phi(t)} \right]
\]

where:
- \( \lambda \) is the forgetting factor (controls influence of past data),
- \( P \) is the covariance matrix,
- \( \phi(t) \) is the regression vector.

### 4. **Impulse Noise Detector**
Impulse noise was detected using a threshold based on the local standard deviation of prediction errors:

\[
\sigma_{\text{local}} = \text{std} (\epsilon_{t-N:t-1})
\]

If the current prediction error exceeded \( 3 \times \sigma_{\text{local}} \), the sample was marked as noisy and replaced using linear interpolation.

### 5. **Implementation in MATLAB**
The algorithm was implemented in **MATLAB** for computational efficiency. The program consists of:
- **Initialization of EW-LS and AR model parameters**  
- **Main program loop** (parameter estimation, noise detection, correction)  
- **Visualization of results** (error analysis, cleaned signal output)

## Issues and Solutions

### 1. **Zero or Infinite AR Model Parameters**
- Problem: When encountering zero-value samples, AR model coefficients became unstable or diverged to infinity.
- Solution: Added low-amplitude white noise to prevent zero-value samples.

### 2. **Sudden Spikes in AR Parameters**
- Problem: AR parameters exhibited sudden, unrealistic spikes.
- Solution: Introduced a **maximum change limit** for AR coefficients to stabilize estimation.

## Results
- The algorithm successfully removed impulse noise from music recordings.
- The AR model dynamically adapted to changes in the signal.
- Some residual noise remained due to added white noise, but impulse artifacts were eliminated.

## Conclusion
This project demonstrated the effectiveness of an **autoregressive model** combined with **EW-LS parameter estimation** and **statistical noise detection** for impulse noise removal in audio signals. MATLAB was chosen for its efficiency in numerical computation. The methodology can be extended for use in real-time audio processing.

## Authors
- **Michał Blicharz** (184430)
- **Maciek Dziewit** (185755)

## Repository Structure

