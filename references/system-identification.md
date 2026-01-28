# System Identification Reference

Formal methods for building mathematical models from input-output data.

## Table of Contents

- [Model Structure Summary](#model-structure-summary)
- [Parameter Estimation](#parameter-estimation)
- [Subspace Identification (N4SID)](#subspace-identification-n4sid)
- [Frequency Domain Methods](#frequency-domain-methods)
- [NARMAX Structure Selection](#narmax-structure-selection)
- [Neural Network Approaches](#neural-network-approaches)
- [Information Criteria](#information-criteria)
- [Residual Validation](#residual-validation)
- [Practical Workflow](#practical-workflow)

## Model Structure Summary

### Linear Models

| Model | Equation | When to Use |
|-------|----------|-------------|
| ARX | A(q)y = B(q)u + e | Simple linear systems, first attempt |
| ARMAX | A(q)y = B(q)u + C(q)e | Colored noise in residuals |
| OE | y = B(q)/F(q)·u + e | Focus on dynamics, not noise |
| Box-Jenkins | y = B/F·u + C/D·e | Maximum flexibility (separate plant/noise) |
| State-Space | x' = Ax + Bu; y = Cx + Du | Multi-output, state estimation |

### Nonlinear Models

| Model | Form | When to Use |
|-------|------|-------------|
| NARMAX | y = f(y_{-1},...,u_{-1},...) + e | General nonlinear |
| Polynomial NARMAX | Polynomial expansion of NARMAX | Structured nonlinearity |
| Neural Network | y = NN(y_{-1},...,u_{-1},...) | Complex unknown nonlinearity |
| EFSM | (S, s₀, I, O, V, T) | Discrete modes/protocols |
| Koopman/DMD | Linear in lifted space | Nonlinear with linear structure |

## Parameter Estimation

### Ordinary Least Squares
```python
import numpy as np

def estimate_ols(Phi, y):
    """
    Phi: regressor matrix (N x p)
    y: output vector (N,)
    Returns: parameter estimates, residuals
    """
    # QR decomposition for numerical stability
    Q, R = np.linalg.qr(Phi)
    theta = np.linalg.solve(R, Q.T @ y)
    residuals = y - Phi @ theta
    return theta, residuals
```

### ARX Estimation
```python
def build_arx_regressors(y, u, na, nb, nk=1):
    """
    Build regressor matrix for ARX model.
    na: number of y lags
    nb: number of u lags  
    nk: input delay
    """
    N = len(y)
    n_start = max(na, nb + nk - 1)
    
    Phi = np.zeros((N - n_start, na + nb))
    for i in range(n_start, N):
        # y regressors: -y(t-1), -y(t-2), ...
        Phi[i - n_start, :na] = -y[i-1:i-na-1:-1] if na > 0 else []
        # u regressors: u(t-nk), u(t-nk-1), ...
        Phi[i - n_start, na:] = u[i-nk:i-nk-nb:-1] if nb > 0 else []
    
    return Phi, y[n_start:]
```

## Subspace Identification (N4SID)

Direct state-space estimation without polynomial structure—essential for MIMO systems.

### Algorithm
```python
def n4sid(y, u, i, n):
    """
    N4SID subspace identification.
    y: output data (N x p)
    u: input data (N x m)
    i: block rows in Hankel matrix
    n: desired state order
    """
    N, p = y.shape
    _, m = u.shape
    
    # 1. Build block Hankel matrices
    Y = block_hankel(y, i)  # (i*p) x (N-2i+1)
    U = block_hankel(u, i)  # (i*m) x (N-2i+1)
    
    # 2. Oblique projection: future outputs onto past I/O
    Yf = Y[i*p:, :]   # Future outputs
    Yp = Y[:i*p, :]   # Past outputs
    Up = U[:i*m, :]   # Past inputs
    Uf = U[i*m:, :]   # Future inputs
    
    Wp = np.vstack([Up, Yp])  # Past data
    Ob = oblique_projection(Yf, Uf, Wp)
    
    # 3. SVD to extract observability matrix
    U_svd, S, Vh = np.linalg.svd(Ob)
    
    # 4. Truncate to order n
    Gamma = U_svd[:, :n] @ np.diag(np.sqrt(S[:n]))
    
    # 5. Recover A, C from shift structure
    C = Gamma[:p, :]
    A = np.linalg.lstsq(Gamma[:-p, :], Gamma[p:, :], rcond=None)[0]
    
    # 6. Recover B, D via least squares
    # ... (system of linear equations)
    
    return A, B, C, D
```

### Variants
| Method | Projection | Best For |
|--------|------------|----------|
| N4SID | Oblique | General purpose |
| MOESP | Orthogonal | Better numerical conditioning |
| CVA | Statistically weighted | Finite samples, optimal asymptotically |

## Frequency Domain Methods

### Empirical Transfer Function Estimate (ETFE)
```python
def etfe(u, y, fs):
    """Direct frequency response estimate."""
    U = np.fft.fft(u)
    Y = np.fft.fft(y)
    G = Y / (U + 1e-10)  # Avoid division by zero
    freqs = np.fft.fftfreq(len(u), 1/fs)
    return freqs, G
```
**Warning**: ETFE variance does NOT decrease with sample size. Use smoothed estimates.

### FRF Estimators for Noisy Data
| Estimator | Formula | Assumption | Best At |
|-----------|---------|------------|---------|
| H1 | Σ(X*Y)/Σ\|X\|² | Noise uncorrelated with input | Antiresonances |
| H2 | Σ\|Y\|²/Σ(Y*X) | Noise uncorrelated with output | Resonances |
| Hv | √(H1·H2) | Geometric mean | Most robust overall |

```python
def frf_h1(u, y, nperseg=256):
    """H1 estimator via Welch's method."""
    from scipy.signal import csd, welch
    f, Puu = welch(u, nperseg=nperseg)
    f, Puy = csd(u, y, nperseg=nperseg)
    H1 = Puy / Puu
    return f, H1
```

### Coherence Function
```python
def coherence(u, y, nperseg=256):
    """
    γ²(f) = |Gxy|² / (Gxx·Gyy)
    
    γ² < 1 indicates: noise, nonlinearity, or unmeasured inputs
    Use as DATA QUALITY metric before identification.
    """
    from scipy.signal import coherence as coh
    f, gamma2 = coh(u, y, nperseg=nperseg)
    return f, gamma2
```

## NARMAX Structure Selection

### Error Reduction Ratio (ERR)
```python
def frols(y, candidates, threshold=0.99):
    """
    Forward Regression Orthogonal Least Squares.
    Selects minimal term set explaining variance.
    """
    selected = []
    total_err = 0
    residual = y.copy()
    
    while total_err < threshold and candidates:
        best_err, best_term = 0, None
        
        for term in candidates:
            # Orthogonalize against selected terms
            w = orthogonalize(term, selected)
            
            # Compute ERR
            g = np.dot(w, y) / np.dot(w, w)
            err = (g**2 * np.dot(w, w)) / np.dot(y, y)
            
            if err > best_err:
                best_err, best_term = err, term
        
        if best_term:
            selected.append(best_term)
            candidates.remove(best_term)
            total_err += best_err
    
    return selected, total_err
```

## Neural Network Approaches

### LSTM for Dynamic Systems
```python
import torch.nn as nn

class LSTMIdentifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

### Koopman/DMD for Nonlinear Systems
```python
def dmd(X, r=None):
    """
    Dynamic Mode Decomposition.
    X: data matrix where columns are snapshots
    Returns: eigenvalues, modes
    """
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    if r:
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
    
    Atilde = U.T @ X2 @ Vh.T @ np.diag(1/S)
    eigenvalues, W = np.linalg.eig(Atilde)
    modes = X2 @ Vh.T @ np.diag(1/S) @ W
    
    return eigenvalues, modes
```

### SINDy (Sparse Identification of Nonlinear Dynamics)
```python
def sindy(X, Xdot, library_fn, threshold=0.1):
    """
    Discover governing equations: Ẋ = Θ(X)·Ξ
    """
    Theta = library_fn(X)  # Candidate terms
    
    # Sparse regression (sequential thresholding)
    Xi = np.linalg.lstsq(Theta, Xdot, rcond=None)[0]
    
    for _ in range(10):
        small_idx = np.abs(Xi) < threshold
        Xi[small_idx] = 0
        for i in range(Xi.shape[1]):
            big_idx = ~small_idx[:, i]
            Xi[big_idx, i] = np.linalg.lstsq(
                Theta[:, big_idx], Xdot[:, i], rcond=None)[0]
    
    return Xi
```

## Information Criteria

### Model Selection
```python
def compute_criteria(n, k, rss):
    """
    n: number of samples
    k: number of parameters
    rss: residual sum of squares
    """
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    fpe = (rss / n) * (n + k) / (n - k)
    
    return {'AIC': aic, 'BIC': bic, 'FPE': fpe}
```

### Selection Rule
1. Compute AIC/BIC for candidate models
2. Select model with lowest criterion
3. If AIC and BIC disagree, prefer BIC (more parsimonious)
4. Check that selected model passes residual tests

## Residual Validation

### Whiteness Test
```python
def whiteness_test(residuals, max_lag=20, alpha=0.05):
    """Test if residuals are white noise."""
    N = len(residuals)
    acf = np.correlate(residuals, residuals, mode='full')
    acf = acf[N-1:] / acf[N-1]
    bound = 1.96 / np.sqrt(N)
    is_white = np.all(np.abs(acf[1:max_lag+1]) < bound)
    return is_white, acf[:max_lag+1], bound
```

### Ljung-Box Test
```python
def ljung_box(residuals, lags=20):
    """
    Q = n(n+2) Σ(ρ̂k² / (n-k)) ~ χ²(h) under null
    """
    from scipy.stats import chi2
    n = len(residuals)
    acf = np.correlate(residuals, residuals, mode='full')[n-1:]
    acf = acf / acf[0]
    
    Q = n * (n + 2) * np.sum(acf[1:lags+1]**2 / (n - np.arange(1, lags+1)))
    p_value = 1 - chi2.cdf(Q, lags)
    
    return Q, p_value

def cross_validate(y, u, model_fn, k_folds=5):
    """Forward chaining cross-validation for time series."""
    N = len(y)
    fold_size = N // (k_folds + 1)
    r2_scores = []
    
    for i in range(k_folds):
        train_end = (i + 2) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        
        model = model_fn(y[:train_end], u[:train_end])
        y_pred = model.predict(u[test_start:test_end])
        
        ss_res = np.sum((y[test_start:test_end] - y_pred) ** 2)
        ss_tot = np.sum((y[test_start:test_end] - np.mean(y[test_start:test_end])) ** 2)
        r2_scores.append(1 - ss_res / ss_tot)
    
    return np.mean(r2_scores), np.std(r2_scores)
```

## Practical Workflow

1. **Data quality check**: Compute coherence γ²(f) — if < 0.8 at frequencies of interest, improve data
2. **Start simple**: ARX with low orders (na=nb=2)
3. **Check residuals**: Ljung-Box test — if fail, increase complexity
4. **Compare models**: Use AIC/BIC, prefer simpler
5. **Validate**: Forward-chain CV, R² > 0.8
6. **Document uncertainty**: Bootstrap or Bayesian credible intervals
