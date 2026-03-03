import numpy as np

def compute_acf(signal: np.ndarray, max_lag: int = None) -> np.ndarray:
    """Автокорреляционная функция"""
    n = len(signal)
    max_lag = max_lag or n // 2
    signal_centered = signal - np.mean(signal)
    acf = np.correlate(signal_centered, signal_centered, mode='full')
    acf = acf[n-1:n-1+max_lag] / acf[n-1]
    return acf

def compute_psd(signal: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Спектральная плотность мощности"""
    n = len(signal)
    fft = np.fft.rfft(signal)
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, 1/fs)
    return freqs, psd
