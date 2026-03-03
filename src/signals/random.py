import numpy as np


def white_noise(n: int, power: float = 0.1) -> np.ndarray:
    """Белый шум"""
    return np.random.normal(0, np.sqrt(power), n)


def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Добавить шум к сигналу с заданным SNR"""
    power_signal = np.mean(signal**2)
    power_noise = power_signal / (10 ** (snr_db / 10))
    return signal + white_noise(len(signal), power_noise)
