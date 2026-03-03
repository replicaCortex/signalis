import numpy as np


def gaussian_pulse(
    t: np.ndarray, a: float = 1.0, t0: float = 0.5, sigma: float = 0.1
) -> np.ndarray:
    """Гауссов импульс"""
    return a * np.exp(-((t - t0) ** 2) / (2 * sigma**2))


def sawtooth(t: np.ndarray, freq: float = 5.0, a: float = 1.0) -> np.ndarray:
    """Пилообразный сигнал"""
    return a * (2 * (t * freq - np.floor(t * freq + 0.5)))


def polyharmonic(
    t: np.ndarray, freqs: list = [1, 3, 5], amps: list = [1, 0.5, 0.3]
) -> np.ndarray:
    """Полигармонический сигнал"""
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
