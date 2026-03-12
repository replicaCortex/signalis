from enum import IntEnum

import numpy as np


class SignalType(IntEnum):
    HARMONIC = 0
    GAUSSIAN = 1
    SAWTOOTH = 2


def generate_harmonics(
    n: int,
    dt: float,
    freqs: list[float],
    amps: list[float],
    phases: list[float],
) -> np.ndarray:
    """Сумма гармоник: Σ A·cos(2π·f·t + φ)."""
    t = np.arange(n) * dt

    if not freqs:
        return np.zeros(n)

    angles = 2 * np.pi * np.outer(t, freqs) + np.array(phases)
    return np.sum(np.array(amps) * np.cos(angles), axis=1)


def generate_gaussian(
    n: int,
    dt: float,
    freqs: list[float],
    amps: list[float],
) -> np.ndarray:
    """
    Гауссов импульс.

    Для каждой компоненты генерируется импульс:
        g(t) = A · exp(-(t - t0)² / (2σ²))

    Частота определяет ширину импульса: σ = 1 / (2π·f).
    Центр импульса — середина временного интервала.
    """
    t = np.arange(n) * dt

    if not freqs:
        return np.zeros(n)

    t0 = t[-1] / 2  # Центр — середина записи

    result = np.zeros(n)
    for f, a in zip(freqs, amps):
        if f == 0:
            result += a
            continue
        sigma = 1.0 / (2 * np.pi * f)
        result += a * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    return result


def generate_sawtooth(
    n: int,
    dt: float,
    freqs: list[float],
    amps: list[float],
) -> np.ndarray:
    """
    Пилообразный сигнал.

    Линейно нарастает от -A до +A за один период.
    """
    t = np.arange(n) * dt

    if not freqs:
        return np.zeros(n)

    result = np.zeros(n)
    for f, a in zip(freqs, amps):
        if f == 0:
            result += a
            continue
        period = 1.0 / f
        phase = (t % period) / period
        result += a * (2.0 * phase - 1.0)

    return result


def generate_noise(
    n: int,
    dt: float,
    freq: float,
    amp_min: int,
    amp_max: int,
    n_components: int,
    seed: int = 42,
) -> np.ndarray:
    """Случайная помеха: сумма cos с рандомными амплитудами."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    angle = 2 * np.pi * freq * t

    result = np.zeros(n)
    for _ in range(n_components):
        random_amps = amp_min + rng.integers(max(amp_max - amp_min, 1), size=n)
        result += random_amps * np.cos(angle)
    return result


def compute_dft(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ручное ДПФ.

    Возвращает: (real, imag, magnitude, phase_degrees).
    """
    n = len(signal)
    indices = np.arange(n)

    angles = 2 * np.pi * np.outer(indices, indices) / n

    real = signal @ np.cos(angles).T
    imag = signal @ np.sin(angles).T

    scale = np.full(n, 2.0 / n)
    scale[0] = 1.0 / n
    real *= scale
    imag *= scale

    real = np.round(real, 6)
    imag = np.round(imag, 6)

    mag = np.sqrt(real**2 + imag**2)

    phase_rad = np.where(real != 0, -np.arctan(imag / real), 0.0)
    phase_deg = np.degrees(phase_rad)

    return real, imag, mag, phase_deg
