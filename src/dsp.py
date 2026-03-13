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
    """
    Сумма гармоник.
    Формула: s(t) = Σ [ A_i * cos(2π * f_i * t + φ_i) ]
    где:
        t = n * dt (вектор времени)
        A_i - амплитуда i-й гармоники
        f_i - частота i-й гармоники
        φ_i - фаза i-й гармоники
    """
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
    Гауссов импульс (колоколообразный сигнал).
    Формула: g(t) = A * exp( -(t - t0)² / (2 * σ²) )
    Связь параметров:
        t0 = max(t) / 2 (центр импульса)
        σ = 1 / (2π * f) (ширина импульса через эквивалентную частоту)
    """
    t = np.arange(n) * dt

    if not freqs:
        return np.zeros(n)

    t0 = t[-1] / 2

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
    Пилообразный сигнал (линейно нарастающий).
    Формула одного периода: s(t) = A * (2 * (t / T mod 1) - 1)
    где:
        T = 1 / f (период сигнала)
        (t / T mod 1) - нормализованная фаза [0, 1)
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
    """
    Случайная помеха на базе суммы гармоник.
    Формула: noise(t) = Σ [ A_rand(t) * cos(2π * f_noise * t) ]
    где:
        A_rand(t) - случайная амплитуда в диапазоне [amp_min, amp_max]
        генерируемая для каждого момента времени.
    """
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
    Дискретное Преобразование Фурье (ДПФ).
    Формулы:
        1. Косинусная часть: Re[k] = (2/N) * Σ [ x[n] * cos(2π * k * n / N) ]
        2. Синусная часть:   Im[k] = (2/N) * Σ [ x[n] * sin(2π * k * n / N) ]
        3. Амплитуда:        Mag[k] = sqrt( Re[k]² + Im[k]² )
        4. Фаза:                      Phi[k] = -arctan( Im[k] / Re[k] )
    """
    n = len(signal)
    indices = np.arange(n)

    angles = 2 * np.pi * np.outer(indices, indices) / n

    rl = signal @ np.cos(angles).T
    im = signal @ np.sin(angles).T

    scale = np.full(n, 2.0 / n)
    scale[0] = 1.0 / n
    rl *= scale
    im *= scale

    rl = np.round(rl, 6)
    im = np.round(im, 6)

    magnitude = np.sqrt(rl**2 + im**2)

    phase_rad = np.where(rl != 0, -np.arctan(im / rl), 0.0)
    phase_deg = np.degrees(phase_rad)

    return rl, im, magnitude, phase_deg
