# src/dsp.py
from enum import IntEnum

import numpy as np


class SignalType(IntEnum):
    HARMONIC = 0
    GAUSSIAN = 1
    SAWTOOTH = 2
    IMPULSE = 3
    EXPONENTIAL_IMPULSE = 4


class NoiseType(IntEnum):
    UNIFORM = 0
    WHITE = 1
    PINK = 2
    BROWN = 3
    BLUE = 4
    VIOLET = 5


def generate_harmonics(
    n: int,
    dt: float,
    freqs: list[float],
    amps: list[float],
    phases: list[float],
) -> np.ndarray:
    t = np.arange(n) * dt
    if not freqs:
        return np.zeros(n)
    angles = 2 * np.pi * np.outer(t, freqs) + np.array(phases)
    return np.sum(np.array(amps) * np.cos(angles), axis=1)


def generate_gaussian(
    n: int,
    dt: float,
    centers: list[float],
    amps: list[float],
    sigmas: list[float],
) -> np.ndarray:
    """
    Гауссов импульс.
    centers — центр импульса (в секундах),
    sigmas — ширина (сигма) импульса (в секундах).
    """
    t = np.arange(n) * dt
    if not centers:
        return np.zeros(n)
    result = np.zeros(n)
    for c, a, s in zip(centers, amps, sigmas):
        if s <= 0:
            s = 0.001
        result += a * np.exp(-((t - c) ** 2) / (2 * s**2))
    return result


def generate_sawtooth(
    n: int,
    dt: float,
    freqs: list[float],
    amps: list[float],
) -> np.ndarray:
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


def generate_impulse(
    n: int,
    dt: float,
    widths: list[float],
    amps: list[float],
    periods: list[float],  # Переименовали distances в periods
) -> np.ndarray:
    """
    Периодическая последовательность прямоугольных импульсов.
    widths - длительность импульса (с)
    periods - период повторения (с). Если <= 0, то одиночный импульс.
    """
    t = np.arange(n) * dt
    result = np.zeros(n)
    for w, a, period in zip(widths, amps, periods):
        if period <= 0:
            # Одиночный импульс от t=0 до t=w
            mask = (t >= 0) & (t <= w)
            result[mask] += a
        else:
            # Периодические импульсы (остаток от деления времени на период)
            # Если время внутри периода меньше длительности импульса - сигнал включен
            mask = (t % period) <= w
            result[mask] += a
    return result


def generate_exponential_impulse(
    n: int,
    dt: float,
    alphas: list[float],
    amps: list[float],
    delays: list[float],  # Добавили задержку
) -> np.ndarray:
    """
    Экспоненциальный импульс: A * exp(alpha * (t - delay)) для t >= delay.
    alpha < 0 - затухание, alpha > 0 - нарастание.
    """
    t = np.arange(n) * dt
    result = np.zeros(n)
    for alpha, a, delay in zip(alphas, amps, delays):
        # Включаем сигнал только после времени delay
        mask = t >= delay
        # Считаем экспоненту только для нужных точек (чтобы не было переполнения)
        result[mask] += a * np.exp(alpha * (t[mask] - delay))
    return result


def generate_noise(
    n: int,
    dt: float,
    freq: float,
    amp_min: int,
    amp_max: int,
    n_components: int,
) -> np.ndarray:
    rng = np.random.default_rng()
    t = np.arange(n) * dt
    angle = 2 * np.pi * freq * t
    result = np.zeros(n)
    for _ in range(n_components):
        random_amps = amp_min + rng.integers(max(amp_max - amp_min, 1), size=n)
        result += random_amps * np.cos(angle)
    return result


# ─── Цветные шумы ───


def generate_uniform_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    return amplitude * (2.0 * rng.random(n) - 1.0)


def generate_white_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    return amplitude * rng.standard_normal(n)


def generate_pink_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    freqs[0] = 1.0
    spectrum *= 1.0 / np.sqrt(freqs)
    spectrum[0] = 0.0

    result = np.fft.irfft(spectrum, n=n)
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_brown_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    white = rng.standard_normal(n)
    result = np.cumsum(white)

    result -= np.linspace(result[0], result[-1], n)

    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_blue_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    spectrum *= np.sqrt(freqs)
    spectrum[0] = 0.0

    result = np.fft.irfft(spectrum, n=n)
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_violet_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng()
    white = rng.standard_normal(n)
    result = np.diff(white, prepend=0.0)

    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_colored_noise(
    noise_type: NoiseType,
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    generators = {
        NoiseType.UNIFORM: generate_uniform_noise,
        NoiseType.WHITE: generate_white_noise,
        NoiseType.PINK: generate_pink_noise,
        NoiseType.BROWN: generate_brown_noise,
        NoiseType.BLUE: generate_blue_noise,
        NoiseType.VIOLET: generate_violet_noise,
    }
    gen = generators[noise_type]
    return gen(n, amplitude)


def compute_dft(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def compute_fft(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Быстрое преобразование Фурье.
    Возвращает: (magnitude, phase_deg, complex_spectrum)
    """
    n = len(signal)
    spectrum = np.fft.fft(signal)

    # Нормировка
    magnitude = np.abs(spectrum) * 2.0 / n
    magnitude[0] /= 2.0  # DC компонента

    # Фаза в градусах
    phase_rad = np.angle(spectrum)
    phase_deg = np.degrees(phase_rad)

    return magnitude, phase_deg, spectrum


def compute_psd(
    signal: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Спектральная плотность мощности (СПМ).
    """
    n = len(signal)
    spectrum = np.fft.fft(signal)
    psd = (np.abs(spectrum) ** 2) / (n * dt)
    return psd


def compute_acf(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Автокорреляционная функция (АКФ).
    """
    n = len(signal)
    mean = np.mean(signal)
    signal_centered = signal - mean

    acf = np.correlate(signal_centered, signal_centered, mode="full")
    acf = acf[n - 1 :]  # Берём только положительные лаги
    acf = acf / acf[0] if acf[0] != 0 else acf  # Нормировка

    return acf
