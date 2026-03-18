from enum import IntEnum

import numpy as np


class SignalType(IntEnum):
    HARMONIC = 0
    GAUSSIAN = 1
    SAWTOOTH = 2


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
    freqs: list[float],
    amps: list[float],
) -> np.ndarray:
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
    rng = np.random.default_rng(seed)
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
    seed: int = 42,
) -> np.ndarray:
    """
    Равномерный шум.
    Значения распределены равномерно в диапазоне [-amplitude, +amplitude].
    """
    rng = np.random.default_rng(seed)
    return amplitude * (2.0 * rng.random(n) - 1.0)


def generate_white_noise(
    n: int,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Белый шум (гауссовский).
    Спектральная плотность мощности постоянна на всех частотах.
    PSD(f) = const
    """
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(n)


def generate_pink_noise(
    n: int,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Розовый шум (1/f шум).
    Спектральная плотность мощности обратно пропорциональна частоте:
    PSD(f) ∝ 1/f

    Метод: генерируем белый шум в частотной области,
    умножаем амплитуду на 1/sqrt(f), затем ОБПФ.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    # Избегаем деления на ноль для нулевой частоты
    freqs[0] = 1.0
    spectrum *= 1.0 / np.sqrt(freqs)
    spectrum[0] = 0.0  # убираем DC-компоненту

    result = np.fft.irfft(spectrum, n=n)
    # Нормализация
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_brown_noise(
    n: int,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Красный (броуновский) шум (1/f² шум).
    Спектральная плотность мощности:
    PSD(f) ∝ 1/f²

    Метод: кумулятивная сумма белого шума (случайное блуждание).
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    result = np.cumsum(white)

    # Удаляем линейный тренд
    result -= np.linspace(result[0], result[-1], n)

    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_blue_noise(
    n: int,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Синий шум (f шум).
    Спектральная плотность мощности пропорциональна частоте:
    PSD(f) ∝ f

    Метод: генерируем белый шум в частотной области,
    умножаем амплитуду на sqrt(f), затем ОБПФ.
    """
    rng = np.random.default_rng(seed)
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
    seed: int = 42,
) -> np.ndarray:
    """
    Фиолетовый шум (f² шум).
    Спектральная плотность мощности:
    PSD(f) ∝ f²

    Метод: дифференцирование белого шума (конечные разности).
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    result = np.diff(white, prepend=0.0)

    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * amplitude
    return result


def generate_colored_noise(
    noise_type: NoiseType,
    n: int,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Диспетчер генерации цветного шума."""
    generators = {
        NoiseType.UNIFORM: generate_uniform_noise,
        NoiseType.WHITE: generate_white_noise,
        NoiseType.PINK: generate_pink_noise,
        NoiseType.BROWN: generate_brown_noise,
        NoiseType.BLUE: generate_blue_noise,
        NoiseType.VIOLET: generate_violet_noise,
    }
    gen = generators[noise_type]
    return gen(n, amplitude, seed)


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
