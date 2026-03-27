from enum import IntEnum

import numpy as np


class SignalType(IntEnum):
    HARMONIC = 0
    GAUSSIAN = 1
    SAWTOOTH = 2
    IMPULSE = 3
    EXPONENTIAL_IMPULSE = 4
    SPEECH = 5


class NoiseType(IntEnum):
    UNIFORM = 0
    WHITE = 1
    PINK = 2
    BROWN = 3
    BLUE = 4
    VIOLET = 5
    EXPONENTIAL = 6


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
    phases: list[float] | None = None,
) -> np.ndarray:
    t = np.arange(n) * dt
    if not freqs:
        return np.zeros(n)
    if phases is None:
        phases = [0.0] * len(freqs)
    result = np.zeros(n)
    for i, (f, a) in enumerate(zip(freqs, amps)):
        if f == 0:
            result += a
            continue
        period = 1.0 / f
        phase_shift = phases[i] if i < len(phases) else 0.0
        # Фаза в градусах -> доля периода
        phase_frac = (phase_shift % 360.0) / 360.0
        phase_time = phase_frac * period
        phase = ((t + phase_time) % period) / period
        result += a * (2.0 * phase - 1.0)
    return result


def generate_impulse(
    n: int,
    dt: float,
    widths: list[float],
    amps: list[float],
    periods: list[float],
    delays: list[float] | None = None,
) -> np.ndarray:
    t = np.arange(n) * dt
    result = np.zeros(n)
    if delays is None:
        delays = [0.0] * len(widths)
    for i, (w, a, period) in enumerate(zip(widths, amps, periods)):
        delay = delays[i] if i < len(delays) else 0.0
        t_shifted = t - delay
        if period <= 0:
            mask = (t_shifted >= 0) & (t_shifted <= w)
            result[mask] += a
        else:
            valid = t_shifted >= 0
            mask = valid & ((t_shifted % period) <= w)
            result[mask] += a
    return result


def generate_exponential_impulse(
    n: int,
    dt: float,
    alphas: list[float],
    amps: list[float],
    delays: list[float],
) -> np.ndarray:
    t = np.arange(n) * dt
    result = np.zeros(n)
    for alpha, a, delay in zip(alphas, amps, delays):
        mask = t >= delay
        result[mask] += a * np.exp(alpha * (t[mask] - delay))
    return result


def load_speech_signal(
    file_path: str,
    n: int,
    target_sr: float,
) -> np.ndarray:
    """
    Загружает WAV-файл и приводит к нужной длине и частоте дискретизации.
    Возвращает нормализованный сигнал длиной n.
    """
    import wave

    try:
        with wave.open(file_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            raw = wf.readframes(n_frames)

        # Декодируем
        if sampwidth == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        elif sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        else:
            return np.zeros(n)

        # Моно
        if n_channels > 1:
            data = data.reshape(-1, n_channels)[:, 0]

        # Нормализация
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # Ресэмплинг если частоты не совпадают
        if framerate != target_sr:
            original_duration = len(data) / framerate
            target_n = int(original_duration * target_sr)
            indices = np.linspace(0, len(data) - 1, target_n)
            data = np.interp(indices, np.arange(len(data)), data)

        # Приведение к нужной длине
        if len(data) >= n:
            result = data[:n]
        else:
            # Повторяем сигнал если он короче
            repeats = int(np.ceil(n / len(data)))
            result = np.tile(data, repeats)[:n]

        return result

    except Exception:
        return np.zeros(n)


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


def generate_exponential_noise(
    n: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Экспоненциальный шум: значения с экспоненциальным распределением,
    центрированные (среднее вычтено) и нормализованные.
    """
    rng = np.random.default_rng()
    raw = rng.exponential(scale=1.0, size=n)
    # Центрируем
    raw -= np.mean(raw)
    # Нормализуем
    max_val = np.max(np.abs(raw))
    if max_val > 0:
        raw = raw / max_val
    return amplitude * raw


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
        NoiseType.EXPONENTIAL: generate_exponential_noise,
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
    n = len(signal)
    spectrum = np.fft.fft(signal)

    magnitude = np.abs(spectrum) * 2.0 / n
    magnitude[0] /= 2.0

    phase_rad = np.angle(spectrum)
    phase_deg = np.degrees(phase_rad)

    return magnitude, phase_deg, spectrum


def compute_psd(
    signal: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Метод Шустера"""
    n = len(signal)
    spectrum = np.fft.fft(signal)
    psd = (np.abs(spectrum) ** 2) / (n * dt)
    return psd


def compute_acf(
    signal: np.ndarray,
) -> np.ndarray:
    n = len(signal)
    mean = np.mean(signal)
    signal_centered = signal - mean

    acf = np.correlate(signal_centered, signal_centered, mode="full")
    acf = acf[n - 1 :]
    acf = acf / acf[0] if acf[0] != 0 else acf

    return acf
