# src/app.py
from dataclasses import dataclass, field

import numpy as np

from dsp import (
    NoiseType,
    SignalType,
    compute_acf,
    compute_dft,
    compute_fft,
    compute_psd,
    generate_colored_noise,
    generate_exponential_impulse,
    generate_gaussian,
    generate_harmonics,
    generate_impulse,
    generate_noise,
    generate_sawtooth,
    load_speech_signal,
)
from filters import (
    FilterType,
    apply_filter,
    compute_frequency_response,
)


@dataclass
class SignalParams:
    duration: float = 1.0
    sample_rate: float = 100.0

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def n_samples(self) -> int:
        return int(round(self.duration / self.dt))

    @property
    def df(self) -> float:
        return 1.0 / self.duration


@dataclass
class NoiseParams:
    enabled: bool = False
    amp_min: int = 0
    amp_max: int = 1
    freq: float = 50.0


@dataclass
class ColoredNoiseEntry:
    noise_type: NoiseType = NoiseType.WHITE
    amplitude: float = 1.0
    enabled: bool = True


@dataclass
class FilterParams:
    filter_type: FilterType = FilterType.HANNING
    cutoff: float = 10.0
    r: float = 0.9


@dataclass
class SignalStatistics:
    total_power: float = 0.0
    mean_power: float = 0.0
    min_amplitude: float = 0.0
    max_amplitude: float = 0.0
    signal_mean: float = 0.0
    signal_variance: float = 0.0
    signal_std: float = 0.0
    acf_mean: float = 0.0
    acf_variance: float = 0.0
    acf_std: float = 0.0
    psd_mean: float = 0.0
    psd_variance: float = 0.0
    psd_std: float = 0.0


@dataclass
class SignalData:
    n: int = 0
    base: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise: np.ndarray = field(default_factory=lambda: np.zeros(0))
    colored_noise: np.ndarray = field(default_factory=lambda: np.zeros(0))
    combined: np.ndarray = field(default_factory=lambda: np.zeros(0))
    filtered: np.ndarray = field(default_factory=lambda: np.zeros(0))

    spectrum_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spectrum_phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    filtered_spectrum_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    freq_response: np.ndarray = field(default_factory=lambda: np.zeros(0))

    fft_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fft_phase: np.ndarray = field(default_factory=lambda: np.zeros(0))

    psd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    acf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phase_psd: np.ndarray = field(default_factory=lambda: np.zeros(0))

    statistics: SignalStatistics = field(default_factory=SignalStatistics)
    has_noise: bool = False

    segment_boundaries: list = field(default_factory=list)
    segment_labels: list = field(default_factory=list)

    def allocate(self, n: int):
        self.n = n
        for attr in [
            "base",
            "noise",
            "colored_noise",
            "combined",
            "filtered",
            "spectrum_mag",
            "spectrum_phase",
            "filtered_spectrum_mag",
            "freq_response",
            "fft_mag",
            "fft_phase",
            "psd",
            "acf",
            "phase_psd",
        ]:
            setattr(self, attr, np.zeros(n))


# ── Сегментированный сигнал ──


@dataclass
class SegmentPoolEntry:
    """Один элемент пула сегментов."""

    signal_type: SignalType = SignalType.HARMONIC
    enabled: bool = True

    # Гармонический / Пилообразный
    freq: float = 5.0
    amp: float = 1.0
    phase_deg: float = 0.0

    # Гауссов импульс
    gauss_amp: float = 1.0
    gauss_sigma: float = 0.05

    # Импульсный
    impulse_width: float = 0.02
    impulse_amp: float = 1.0
    impulse_period: float = 0.1

    # Экспоненциальный
    exp_alpha: float = -5.0
    exp_amp: float = 1.0

    # Речевой сигнал
    speech_file: str = ""
    speech_amp: float = 1.0

    def min_duration_for_one_period(self) -> float:
        """Минимальная длительность сегмента для одного полного периода."""
        match self.signal_type:
            case SignalType.HARMONIC:
                if self.freq > 0:
                    return 1.0 / self.freq
                return 0.01
            case SignalType.SAWTOOTH:
                if self.freq > 0:
                    return 1.0 / self.freq
                return 0.01
            case SignalType.IMPULSE:
                if self.impulse_period > 0:
                    return self.impulse_period
                return self.impulse_width * 2
            case SignalType.GAUSSIAN:
                return self.gauss_sigma * 6
            case SignalType.EXPONENTIAL_IMPULSE:
                if self.exp_alpha != 0:
                    return min(abs(3.0 / self.exp_alpha), 1.0)
                return 0.1
            case SignalType.SPEECH:
                return 0.01
        return 0.01


def _generate_segment_signal(
    entry: SegmentPoolEntry,
    n_samples: int,
    dt: float,
) -> np.ndarray:
    """Генерирует сигнал для одного сегмента."""
    match entry.signal_type:
        case SignalType.HARMONIC:
            return generate_harmonics(
                n_samples, dt, [entry.freq], [entry.amp], [np.radians(entry.phase_deg)]
            )
        case SignalType.GAUSSIAN:
            duration = n_samples * dt
            center = duration / 2.0
            return generate_gaussian(
                n_samples, dt, [center], [entry.gauss_amp], [entry.gauss_sigma]
            )
        case SignalType.SAWTOOTH:
            return generate_sawtooth(n_samples, dt, [entry.freq], [entry.amp])
        case SignalType.IMPULSE:
            return generate_impulse(
                n_samples,
                dt,
                [entry.impulse_width],
                [entry.impulse_amp],
                [entry.impulse_period],
            )
        case SignalType.EXPONENTIAL_IMPULSE:
            return generate_exponential_impulse(
                n_samples, dt, [entry.exp_alpha], [entry.exp_amp], [0.0]
            )
        case SignalType.SPEECH:
            if entry.speech_file:
                target_sr = 1.0 / dt
                signal = load_speech_signal(entry.speech_file, n_samples, target_sr)
                return signal * entry.speech_amp
            return np.zeros(n_samples)
    return np.zeros(n_samples)


def _build_colored_noise(
    n: int,
    noise_entries: list[ColoredNoiseEntry],
) -> np.ndarray:
    result = np.zeros(n)
    for entry in noise_entries:
        if entry.enabled:
            result += generate_colored_noise(entry.noise_type, n, entry.amplitude)
    return result


def _finalize_signal_data(
    data: SignalData,
    dt: float,
    colored_noise_entries: list[ColoredNoiseEntry] | None = None,
) -> SignalData:
    """Общая финализация: шумы, спектры, статистики."""
    total_n = data.n

    if colored_noise_entries:
        data.colored_noise = _build_colored_noise(total_n, colored_noise_entries)

    data.combined = data.base + data.noise + data.colored_noise

    has_noise = colored_noise_entries and any(e.enabled for e in colored_noise_entries)
    data.has_noise = has_noise

    _, _, mag, phase_deg = compute_dft(data.combined)
    data.spectrum_mag = mag
    data.spectrum_phase = phase_deg

    fft_mag, fft_phase, _ = compute_fft(data.combined)
    data.fft_mag = fft_mag
    data.fft_phase = fft_phase

    data.psd = compute_psd(data.combined, dt)
    data.acf = compute_acf(data.combined)

    if has_noise:
        _, _, spectrum = compute_fft(data.combined)
        phase_signal = np.angle(spectrum)
        data.phase_psd = compute_psd(phase_signal, dt)

    data.statistics = compute_statistics(data.combined, dt)

    return data


def generate_single_type_signal(
    params: SignalParams,
    entry: SegmentPoolEntry,
    colored_noise_entries: list[ColoredNoiseEntry] | None = None,
) -> SignalData:
    """Генерирует сигнал одного типа на всю длительность."""
    n = params.n_samples
    dt = params.dt
    data = SignalData()
    data.allocate(n)

    data.base = _generate_segment_signal(entry, n, dt)

    return _finalize_signal_data(data, dt, colored_noise_entries)


def generate_segmented_signal(
    params: SignalParams,
    pool: list[SegmentPoolEntry],
    seg_min_duration: float = 0.05,
    seg_max_duration: float = 0.3,
    colored_noise_entries: list[ColoredNoiseEntry] | None = None,
) -> SignalData:
    """Генерирует сигнал из случайно расположенных сегментов."""
    dt = params.dt
    total_n = params.n_samples
    data = SignalData()
    data.allocate(total_n)

    enabled_pool = [e for e in pool if e.enabled]
    if not enabled_pool:
        data.statistics = compute_statistics(data.combined, dt)
        return data

    rng = np.random.default_rng()

    result = np.zeros(total_n)
    boundaries = []
    labels = []

    type_names = {
        SignalType.HARMONIC: "Гарм",
        SignalType.GAUSSIAN: "Гаусс",
        SignalType.SAWTOOTH: "Пила",
        SignalType.IMPULSE: "Импульс",
        SignalType.EXPONENTIAL_IMPULSE: "Эксп",
        SignalType.SPEECH: "Речь",
    }

    pos = 0

    while pos < total_n:
        remaining_time = (total_n - pos) * dt

        indices = list(range(len(enabled_pool)))
        rng.shuffle(indices)

        placed = False
        for idx in indices:
            entry = enabled_pool[idx]

            min_period = entry.min_duration_for_one_period()
            effective_min = max(seg_min_duration, min_period)
            effective_max = max(seg_max_duration, effective_min)

            if remaining_time < effective_min:
                continue

            actual_max = min(effective_max, remaining_time)
            actual_min = effective_min

            if actual_min > actual_max:
                continue

            seg_duration = rng.uniform(actual_min, actual_max)
            seg_n = int(round(seg_duration / dt))

            min_samples = max(1, int(np.ceil(effective_min / dt)))
            seg_n = max(seg_n, min_samples)

            if pos + seg_n > total_n:
                seg_n = total_n - pos

            if seg_n < min_samples:
                continue

            if seg_n <= 0:
                continue

            seg_signal = _generate_segment_signal(entry, seg_n, dt)

            fade_len = max(1, seg_n // 20)
            if seg_n > fade_len * 2:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                seg_signal[:fade_len] *= fade_in
                seg_signal[-fade_len:] *= fade_out

            result[pos : pos + seg_n] += seg_signal

            label = type_names.get(entry.signal_type, "?")
            boundaries.append((pos, pos + seg_n))
            labels.append(label)

            pos += seg_n
            placed = True
            break

        if not placed:
            break

    data.base = result
    data.segment_boundaries = boundaries
    data.segment_labels = labels

    return _finalize_signal_data(data, dt, colored_noise_entries)


def export_signal_to_wav(
    signal: np.ndarray,
    sample_rate: float,
    file_path: str,
) -> str:
    """
    Экспортирует сигнал в WAV файл (16-bit PCM).
    Возвращает пустую строку при успехе или сообщение об ошибке.
    """
    import wave

    try:
        if len(signal) == 0:
            return "Сигнал пуст — нечего экспортировать."

        # Нормализация в диапазон int16
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            normalized = signal

        # Преобразуем в int16
        int_data = np.clip(normalized * 32767, -32768, 32767).astype(np.int16)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(int_data.tobytes())

        return ""

    except Exception as e:
        return f"Ошибка экспорта: {e}"


def compute_statistics(signal: np.ndarray, dt: float) -> SignalStatistics:
    stats = SignalStatistics()
    if len(signal) == 0:
        return stats

    stats.total_power = np.sum(signal**2) * dt
    stats.mean_power = np.mean(signal**2)
    stats.min_amplitude = np.min(signal)
    stats.max_amplitude = np.max(signal)
    stats.signal_mean = np.mean(signal)
    stats.signal_variance = np.var(signal)
    stats.signal_std = np.std(signal)

    acf = compute_acf(signal)
    stats.acf_mean = np.mean(acf)
    stats.acf_variance = np.var(acf)
    stats.acf_std = np.std(acf)

    psd = compute_psd(signal, dt)
    stats.psd_mean = np.mean(psd)
    stats.psd_variance = np.var(psd)
    stats.psd_std = np.std(psd)

    return stats


def apply_filter_to_data(
    data: SignalData,
    filter_params: FilterParams,
    sample_rate: float,
    noise_enabled: bool,
) -> SignalData:
    n = data.n
    ft = filter_params.filter_type
    r = filter_params.r
    fc = filter_params.cutoff

    data.filtered = apply_filter(data.combined, ft, r, fc, sample_rate)
    data.freq_response = compute_frequency_response(ft, r, fc, sample_rate, n)

    _, _, mag, _ = compute_dft(data.filtered)
    if noise_enabled:
        mag = mag**2 / n
    data.filtered_spectrum_mag = mag

    dt = 1.0 / sample_rate
    data.statistics = compute_statistics(data.filtered, dt)

    return data
