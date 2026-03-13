from dataclasses import dataclass, field

import numpy as np

from dsp import (
    SignalType,
    compute_dft,
    generate_gaussian,
    generate_harmonics,
    generate_noise,
    generate_sawtooth,
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
class Harmonic:
    freq: float
    amp: float
    phase_deg: float

    @property
    def phase_rad(self) -> float:
        return np.radians(self.phase_deg)


@dataclass
class NoiseParams:
    enabled: bool = False
    amp_min: int = 0
    amp_max: int = 1
    freq: float = 50.0


@dataclass
class FilterParams:
    filter_type: FilterType = FilterType.HANNING
    cutoff: float = 10.0
    r: float = 0.9


@dataclass
class SignalData:
    n: int = 0
    base: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise: np.ndarray = field(default_factory=lambda: np.zeros(0))
    combined: np.ndarray = field(default_factory=lambda: np.zeros(0))
    filtered: np.ndarray = field(default_factory=lambda: np.zeros(0))

    spectrum_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spectrum_phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    filtered_spectrum_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    freq_response: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def allocate(self, n: int):
        self.n = n
        for attr in [
            "base",
            "noise",
            "combined",
            "filtered",
            "spectrum_mag",
            "spectrum_phase",
            "filtered_spectrum_mag",
            "freq_response",
        ]:
            setattr(self, attr, np.zeros(n))


def _build_base_signal(
    signal_type: SignalType,
    n: int,
    dt: float,
    harmonics: list[Harmonic],
) -> np.ndarray:
    """Генерирует базовый сигнал выбранного типа."""
    freqs = [h.freq for h in harmonics]
    amps = [h.amp for h in harmonics]
    phases = [h.phase_rad for h in harmonics]

    match signal_type:
        case SignalType.HARMONIC:
            return generate_harmonics(n, dt, freqs, amps, phases)
        case SignalType.GAUSSIAN:
            return generate_gaussian(n, dt, freqs, amps)
        case SignalType.SAWTOOTH:
            return generate_sawtooth(n, dt, freqs, amps)


def generate_signal(
    params: SignalParams,
    harmonics: list[Harmonic],
    noise_params: NoiseParams,
    signal_type: SignalType = SignalType.HARMONIC,
) -> SignalData:
    """Генерирует сигнал и вычисляет его спектр."""
    n = params.n_samples
    dt = params.dt
    data = SignalData()
    data.allocate(n)

    data.base = _build_base_signal(signal_type, n, dt, harmonics)

    if noise_params.enabled and harmonics:
        data.noise = generate_noise(
            n,
            dt,
            noise_params.freq,
            noise_params.amp_min,
            noise_params.amp_max,
            n_components=len(harmonics),
        )
        data.combined = data.base + data.noise
    else:
        data.combined = data.base.copy()

    _, _, mag, phase_deg = compute_dft(data.combined)

    # if noise_params.enabled:
    #     mag = mag**2 / n

    data.spectrum_mag = mag
    data.spectrum_phase = phase_deg

    return data


def apply_filter_to_data(
    data: SignalData,
    filter_params: FilterParams,
    sample_rate: float,
    noise_enabled: bool,
) -> SignalData:
    """Применяет фильтр и пересчитывает спектры."""
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

    return data
