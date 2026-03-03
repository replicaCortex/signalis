import sys

import numpy as np

sys.path.insert(0, "src")

from analysis.mod import compute_acf, compute_psd, compute_stats
from signals.mod import add_noise, gaussian_pulse, polyharmonic, sawtooth
from ui.mod import App
from wav.wav import read_wav, write_wav


def generate_signal(signal_type: str, t: np.ndarray) -> np.ndarray:
    """Генерация сигнала по типу"""
    match signal_type:
        case "gaussian":
            return gaussian_pulse(t, a=1.0, t0=0.5, sigma=0.1)
        case "sawtooth":
            return sawtooth(t, freq=5.0, a=1.0)
        case "polyharmonic":
            return polyharmonic(t, freqs=[1, 3, 5], amps=[1, 0.5, 0.3])
        case _:
            return np.zeros_like(t)


def main():
    app = App(width=1200, height=600)
    app.run(
        generate_signal=generate_signal,
        compute_stats=compute_stats,
        compute_acf=compute_acf,
        compute_psd=compute_psd,
        add_noise=add_noise,
        read_wav=read_wav,
        write_wav=write_wav,
    )


if __name__ == "__main__":
    main()
