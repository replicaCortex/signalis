from dataclasses import dataclass

import numpy as np


@dataclass
class SignalStats:
    total_power: float  # Полная мощность
    mean_power: float  # Средняя мощность
    min_amp: float  # Минимальная амплитуда
    max_amp: float  # Максимальная амплитуда
    mean: float  # Математическое ожидание
    variance: float  # Дисперсия
    std: float  # СКО


def compute_stats(signal: np.ndarray) -> SignalStats:
    """Вычисление статистических параметров сигнала"""

    return SignalStats(
        total_power=np.sum(signal**2),
        mean_power=np.mean(signal**2),
        min_amp=np.min(signal),
        max_amp=np.max(signal),
        mean=np.mean(signal),
        variance=np.var(signal),
        std=np.std(signal),
    )
