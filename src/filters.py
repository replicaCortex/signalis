from enum import IntEnum

import numpy as np


class FilterType(IntEnum):
    HANNING = 0
    PARABOLIC = 1
    FIRST_ORDER = 2
    LOW_PASS = 3
    HIGH_PASS = 4
    BAND_PASS = 5
    BAND_STOP = 6


def get_coefficients(
    ft: FilterType, r: float, fc: float, fd: float
) -> tuple[float, float, float, float, float]:
    wc = 2 * np.pi * fc / fd

    match ft:
        case FilterType.HANNING:
            return 0.25, 0.5, 0.25, 0.0, 0.0
        case FilterType.PARABOLIC:
            return 1, 0, 0, 0, 0
        case FilterType.FIRST_ORDER:
            return 1, 0, 0, -r, 0
        case FilterType.LOW_PASS:
            return 1, 2, 1, -2 * r * np.cos(wc), r**2
        case FilterType.HIGH_PASS:
            return 1, -2, 1, -2 * r * np.cos(wc), r**2
        case FilterType.BAND_PASS:
            return 1, 0, -1, -2 * r * np.cos(wc), r**2
        case FilterType.BAND_STOP:
            return 1, -2 * np.cos(wc), 1, -r * np.cos(wc), r**2


def apply_filter(
    signal: np.ndarray,
    ft: FilterType,
    r: float,
    fc: float,
    fd: float,
) -> np.ndarray:
    """Применяет фильтр к сигналу."""
    n = len(signal)
    out = np.zeros(n)

    if ft == FilterType.PARABOLIC:
        for i in range(2, n - 2):
            out[i] = (
                -3 * signal[i - 2]
                + 12 * signal[i - 1]
                + 17 * signal[i]
                + 12 * signal[i + 1]
                - 3 * signal[i + 2]
            ) / 35
        return out

    a0, a1, a2, b1, b2 = get_coefficients(ft, r, fc, fd)

    start = 1 if ft == FilterType.FIRST_ORDER else 2
    for i in range(start, n):
        out[i] = a0 * signal[i]
        if i >= 1:
            out[i] += a1 * signal[i - 1] - b1 * out[i - 1]
        if i >= 2:
            out[i] += a2 * signal[i - 2] - b2 * out[i - 2]

    return out


def compute_frequency_response(
    ft: FilterType, r: float, fc: float, fd: float, n: int
) -> np.ndarray:
    """Вычисляет АЧХ фильтра."""
    a0, a1, a2, b1, b2 = get_coefficients(ft, r, fc, fd)

    j = np.arange(n)
    w1 = 2 * np.pi * j / n
    w2 = 2 * w1

    num_cos = a0 + a1 * np.cos(w1) + a2 * np.cos(w2)
    num_sin = a1 * np.sin(w1) + a2 * np.sin(w2)
    den_cos = 1 + b1 * np.cos(w1) + b2 * np.cos(w2)
    den_sin = b1 * np.sin(w1) + b2 * np.sin(w2)

    # Добавляем защиту от деления на ноль
    denominator = den_cos**2 + den_sin**2
    denominator = np.where(denominator == 0, 1e-10, denominator)

    response = np.sqrt((num_cos**2 + num_sin**2) / denominator)

    # Нормализуем АЧХ
    max_response = np.max(response)
    if max_response > 0:
        response = response / max_response

    return response
