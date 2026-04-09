from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SegmentFeatures:
    """Признаки сегмента для классификации."""

    mean: float
    std: float
    peak_freq: float
    peak_amplitude: float
    zero_crossing_rate: float
    energy: float


def extract_features(signal: np.ndarray, dt: float) -> SegmentFeatures:
    """Извлекает признаки из сегмента сигнала."""
    # Базовые статистики
    mean = np.mean(signal)
    std = np.std(signal)
    energy = np.sum(signal**2)

    # Частота нулевых пересечений
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    zcr = zero_crossings / len(signal) if len(signal) > 0 else 0.0

    # Спектральные характеристики
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum[: len(spectrum) // 2])
    freqs = np.fft.fftfreq(len(signal), dt)[: len(spectrum) // 2]

    if len(magnitude) > 0:
        peak_idx = np.argmax(magnitude)
        peak_freq = abs(freqs[peak_idx])
        peak_amplitude = magnitude[peak_idx]
    else:
        peak_freq = 0.0
        peak_amplitude = 0.0

    return SegmentFeatures(
        mean=mean,
        std=std,
        peak_freq=peak_freq,
        peak_amplitude=peak_amplitude,
        zero_crossing_rate=zcr,
        energy=energy,
    )


def features_to_vector(features: SegmentFeatures) -> np.ndarray:
    """Преобразует признаки в вектор для кластеризации."""
    return np.array(
        [
            features.mean,
            features.std,
            features.peak_freq,
            features.peak_amplitude,
            features.zero_crossing_rate,
            features.energy,
        ]
    )


class SignalClassifier:
    """K-means классификатор сегментов сигнала."""

    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels_map = {}  # кластер -> метка
        self.mean = None
        self.std = None

    def _normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Нормализация признаков."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # избегаем деления на 0
        X_norm = (X - mean) / std
        return X_norm, mean, std

    def fit(self, features_list: List[SegmentFeatures], labels: List[str]):
        """Обучение на эталонных сегментах."""
        # Преобразуем признаки в матрицу
        X = np.array([features_to_vector(f) for f in features_list])

        # Нормализация
        X_norm, self.mean, self.std = self._normalize(X)

        # K-means
        n_samples = len(X_norm)
        self.n_clusters = min(self.n_clusters, n_samples)

        # Инициализация центроидов (метод k-means++)
        rng = np.random.default_rng(42)
        centroids = [X_norm[rng.integers(0, n_samples)]]

        for _ in range(1, self.n_clusters):
            distances = np.array(
                [min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X_norm]
            )
            probs = distances / distances.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids.append(X_norm[idx])

        self.centroids = np.array(centroids)

        # Итеративное уточнение
        for iteration in range(self.max_iter):
            # Назначение кластеров
            cluster_assignments = []
            for x in X_norm:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster_assignments.append(np.argmin(distances))

            # Обновление центроидов
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X_norm[np.array(cluster_assignments) == k]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(self.centroids[k])

            new_centroids = np.array(new_centroids)

            # Проверка сходимости
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break

            self.centroids = new_centroids

        # Сопоставление кластеров с метками
        cluster_labels = {}
        for i, (label, cluster_id) in enumerate(zip(labels, cluster_assignments)):
            if cluster_id not in cluster_labels:
                cluster_labels[cluster_id] = {}
            cluster_labels[cluster_id][label] = (
                cluster_labels[cluster_id].get(label, 0) + 1
            )

        # Выбираем наиболее частую метку для каждого кластера
        self.labels_map = {}
        for cluster_id, label_counts in cluster_labels.items():
            self.labels_map[cluster_id] = max(label_counts.items(), key=lambda x: x[1])[
                0
            ]

    def predict_with_confidence(
        self, features: SegmentFeatures
    ) -> Tuple[str, Dict[str, float]]:
        """
        Классифицирует сегмент и возвращает метку + вероятности для всех классов.

        Returns:
            (predicted_label, confidence_dict) где confidence_dict = {label: probability}
            Сумма всех вероятностей = 1.0
        """
        if self.centroids is None:
            return "Неизвестно", {}

        # Нормализация
        x = features_to_vector(features)
        x_norm = (x - self.mean) / self.std

        # Вычисление расстояний до всех центроидов
        distances = np.array([np.linalg.norm(x_norm - c) for c in self.centroids])

        # ИСПРАВЛЕНО: Преобразуем расстояния в вероятности через softmax
        # Используем отрицательные расстояния с temperature scaling
        temperature = 1.0  # Можно настраивать для контроля "уверенности"

        # Вариант 1: Softmax на основе отрицательных расстояний
        neg_distances = -distances / temperature
        # Вычитаем максимум для численной стабильности
        neg_distances_shifted = neg_distances - np.max(neg_distances)
        exp_neg_dist = np.exp(neg_distances_shifted)
        probabilities = exp_neg_dist / np.sum(exp_neg_dist)

        # ПРОВЕРКА: сумма вероятностей должна быть 1.0
        assert np.isclose(np.sum(probabilities), 1.0), (
            f"Сумма вероятностей = {np.sum(probabilities)}"
        )

        # Создаем словарь уверенности по меткам
        confidence_dict = {}
        for cluster_id, prob in enumerate(probabilities):
            label = self.labels_map.get(cluster_id, f"Кластер_{cluster_id}")
            # Если несколько кластеров имеют одну метку, суммируем их вероятности
            confidence_dict[label] = confidence_dict.get(label, 0.0) + prob

        # ПРОВЕРКА: сумма вероятностей по меткам тоже должна быть 1.0
        total_prob = sum(confidence_dict.values())
        if not np.isclose(total_prob, 1.0):
            # Нормализуем на всякий случай
            confidence_dict = {k: v / total_prob for k, v in confidence_dict.items()}

        # Предсказанная метка - с максимальной вероятностью
        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]

        return predicted_label, confidence_dict


def segment_and_classify(
    signal: np.ndarray, dt: float, window_size: int, classifier: SignalClassifier
) -> Tuple[List[Tuple[int, int]], List[str], List[Dict[str, float]]]:
    """
    Сегментирует сигнал скользящим окном и классифицирует каждый сегмент.

    Returns:
        (boundaries, labels, confidences)
    """
    boundaries = []
    labels = []
    confidences = []

    n = len(signal)
    step = window_size // 2  # 50% перекрытие

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        segment = signal[start:end]

        # Извлечение признаков
        features = extract_features(segment, dt)

        # Классификация
        label, confidence = classifier.predict_with_confidence(features)

        boundaries.append((start, end))
        labels.append(label)
        confidences.append(confidence)

    return boundaries, labels, confidences
