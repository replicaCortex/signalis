from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np


class ClassificationMethod(IntEnum):
    """Методы классификации."""

    KMEANS = 0
    KNN = 1
    EUCLIDEAN_DISTANCE = 2
    CORRELATION = 3
    DTW = 4  # Dynamic Time Warping


@dataclass
class SegmentFeatures:
    """Признаки сегмента для классификации."""

    mean: float
    std: float
    peak_freq: float
    peak_amplitude: float
    zero_crossing_rate: float
    energy: float
    # Дополнительные признаки
    skewness: float = 0.0
    kurtosis: float = 0.0
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0


def extract_features(signal: np.ndarray, dt: float) -> SegmentFeatures:
    """Извлекает признаки из сегмента сигнала."""
    # Базовые статистики
    mean = np.mean(signal)
    std = np.std(signal)
    energy = np.sum(signal**2)

    # Частота нулевых пересечений
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    zcr = zero_crossings / len(signal) if len(signal) > 0 else 0.0

    # Асимметрия (skewness) и эксцесс (kurtosis)
    if std > 0:
        normalized = (signal - mean) / std
        skewness = np.mean(normalized**3)
        kurtosis = np.mean(normalized**4) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Спектральные характеристики
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum[: len(spectrum) // 2])
    freqs = np.fft.fftfreq(len(signal), dt)[: len(spectrum) // 2]

    if len(magnitude) > 0 and np.sum(magnitude) > 0:
        peak_idx = np.argmax(magnitude)
        peak_freq = abs(freqs[peak_idx])
        peak_amplitude = magnitude[peak_idx]

        # Спектральный центроид (центр масс спектра)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

        # Спектральная ширина полосы
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude)
        )
    else:
        peak_freq = 0.0
        peak_amplitude = 0.0
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0

    return SegmentFeatures(
        mean=mean,
        std=std,
        peak_freq=peak_freq,
        peak_amplitude=peak_amplitude,
        zero_crossing_rate=zcr,
        energy=energy,
        skewness=skewness,
        kurtosis=kurtosis,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
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
            features.skewness,
            features.kurtosis,
            features.spectral_centroid,
            features.spectral_bandwidth,
        ]
    )


class SignalClassifier:
    """Универсальный классификатор сегментов сигнала."""

    def __init__(
        self,
        method: ClassificationMethod = ClassificationMethod.KMEANS,
        n_clusters: int = 5,
        max_iter: int = 100,
        k_neighbors: int = 3,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.k_neighbors = k_neighbors

        # Для K-means
        self.centroids = None
        self.labels_map = {}

        # Для KNN и других методов
        self.reference_features = []
        self.reference_labels = []
        self.reference_signals = []  # Для DTW

        self.mean = None
        self.std = None

    def _normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Нормализация признаков."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X_norm = (X - mean) / std
        return X_norm, mean, std

    def _kmeans_fit(self, X_norm: np.ndarray, labels: List[str]):
        """K-means кластеризация."""
        n_samples = len(X_norm)
        self.n_clusters = min(self.n_clusters, n_samples)

        # Инициализация центроидов (метод k-means++)
        rng = np.random.default_rng(42)
        centroids = [X_norm[rng.integers(0, n_samples)]]

        for _ in range(1, self.n_clusters):
            distances = np.array(
                [min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X_norm]
            )
            dist_sum = distances.sum()
            if dist_sum > 0 and np.isfinite(dist_sum):
                probs = distances / dist_sum
            else:
                # Все точки уже покрыты центроидами — равномерное распределение
                probs = np.ones(n_samples) / n_samples
            idx = rng.choice(n_samples, p=probs)
            centroids.append(X_norm[idx])

        self.centroids = np.array(centroids)

        # Итеративное уточнение
        for iteration in range(self.max_iter):
            cluster_assignments = []
            for x in X_norm:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster_assignments.append(np.argmin(distances))

            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = X_norm[np.array(cluster_assignments) == k]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(self.centroids[k])

            new_centroids = np.array(new_centroids)

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

        self.labels_map = {}
        for cluster_id, label_counts in cluster_labels.items():
            self.labels_map[cluster_id] = max(label_counts.items(), key=lambda x: x[1])[
                0
            ]

    def fit(
        self,
        features_list: List[SegmentFeatures],
        labels: List[str],
        signals: List[np.ndarray] = None,
    ):
        """Обучение классификатора."""
        X = np.array([features_to_vector(f) for f in features_list])
        X_norm, self.mean, self.std = self._normalize(X)

        # Сохраняем эталонные данные
        self.reference_features = features_list
        self.reference_labels = labels
        if signals is not None:
            self.reference_signals = signals

        # Обучаем в зависимости от метода
        if self.method == ClassificationMethod.KMEANS:
            self._kmeans_fit(X_norm, labels)

    def _predict_kmeans(self, x_norm: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Классификация методом K-means."""
        distances = np.array([np.linalg.norm(x_norm - c) for c in self.centroids])

        # Softmax с отрицательными расстояниями (численно стабильный)
        temperature = 1.0
        neg_distances = -distances / temperature
        # Численно стабильный softmax: вычитаем максимум
        neg_distances_shifted = neg_distances - np.max(neg_distances)
        exp_neg_dist = np.exp(neg_distances_shifted)
        sum_exp = np.sum(exp_neg_dist)

        if sum_exp <= 0 or not np.isfinite(sum_exp):
            # Fallback: равномерное распределение
            probabilities = np.ones(len(distances)) / len(distances)
        else:
            probabilities = exp_neg_dist / sum_exp

        confidence_dict = {}
        for cluster_id, prob in enumerate(probabilities):
            label = self.labels_map.get(cluster_id, f"Кластер_{cluster_id}")
            confidence_dict[label] = confidence_dict.get(label, 0.0) + prob

        total_prob = sum(confidence_dict.values())
        if total_prob > 0 and not np.isclose(total_prob, 1.0):
            confidence_dict = {k: v / total_prob for k, v in confidence_dict.items()}

        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]
        return predicted_label, confidence_dict

    def _predict_knn(self, x_norm: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Классификация методом K-ближайших соседей."""
        # Вычисляем расстояния до всех эталонов
        ref_vectors = np.array([features_to_vector(f) for f in self.reference_features])
        ref_norm = (ref_vectors - self.mean) / self.std

        distances = np.array([np.linalg.norm(x_norm - r) for r in ref_norm])

        # Находим K ближайших
        k = min(self.k_neighbors, len(distances))
        nearest_indices = np.argsort(distances)[:k]

        # Голосование с весами (обратные расстояния)
        votes = {}
        total_weight = 0.0

        for idx in nearest_indices:
            label = self.reference_labels[idx]
            dist = distances[idx]
            weight = 1.0 / (dist + 1e-10)  # Избегаем деления на 0

            if not np.isfinite(weight):
                weight = 1e10  # Большой вес для очень близких соседей

            votes[label] = votes.get(label, 0.0) + weight
            total_weight += weight

        # Нормализуем в вероятности
        if total_weight > 0:
            confidence_dict = {
                label: vote / total_weight for label, vote in votes.items()
            }
        else:
            confidence_dict = {label: 1.0 / len(votes) for label in votes}

        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]
        return predicted_label, confidence_dict

    def _predict_euclidean(self, x_norm: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Классификация по евклидову расстоянию до ближайшего эталона."""
        ref_vectors = np.array([features_to_vector(f) for f in self.reference_features])
        ref_norm = (ref_vectors - self.mean) / self.std

        distances = np.array([np.linalg.norm(x_norm - r) for r in ref_norm])

        # Численно стабильный softmax
        temperature = 0.5
        neg_distances = -distances / temperature
        neg_distances_shifted = neg_distances - np.max(neg_distances)
        exp_neg_dist = np.exp(neg_distances_shifted)
        sum_exp = np.sum(exp_neg_dist)

        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probabilities = np.ones(len(distances)) / len(distances)
        else:
            probabilities = exp_neg_dist / sum_exp

        # Группируем по меткам
        confidence_dict = {}
        for label, prob in zip(self.reference_labels, probabilities):
            confidence_dict[label] = confidence_dict.get(label, 0.0) + prob

        # Нормализуем
        total = sum(confidence_dict.values())
        if total > 0:
            confidence_dict = {k: v / total for k, v in confidence_dict.items()}

        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]
        return predicted_label, confidence_dict

    def _predict_correlation(
        self, features: SegmentFeatures
    ) -> Tuple[str, Dict[str, float]]:
        """Классификация по корреляции признаков."""
        x = features_to_vector(features)

        correlations = []
        for ref_feat in self.reference_features:
            ref_vec = features_to_vector(ref_feat)

            # Нормализуем векторы
            x_norm_val = x - np.mean(x)
            ref_norm_val = ref_vec - np.mean(ref_vec)

            x_std = np.std(x)
            ref_std = np.std(ref_vec)

            if x_std > 0 and ref_std > 0:
                corr = np.dot(x_norm_val, ref_norm_val) / (len(x) * x_std * ref_std)
            else:
                corr = 0.0

            correlations.append(corr)

        # Преобразуем корреляции в вероятности
        correlations = np.array(correlations)
        correlations_pos = (correlations + 1.0) / 2.0  # 0..1

        # Численно стабильный softmax
        temperature = 0.3
        scaled = correlations_pos / temperature
        scaled_shifted = scaled - np.max(scaled)
        exp_corr = np.exp(scaled_shifted)
        sum_exp = np.sum(exp_corr)

        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probabilities = np.ones(len(correlations)) / len(correlations)
        else:
            probabilities = exp_corr / sum_exp

        # Группируем по меткам
        confidence_dict = {}
        for label, prob in zip(self.reference_labels, probabilities):
            confidence_dict[label] = confidence_dict.get(label, 0.0) + prob

        total = sum(confidence_dict.values())
        if total > 0:
            confidence_dict = {k: v / total for k, v in confidence_dict.items()}

        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]
        return predicted_label, confidence_dict

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Dynamic Time Warping расстояние между двумя сигналами."""
        n, m = len(s1), len(s2)

        # Ограничиваем размер для производительности
        if n > 1000 or m > 1000:
            # Downsample
            factor = max(n // 500, m // 500, 1)
            s1 = s1[::factor]
            s2 = s2[::factor]
            n, m = len(s1), len(s2)

        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        return dtw_matrix[n, m]

    def _predict_dtw(self, signal: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Классификация методом Dynamic Time Warping."""
        if not self.reference_signals:
            # Fallback на евклидово расстояние
            x = features_to_vector(extract_features(signal, 1.0))
            x_norm = (x - self.mean) / self.std
            return self._predict_euclidean(x_norm)

        distances = []
        for ref_signal in self.reference_signals:
            dist = self._dtw_distance(signal, ref_signal)
            distances.append(dist)

        distances = np.array(distances)

        # Численно стабильный softmax
        mean_dist = np.mean(distances)
        temperature = mean_dist if mean_dist > 0 else 1.0
        neg_distances = -distances / temperature
        neg_distances_shifted = neg_distances - np.max(neg_distances)
        exp_neg_dist = np.exp(neg_distances_shifted)
        sum_exp = np.sum(exp_neg_dist)

        if sum_exp <= 0 or not np.isfinite(sum_exp):
            probabilities = np.ones(len(distances)) / len(distances)
        else:
            probabilities = exp_neg_dist / sum_exp

        confidence_dict = {}
        for label, prob in zip(self.reference_labels, probabilities):
            confidence_dict[label] = confidence_dict.get(label, 0.0) + prob

        total = sum(confidence_dict.values())
        if total > 0:
            confidence_dict = {k: v / total for k, v in confidence_dict.items()}

        predicted_label = max(confidence_dict.items(), key=lambda x: x[1])[0]
        return predicted_label, confidence_dict

    def predict_with_confidence(
        self, features: SegmentFeatures, signal: np.ndarray = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Классифицирует сегмент и возвращает метку + вероятности.
        """
        if self.method == ClassificationMethod.DTW:
            if signal is None:
                raise ValueError("DTW требует исходный сигнал")
            return self._predict_dtw(signal)

        if self.method == ClassificationMethod.CORRELATION:
            return self._predict_correlation(features)

        # Для остальных методов нужна нормализация
        x = features_to_vector(features)
        x_norm = (x - self.mean) / self.std

        if self.method == ClassificationMethod.KMEANS:
            return self._predict_kmeans(x_norm)
        elif self.method == ClassificationMethod.KNN:
            return self._predict_knn(x_norm)
        elif self.method == ClassificationMethod.EUCLIDEAN_DISTANCE:
            return self._predict_euclidean(x_norm)

        return "Неизвестно", {}
