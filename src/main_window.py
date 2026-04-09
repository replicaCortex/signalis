import numpy as np
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QTabWidget, QWidget

from app import (
    FilterParams,
    SignalData,
    SignalParams,
    apply_filter_to_data,
    export_signal_to_wav,
    generate_segmented_signal,
    generate_single_type_signal,
)
from classifier import SignalClassifier, extract_features, segment_and_classify
from widgets.classifier_panel import ClassifierPanel  # НОВОЕ
from widgets.noise_filter_panel import NoiseFilterPanel
from widgets.plot_panel import PlotPanel
from widgets.segmented_panel import SEGMENT_TYPE_NAMES, SegmentedPanel
from widgets.statistics_panel import StatisticsPanel
from widgets.theory_panel import TheoryPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FooBarPrime")
        self.setGeometry(100, 100, 1400, 800)

        self.data = SignalData()
        self.params = SignalParams()
        self.classifier = None  # НОВОЕ

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Левая часть
        self.left_tabs = QTabWidget()

        self.segmented_panel = SegmentedPanel()
        self.noise_filter_panel = NoiseFilterPanel()
        self.classifier_panel = ClassifierPanel()  # НОВОЕ
        self.statistics_panel = StatisticsPanel()
        self.theory_panel = TheoryPanel()

        self.left_tabs.addTab(self.segmented_panel, "Сигнал")
        self.left_tabs.addTab(self.noise_filter_panel, "Шумы и фильтры")
        self.left_tabs.addTab(self.classifier_panel, "Классификация")  # НОВОЕ
        self.left_tabs.addTab(self.statistics_panel, "Характеристики")
        self.left_tabs.addTab(self.theory_panel, "Теория")

        # Правая часть
        self.plot_panel = PlotPanel()

        layout.addWidget(self.left_tabs, stretch=1)
        layout.addWidget(self.plot_panel, stretch=3)

    def _connect_signals(self):
        # Панель шумов/фильтров
        nfp = self.noise_filter_panel
        nfp.filter_requested.connect(self._on_filter)
        nfp.model_requested.connect(self._on_generate)

        # Панель сигнала
        self.segmented_panel.generate_requested.connect(self._on_generate)

        # НОВОЕ: Панель классификации
        self.classifier_panel.classify_requested.connect(self._on_classify)

        # Панель характеристик — экспорт WAV
        self.statistics_panel.export_wav_requested.connect(self._on_export_wav)

        # Чекбоксы отображения
        for cb in [
            self.plot_panel.cb_base,
            self.plot_panel.cb_noise,
            self.plot_panel.cb_colored_noise,
            self.plot_panel.cb_combined,
            self.plot_panel.cb_filtered,
        ]:
            cb.stateChanged.connect(self._refresh_plots)

    def _read_params(self) -> SignalParams:
        sp = self.segmented_panel
        return SignalParams(
            duration=sp.get_duration(),
            sample_rate=sp.get_sample_rate(),
        )

    def _read_filter(self) -> FilterParams:
        nfp = self.noise_filter_panel
        return FilterParams(
            filter_type=nfp.selected_filter,
            cutoff=nfp.get_filter_cutoff(),
            r=nfp.get_filter_r(),
        )

    def _on_generate(self):
        self.params = self._read_params()
        colored_entries = self.noise_filter_panel.get_noise_entries()
        sp = self.segmented_panel

        if sp.is_single_type:
            entry = sp.get_first_enabled_entry()
            if entry is None:
                return
            self.data = generate_single_type_signal(
                self.params,
                entry=entry,
                colored_noise_entries=colored_entries,
            )
        else:
            pool = sp.get_pool_entries()
            self.data = generate_segmented_signal(
                self.params,
                pool=pool,
                seg_min_duration=sp.get_seg_min_duration(),
                seg_max_duration=sp.get_seg_max_duration(),
                colored_noise_entries=colored_entries,
                min_periods=sp.get_min_periods(),
            )

        # Сбросить данные классификации при новой генерации
        self.data.classification_boundaries = []
        self.data.classification_labels = []
        self.data.classification_confidences = []

        self._refresh_plots()

    def _on_classify(self):
        """НОВОЕ: Обучает классификатор и применяет к сигналу."""
        if self.data.n == 0:
            self.classifier_panel.set_status(
                "Сначала сгенерируйте сигнал!", is_error=True
            )
            return

        references = self.classifier_panel.get_references()

        if len(references) < 2:
            self.classifier_panel.set_status(
                "Добавьте минимум 2 эталонных сигнала!", is_error=True
            )
            return

        from app import _generate_segment_signal

        dt = self.params.dt
        window_size_sec = self.classifier_panel.get_window_size()
        window_size = int(window_size_sec / dt)

        if window_size < 10:
            self.classifier_panel.set_status("Размер окна слишком мал!", is_error=True)
            return

        # Генерируем эталонные сегменты
        features_list = []
        labels_list = []

        for name, entry in references:
            # Генерируем эталон
            seg_signal = _generate_segment_signal(entry, window_size, dt)
            features = extract_features(seg_signal, dt)
            features_list.append(features)
            labels_list.append(name)

        # Обучаем классификатор
        n_clusters = min(self.classifier_panel.get_n_clusters(), len(references))
        self.classifier = SignalClassifier(n_clusters=n_clusters, max_iter=100)

        try:
            self.classifier.fit(features_list, labels_list)
        except Exception as e:
            self.classifier_panel.set_status(f"Ошибка обучения: {e}", is_error=True)
            return

        # Применяем к текущему сигналу
        overlap_percent = self.classifier_panel.get_overlap_percent()
        step = max(1, int(window_size * (1 - overlap_percent / 100)))

        boundaries = []
        labels = []
        confidences = []

        n = self.data.n
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            segment = self.data.combined[start:end]

            # Извлечение признаков
            features = extract_features(segment, dt)

            # Классификация
            label, confidence = self.classifier.predict_with_confidence(features)

            # ПРОВЕРКА: сумма вероятностей
            total = sum(confidence.values())
            if not np.isclose(total, 1.0):
                print(f"ВНИМАНИЕ: сегмент {len(boundaries)}, сумма = {total:.6f}")

            boundaries.append((start, end))
            labels.append(label)
            confidences.append(confidence)

        # Сохраняем результаты
        self.data.classification_boundaries = boundaries
        self.data.classification_labels = labels
        self.data.classification_confidences = confidences

        self.classifier_panel.set_status(
            f"✓ Классификация завершена! Найдено {len(boundaries)} сегментов.",
            is_error=False,
        )

        self._refresh_plots()

    def _on_filter(self):
        if self.data.n == 0:
            return
        fp = self._read_filter()
        noise_on = len(self.noise_filter_panel.get_noise_entries()) > 0
        self.data = apply_filter_to_data(
            self.data, fp, self.params.sample_rate, noise_on
        )
        self._refresh_plots()

    def _on_export_wav(self, file_path: str, signal_index: int):
        """Экспорт выбранного сигнала в WAV."""
        if self.data.n == 0:
            self.statistics_panel.set_export_status(
                "Нет данных для экспорта. Сначала сгенерируйте сигнал.", is_error=True
            )
            return

        signal_map = {
            0: self.data.base,
            1: self.data.combined,
            2: self.data.filtered,
        }
        signal_names = {
            0: "Базовый",
            1: "Результирующий",
            2: "Отфильтрованный",
        }

        signal = signal_map.get(signal_index, self.data.combined)
        name = signal_names.get(signal_index, "Сигнал")

        if len(signal) == 0 or (signal_index == 2 and not any(signal)):
            self.statistics_panel.set_export_status(
                f"Сигнал «{name}» пуст. Сначала сгенерируйте/отфильтруйте.",
                is_error=True,
            )
            return

        error = export_signal_to_wav(signal, self.params.sample_rate, file_path)

        if error:
            self.statistics_panel.set_export_status(error, is_error=True)
        else:
            duration = len(signal) / self.params.sample_rate
            self.statistics_panel.set_export_status(
                f"✓ «{name}» сохранён: {file_path}\n"
                f"  {len(signal)} сэмплов, {self.params.sample_rate:.0f} Гц, "
                f"{duration:.3f} с",
                is_error=False,
            )

    def _refresh_plots(self):
        show_filter = self.plot_panel.cb_filtered.isChecked()

        if self.data.segment_labels:
            label = "Сегментированный"
        else:
            entry = self.segmented_panel.get_first_enabled_entry()
            if entry is not None:
                type_idx = int(entry.signal_type)
                if 0 <= type_idx < len(SEGMENT_TYPE_NAMES):
                    label = SEGMENT_TYPE_NAMES[type_idx]
                else:
                    label = "Сигнал"
            else:
                label = "Сигнал"

        self.plot_panel.update_plots(
            self.params,
            self.data,
            show_filter,
            signal_label=label,
        )

        self.statistics_panel.update_statistics(self.data.statistics)
