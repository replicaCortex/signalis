from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import SegmentPoolEntry
from classifier import ClassificationMethod
from dsp import SignalType

SEGMENT_TYPE_NAMES = [
    "Гармонический",
    "Гауссов импульс",
    "Пилообразный",
    "Импульсный",
    "Экспоненциальный",
    "Речевой сигнал",
    "Амплитудная модуляция",
]

SEGMENT_TYPE_MAP = {
    0: SignalType.HARMONIC,
    1: SignalType.GAUSSIAN,
    2: SignalType.SAWTOOTH,
    3: SignalType.IMPULSE,
    4: SignalType.EXPONENTIAL_IMPULSE,
    5: SignalType.SPEECH,
    6: SignalType.AM_MODULATED,
}


class ReferenceSignalWidget(QGroupBox):
    """Виджет одного эталонного сигнала для классификации."""

    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, index: int = 0, parent=None):
        super().__init__(f"Эталон #{index + 1}", parent)
        self._index = index
        self._build_ui()
        self._on_type_changed(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Включение
        self.cb_enabled = QCheckBox("Включён")
        self.cb_enabled.setChecked(True)
        self.cb_enabled.stateChanged.connect(self.changed)
        layout.addWidget(self.cb_enabled)

        # Название эталона
        # row_name = QHBoxLayout()
        # row_name.addWidget(QLabel("Название:"))
        # self.ed_name = QLineEdit()
        # self.ed_name.setPlaceholderText("Например: Гарм_5Гц")
        # self.ed_name.textChanged.connect(self._update_title)
        # row_name.addWidget(self.ed_name)
        # layout.addLayout(row_name)

        # Тип сигнала
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Тип:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(SEGMENT_TYPE_NAMES)
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        row_type.addWidget(self.combo_type)
        layout.addLayout(row_type)

        # === Параметры для гармонического ===
        self.harmonic_group = QWidget()
        hg_layout = QVBoxLayout(self.harmonic_group)
        hg_layout.setContentsMargins(0, 0, 0, 0)

        row_freq = QHBoxLayout()
        row_freq.addWidget(QLabel("Частота, Гц:"))
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.1, 50000.0)
        self.spin_freq.setValue(5.0)
        self.spin_freq.setDecimals(2)
        self.spin_freq.setSingleStep(1.0)
        row_freq.addWidget(self.spin_freq)
        hg_layout.addLayout(row_freq)

        row_amp = QHBoxLayout()
        row_amp.addWidget(QLabel("Амплитуда:"))
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.001, 10000.0)
        self.spin_amp.setValue(1.0)
        self.spin_amp.setDecimals(3)
        self.spin_amp.setSingleStep(0.1)
        row_amp.addWidget(self.spin_amp)
        hg_layout.addLayout(row_amp)

        row_phase = QHBoxLayout()
        row_phase.addWidget(QLabel("Фаза, °:"))
        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-360.0, 360.0)
        self.spin_phase.setValue(0.0)
        self.spin_phase.setDecimals(1)
        self.spin_phase.setSingleStep(15.0)
        row_phase.addWidget(self.spin_phase)
        hg_layout.addLayout(row_phase)

        layout.addWidget(self.harmonic_group)

        # === Параметры для пилообразного ===
        self.sawtooth_group = QWidget()
        sw_layout = QVBoxLayout(self.sawtooth_group)
        sw_layout.setContentsMargins(0, 0, 0, 0)

        row_saw_freq = QHBoxLayout()
        row_saw_freq.addWidget(QLabel("Частота, Гц:"))
        self.spin_saw_freq = QDoubleSpinBox()
        self.spin_saw_freq.setRange(0.1, 50000.0)
        self.spin_saw_freq.setValue(5.0)
        self.spin_saw_freq.setDecimals(2)
        row_saw_freq.addWidget(self.spin_saw_freq)
        sw_layout.addLayout(row_saw_freq)

        row_saw_amp = QHBoxLayout()
        row_saw_amp.addWidget(QLabel("Амплитуда:"))
        self.spin_saw_amp = QDoubleSpinBox()
        self.spin_saw_amp.setRange(0.001, 10000.0)
        self.spin_saw_amp.setValue(1.0)
        self.spin_saw_amp.setDecimals(3)
        row_saw_amp.addWidget(self.spin_saw_amp)
        sw_layout.addLayout(row_saw_amp)

        layout.addWidget(self.sawtooth_group)

        # === Параметры для гауссова импульса ===
        self.gauss_group = QWidget()
        gg_layout = QVBoxLayout(self.gauss_group)
        gg_layout.setContentsMargins(0, 0, 0, 0)

        row_gamp = QHBoxLayout()
        row_gamp.addWidget(QLabel("Амплитуда:"))
        self.spin_gauss_amp = QDoubleSpinBox()
        self.spin_gauss_amp.setRange(0.001, 10000.0)
        self.spin_gauss_amp.setValue(1.0)
        self.spin_gauss_amp.setDecimals(3)
        row_gamp.addWidget(self.spin_gauss_amp)
        gg_layout.addLayout(row_gamp)

        row_sigma = QHBoxLayout()
        row_sigma.addWidget(QLabel("Сигма (σ), с:"))
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.001, 100.0)
        self.spin_gauss_sigma.setValue(0.05)
        self.spin_gauss_sigma.setDecimals(4)
        row_sigma.addWidget(self.spin_gauss_sigma)
        gg_layout.addLayout(row_sigma)

        layout.addWidget(self.gauss_group)

        # === Параметры для импульсного ===
        self.impulse_group = QWidget()
        ig_layout = QVBoxLayout(self.impulse_group)
        ig_layout.setContentsMargins(0, 0, 0, 0)

        row_iw = QHBoxLayout()
        row_iw.addWidget(QLabel("Длительность, с:"))
        self.spin_imp_width = QDoubleSpinBox()
        self.spin_imp_width.setRange(0.001, 100.0)
        self.spin_imp_width.setValue(0.02)
        self.spin_imp_width.setDecimals(4)
        row_iw.addWidget(self.spin_imp_width)
        ig_layout.addLayout(row_iw)

        row_ia = QHBoxLayout()
        row_ia.addWidget(QLabel("Амплитуда:"))
        self.spin_imp_amp = QDoubleSpinBox()
        self.spin_imp_amp.setRange(0.001, 10000.0)
        self.spin_imp_amp.setValue(1.0)
        self.spin_imp_amp.setDecimals(3)
        row_ia.addWidget(self.spin_imp_amp)
        ig_layout.addLayout(row_ia)

        row_ip = QHBoxLayout()
        row_ip.addWidget(QLabel("Период повт., с:"))
        self.spin_imp_period = QDoubleSpinBox()
        self.spin_imp_period.setRange(0.0, 100.0)
        self.spin_imp_period.setValue(0.1)
        self.spin_imp_period.setDecimals(4)
        row_ip.addWidget(self.spin_imp_period)
        ig_layout.addLayout(row_ip)

        layout.addWidget(self.impulse_group)

        # === Параметры для экспоненциального ===
        self.exp_group = QWidget()
        eg_layout = QVBoxLayout(self.exp_group)
        eg_layout.setContentsMargins(0, 0, 0, 0)

        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Коэф. α:"))
        self.spin_exp_alpha = QDoubleSpinBox()
        self.spin_exp_alpha.setRange(-1000.0, 1000.0)
        self.spin_exp_alpha.setValue(-5.0)
        self.spin_exp_alpha.setDecimals(2)
        row_alpha.addWidget(self.spin_exp_alpha)
        eg_layout.addLayout(row_alpha)

        row_ea = QHBoxLayout()
        row_ea.addWidget(QLabel("Амплитуда:"))
        self.spin_exp_amp = QDoubleSpinBox()
        self.spin_exp_amp.setRange(0.001, 10000.0)
        self.spin_exp_amp.setValue(1.0)
        self.spin_exp_amp.setDecimals(3)
        row_ea.addWidget(self.spin_exp_amp)
        eg_layout.addLayout(row_ea)

        layout.addWidget(self.exp_group)

        # === Параметры для речевого сигнала ===
        self.speech_group = QWidget()
        sg_layout = QVBoxLayout(self.speech_group)
        sg_layout.setContentsMargins(0, 0, 0, 0)

        row_file = QHBoxLayout()
        row_file.addWidget(QLabel("WAV файл:"))
        self.ed_speech_file = QLineEdit()
        self.ed_speech_file.setPlaceholderText("Выберите файл...")
        self.ed_speech_file.setReadOnly(True)
        row_file.addWidget(self.ed_speech_file)
        self.btn_browse = QPushButton("📂")
        self.btn_browse.setFixedWidth(36)
        self.btn_browse.clicked.connect(self._browse_wav)
        row_file.addWidget(self.btn_browse)
        sg_layout.addLayout(row_file)

        row_sa = QHBoxLayout()
        row_sa.addWidget(QLabel("Амплитуда:"))
        self.spin_speech_amp = QDoubleSpinBox()
        self.spin_speech_amp.setRange(0.001, 10000.0)
        self.spin_speech_amp.setValue(1.0)
        self.spin_speech_amp.setDecimals(3)
        row_sa.addWidget(self.spin_speech_amp)
        sg_layout.addLayout(row_sa)

        layout.addWidget(self.speech_group)

        # === Параметры для АМ ===
        self.am_group = QWidget()
        am_layout = QVBoxLayout(self.am_group)
        am_layout.setContentsMargins(0, 0, 0, 0)

        row_am_cf = QHBoxLayout()
        row_am_cf.addWidget(QLabel("Несущая частота, Гц:"))
        self.spin_am_carrier_freq = QDoubleSpinBox()
        self.spin_am_carrier_freq.setRange(0.1, 50000.0)
        self.spin_am_carrier_freq.setValue(50.0)
        self.spin_am_carrier_freq.setDecimals(2)
        row_am_cf.addWidget(self.spin_am_carrier_freq)
        am_layout.addLayout(row_am_cf)

        row_am_ca = QHBoxLayout()
        row_am_ca.addWidget(QLabel("Амплитуда несущей:"))
        self.spin_am_carrier_amp = QDoubleSpinBox()
        self.spin_am_carrier_amp.setRange(0.001, 10000.0)
        self.spin_am_carrier_amp.setValue(1.0)
        self.spin_am_carrier_amp.setDecimals(3)
        row_am_ca.addWidget(self.spin_am_carrier_amp)
        am_layout.addLayout(row_am_ca)

        row_am_mf = QHBoxLayout()
        row_am_mf.addWidget(QLabel("Частота модуляции, Гц:"))
        self.spin_am_mod_freq = QDoubleSpinBox()
        self.spin_am_mod_freq.setRange(0.1, 1000.0)
        self.spin_am_mod_freq.setValue(5.0)
        self.spin_am_mod_freq.setDecimals(2)
        row_am_mf.addWidget(self.spin_am_mod_freq)
        am_layout.addLayout(row_am_mf)

        row_am_md = QHBoxLayout()
        row_am_md.addWidget(QLabel("Глубина модуляции:"))
        self.spin_am_mod_depth = QDoubleSpinBox()
        self.spin_am_mod_depth.setRange(0.0, 1.0)
        self.spin_am_mod_depth.setValue(0.8)
        self.spin_am_mod_depth.setDecimals(2)
        row_am_md.addWidget(self.spin_am_mod_depth)
        am_layout.addLayout(row_am_md)

        layout.addWidget(self.am_group)

        # Кнопка удаления
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.btn_remove)

    def _browse_wav(self):
        """Открывает диалог выбора WAV файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите WAV файл", "", "WAV файлы (*.wav);;Все файлы (*.*)"
        )
        if path:
            self.ed_speech_file.setText(path)
            self.changed.emit()

    def _on_type_changed(self, index: int):
        """Показываем/скрываем группы параметров."""
        sig_type = SEGMENT_TYPE_MAP.get(index, SignalType.HARMONIC)

        self.harmonic_group.setVisible(sig_type == SignalType.HARMONIC)
        self.sawtooth_group.setVisible(sig_type == SignalType.SAWTOOTH)
        self.gauss_group.setVisible(sig_type == SignalType.GAUSSIAN)
        self.impulse_group.setVisible(sig_type == SignalType.IMPULSE)
        self.exp_group.setVisible(sig_type == SignalType.EXPONENTIAL_IMPULSE)
        self.speech_group.setVisible(sig_type == SignalType.SPEECH)
        self.am_group.setVisible(sig_type == SignalType.AM_MODULATED)

        self.changed.emit()

    def _update_title(self):
        """Обновляет заголовок группы при изменении названия."""
        name = self.ed_name.text().strip()
        if name:
            self.setTitle(f"Эталон #{self._index + 1}: {name}")
        else:
            self.setTitle(f"Эталон #{self._index + 1}")

    def get_name(self) -> str:
        """Возвращает название эталона."""
        # Генерируем автоматическое название
        idx = self.combo_type.currentIndex()
        sig_type = SEGMENT_TYPE_MAP.get(idx, SignalType.HARMONIC)

        if sig_type == SignalType.HARMONIC:
            name = f"Гарм_{self.spin_freq.value():.1f}Гц"
        elif sig_type == SignalType.SAWTOOTH:
            name = f"Пила_{self.spin_saw_freq.value():.1f}Гц"
        elif sig_type == SignalType.AM_MODULATED:
            name = f"АМ_{self.spin_am_carrier_freq.value():.0f}/{self.spin_am_mod_freq.value():.1f}Гц"
        else:
            type_names = {
                SignalType.GAUSSIAN: "Гаусс",
                SignalType.IMPULSE: "Импульс",
                SignalType.EXPONENTIAL_IMPULSE: "Эксп",
                SignalType.SPEECH: "Речь",
            }
            name = type_names.get(sig_type, "Сигнал")

        return name

    def _update_title(self):
        """Обновляет заголовок группы."""
        # Просто обновляем заголовок с индексом
        self.setTitle(f"Эталон #{self._index + 1}")

    def _on_type_changed(self, index: int):
        """Показываем/скрываем группы параметров."""
        sig_type = SEGMENT_TYPE_MAP.get(index, SignalType.HARMONIC)

        self.harmonic_group.setVisible(sig_type == SignalType.HARMONIC)
        self.sawtooth_group.setVisible(sig_type == SignalType.SAWTOOTH)
        self.gauss_group.setVisible(sig_type == SignalType.GAUSSIAN)
        self.impulse_group.setVisible(sig_type == SignalType.IMPULSE)
        self.exp_group.setVisible(sig_type == SignalType.EXPONENTIAL_IMPULSE)
        self.speech_group.setVisible(sig_type == SignalType.SPEECH)
        self.am_group.setVisible(sig_type == SignalType.AM_MODULATED)

        self._update_title()  # Обновляем заголовок при смене типа
        self.changed.emit()

    def set_index(self, idx: int):
        self._index = idx
        self._update_title()

    def to_entry(self) -> SegmentPoolEntry:
        """Преобразует виджет в SegmentPoolEntry."""
        idx = self.combo_type.currentIndex()
        sig_type = SEGMENT_TYPE_MAP.get(idx, SignalType.HARMONIC)

        if sig_type == SignalType.SAWTOOTH:
            freq = self.spin_saw_freq.value()
            amp = self.spin_saw_amp.value()
            saw_phase = 0.0
        else:
            freq = self.spin_freq.value()
            amp = self.spin_amp.value()
            saw_phase = 0.0

        return SegmentPoolEntry(
            signal_type=sig_type,
            enabled=self.cb_enabled.isChecked(),
            freq=freq,
            amp=amp,
            phase_deg=self.spin_phase.value(),
            saw_phase_deg=saw_phase,
            gauss_amp=self.spin_gauss_amp.value(),
            gauss_sigma=self.spin_gauss_sigma.value(),
            gauss_center=0.1,  # Фиксированный центр для эталона
            impulse_width=self.spin_imp_width.value(),
            impulse_amp=self.spin_imp_amp.value(),
            impulse_period=self.spin_imp_period.value(),
            impulse_delay=0.0,
            exp_alpha=self.spin_exp_alpha.value(),
            exp_amp=self.spin_exp_amp.value(),
            exp_delay=0.0,
            speech_file=self.ed_speech_file.text(),
            speech_amp=self.spin_speech_amp.value(),
            am_carrier_freq=self.spin_am_carrier_freq.value(),
            am_carrier_amp=self.spin_am_carrier_amp.value(),
            am_mod_freq=self.spin_am_mod_freq.value(),
            am_mod_depth=self.spin_am_mod_depth.value(),
        )

    def set_index(self, idx: int):
        self._index = idx
        self._update_title()


class ClassifierPanel(QGroupBox):
    """Панель классификации сигналов."""

    classify_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Классификация сигналов", parent)
        self._references: list[ReferenceSignalWidget] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Настройки окна классификации
        settings_group = QGroupBox("Параметры окна")
        settings_layout = QVBoxLayout(settings_group)

        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Метод классификации:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(
            [
                "K-means кластеризация",
                "K-ближайших соседей (KNN)",
                "Евклидово расстояние",
                "Корреляция признаков",
                "Dynamic Time Warping (DTW)",
            ]
        )
        self.combo_method.setToolTip(
            "K-means: кластеризация в пространстве признаков\n"
            "KNN: голосование K ближайших эталонов\n"
            "Евклидово: расстояние до ближайшего эталона\n"
            "Корреляция: схожесть векторов признаков\n"
            "DTW: сравнение формы сигналов во времени"
        )
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        row_method.addWidget(self.combo_method)
        settings_layout.addLayout(row_method)

        # Размер окна
        row_window = QHBoxLayout()
        row_window.addWidget(QLabel("Размер окна, с:"))
        self.spin_window_size = QDoubleSpinBox()
        self.spin_window_size.setRange(0.01, 10.0)
        self.spin_window_size.setValue(0.2)
        self.spin_window_size.setDecimals(3)
        self.spin_window_size.setSingleStep(0.05)
        self.spin_window_size.setToolTip("Размер скользящего окна для анализа")
        row_window.addWidget(self.spin_window_size)
        settings_layout.addLayout(row_window)

        # Перекрытие
        row_overlap = QHBoxLayout()
        row_overlap.addWidget(QLabel("Перекрытие, %:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 90)
        self.spin_overlap.setValue(50)
        self.spin_overlap.setSuffix("%")
        self.spin_overlap.setToolTip("Процент перекрытия окон")
        row_overlap.addWidget(self.spin_overlap)
        settings_layout.addLayout(row_overlap)

        # Кластеры (для K-means)
        row_clusters = QHBoxLayout()
        self.lbl_clusters = QLabel("Число кластеров:")
        row_clusters.addWidget(self.lbl_clusters)
        self.spin_n_clusters = QSpinBox()
        self.spin_n_clusters.setRange(2, 20)
        self.spin_n_clusters.setValue(5)
        self.spin_n_clusters.setToolTip("Количество кластеров для K-means")
        row_clusters.addWidget(self.spin_n_clusters)
        settings_layout.addLayout(row_clusters)

        # K соседей (для KNN)
        row_k = QHBoxLayout()
        self.lbl_k_neighbors = QLabel("Число соседей K:")
        row_k.addWidget(self.lbl_k_neighbors)
        self.spin_k_neighbors = QSpinBox()
        self.spin_k_neighbors.setRange(1, 20)
        self.spin_k_neighbors.setValue(3)
        self.spin_k_neighbors.setToolTip("Количество соседей для KNN")
        row_k.addWidget(self.spin_k_neighbors)
        settings_layout.addLayout(row_k)

        main_layout.addWidget(settings_group)

        # Пул эталонных сигналов
        pool_group = QGroupBox("Эталонные сигналы")
        pool_outer = QVBoxLayout(pool_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._pool_layout = QVBoxLayout(scroll_widget)
        self._pool_layout.addStretch()
        scroll.setWidget(scroll_widget)
        pool_outer.addWidget(scroll)

        # Кнопки управления пулом
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Добавить эталон")
        self.btn_add.clicked.connect(self._add_reference)
        btn_row.addWidget(self.btn_add)

        self.btn_add_all = QPushButton("Добавить все типы")
        self.btn_add_all.clicked.connect(self._add_all_types)
        btn_row.addWidget(self.btn_add_all)
        pool_outer.addLayout(btn_row)

        self.btn_clear = QPushButton("Очистить пул")
        self.btn_clear.clicked.connect(self._clear_all)
        pool_outer.addWidget(self.btn_clear)

        main_layout.addWidget(pool_group)

        # Кнопка классификации
        self.btn_classify = QPushButton("Определить сигнал")
        self.btn_classify.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px; font-size: 14px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_classify.clicked.connect(self.classify_requested)
        main_layout.addWidget(self.btn_classify)

        # Статус
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px;")
        self.lbl_status.setWordWrap(True)
        main_layout.addWidget(self.lbl_status)

    def _add_reference(self, type_idx: int = 0) -> ReferenceSignalWidget:
        idx = len(self._references)
        ref = ReferenceSignalWidget(index=idx)
        ref.combo_type.setCurrentIndex(type_idx)
        ref.removed.connect(self._remove_reference)

        self._references.append(ref)
        self._pool_layout.insertWidget(self._pool_layout.count() - 1, ref)
        return ref

    def get_classification_method(self) -> ClassificationMethod:
        return ClassificationMethod(self.combo_method.currentIndex())

    def get_k_neighbors(self) -> int:
        return self.spin_k_neighbors.value()

    def _on_method_changed(self, index: int):
        """Показывает/скрывает параметры в зависимости от метода."""
        # K-means нужно число кластеров
        is_kmeans = index == 0
        self.lbl_clusters.setVisible(is_kmeans)
        self.spin_n_clusters.setVisible(is_kmeans)

        # KNN нужно число соседей
        is_knn = index == 1
        self.lbl_k_neighbors.setVisible(is_knn)
        self.spin_k_neighbors.setVisible(is_knn)

    def _remove_reference(self, ref: ReferenceSignalWidget):
        if ref in self._references:
            self._references.remove(ref)
            self._pool_layout.removeWidget(ref)
            ref.deleteLater()
            self._reindex()

    def _clear_all(self):
        for ref in list(self._references):
            self._pool_layout.removeWidget(ref)
            ref.deleteLater()
        self._references.clear()

    def _add_all_types(self):
        for i in range(len(SEGMENT_TYPE_NAMES)):
            # Проверяем, что это не речевой сигнал
            sig_type = SEGMENT_TYPE_MAP.get(i)
            if sig_type != SignalType.SPEECH:
                self._add_reference(type_idx=i)

    def _reindex(self):
        for i, ref in enumerate(self._references):
            ref.set_index(i)

    # Публичный API

    def get_references(self) -> list[tuple[str, SegmentPoolEntry]]:
        """Возвращает список (название, entry) для всех включённых эталонов."""
        result = []
        for ref in self._references:
            entry = ref.to_entry()
            if entry.enabled:
                name = ref.get_name()
                result.append((name, entry))
        return result

    def get_window_size(self) -> float:
        return self.spin_window_size.value()

    def get_overlap_percent(self) -> int:
        return self.spin_overlap.value()

    def get_n_clusters(self) -> int:
        return self.spin_n_clusters.value()

    def set_status(self, message: str, is_error: bool = False):
        color = "#D32F2F" if is_error else "#388E3C"
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.lbl_status.setText(message)
