from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import ColoredNoiseEntry
from dsp import NoiseType
from filters import FilterType

NOISE_TYPE_NAMES = [
    "Равномерный",
    "Экспоненциальный",
    "Белый",
    "Розовый",
    "Броуновский",
    "Синий",
    "Фиолетовый",
]

FILTER_NAMES = [
    "Хеннинга",
    "Параболический",
    "Первого порядка",
    "НЧФ",
    "ВЧФ",
    "Полосовой",
    "Режекторный",
]


class NoiseLayerWidget(QGroupBox):
    """Виджет одного слоя шума с настройками."""

    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, index: int = 0, parent=None):
        super().__init__(f"Шум #{index + 1}", parent)
        self._index = index
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Включение
        self.cb_enabled = QCheckBox("Включён")
        self.cb_enabled.setChecked(True)
        layout.addWidget(self.cb_enabled)

        # Тип шума
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Тип:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(NOISE_TYPE_NAMES)
        row1.addWidget(self.combo_type)
        layout.addLayout(row1)

        # Амплитуда
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Амплитуда:"))
        self.spin_amplitude = QDoubleSpinBox()
        self.spin_amplitude.setRange(0.001, 10000.0)
        self.spin_amplitude.setValue(1.0)
        self.spin_amplitude.setDecimals(3)
        self.spin_amplitude.setSingleStep(0.1)
        row2.addWidget(self.spin_amplitude)
        layout.addLayout(row2)

        # Кнопка удаления
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.btn_remove)

    def to_entry(self) -> ColoredNoiseEntry:
        return ColoredNoiseEntry(
            noise_type=NoiseType(self.combo_type.currentIndex()),
            amplitude=self.spin_amplitude.value(),
            enabled=self.cb_enabled.isChecked(),
        )

    def set_index(self, idx: int):
        self._index = idx
        self.setTitle(f"Шум #{idx + 1}")


class NoiseFilterPanel(QGroupBox):
    """Панель настройки цветных шумов и фильтров."""

    filter_requested = pyqtSignal()
    model_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Шумы и фильтры", parent)
        self._noise_layers: list[NoiseLayerWidget] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Секция шумов ──
        noise_group = QGroupBox("Шумы")
        noise_outer = QVBoxLayout(noise_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._noise_list_layout = QVBoxLayout(scroll_widget)
        self._noise_list_layout.addStretch()
        scroll.setWidget(scroll_widget)
        noise_outer.addWidget(scroll)

        btn_row = QHBoxLayout()
        self.btn_add_noise = QPushButton("+ Добавить шум")
        self.btn_add_noise.clicked.connect(self._add_noise_layer)
        btn_row.addWidget(self.btn_add_noise)

        self.btn_add_all = QPushButton("Добавить все типы")
        self.btn_add_all.clicked.connect(self._add_all_noise_types)
        btn_row.addWidget(self.btn_add_all)
        noise_outer.addLayout(btn_row)

        self.btn_clear_noise = QPushButton("Очистить все шумы")
        self.btn_clear_noise.clicked.connect(self._clear_all_noise)
        noise_outer.addWidget(self.btn_clear_noise)

        main_layout.addWidget(noise_group)

        # ── Кнопка моделировать ──
        self.btn_model = QPushButton("Моделировать")
        self.btn_model.clicked.connect(self.model_requested)
        main_layout.addWidget(self.btn_model)

        # ── Секция фильтров ──
        filter_group = QGroupBox("Фильтр")
        filter_layout = QVBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Тип фильтра:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTER_NAMES)
        filter_layout.addWidget(self.filter_combo)

        row_fc = QHBoxLayout()
        row_fc.addWidget(QLabel("Частота среза Fc, Гц:"))
        self.ed_fc = QLineEdit("10")
        row_fc.addWidget(self.ed_fc)
        filter_layout.addLayout(row_fc)

        row_r = QHBoxLayout()
        row_r.addWidget(QLabel("Коэффициент r:"))
        self.ed_r = QLineEdit("0.9")
        row_r.addWidget(self.ed_r)
        filter_layout.addLayout(row_r)

        self.btn_apply_filter = QPushButton("Применить фильтр")
        self.btn_apply_filter.clicked.connect(self.filter_requested)
        filter_layout.addWidget(self.btn_apply_filter)

        main_layout.addWidget(filter_group)

    def _add_noise_layer(self, noise_type_idx: int = 0) -> NoiseLayerWidget:
        idx = len(self._noise_layers)
        layer = NoiseLayerWidget(index=idx)
        layer.combo_type.setCurrentIndex(noise_type_idx)
        layer.removed.connect(self._remove_noise_layer)

        self._noise_layers.append(layer)
        self._noise_list_layout.insertWidget(self._noise_list_layout.count() - 1, layer)
        return layer

    def _remove_noise_layer(self, layer: NoiseLayerWidget):
        if layer in self._noise_layers:
            self._noise_layers.remove(layer)
            self._noise_list_layout.removeWidget(layer)
            layer.deleteLater()
            self._reindex()

    def _clear_all_noise(self):
        for layer in list(self._noise_layers):
            self._noise_list_layout.removeWidget(layer)
            layer.deleteLater()
        self._noise_layers.clear()

    def _add_all_noise_types(self):
        for i in range(len(NOISE_TYPE_NAMES)):
            self._add_noise_layer(noise_type_idx=i)

    def _reindex(self):
        for i, layer in enumerate(self._noise_layers):
            layer.set_index(i)

    def get_noise_entries(self) -> list[ColoredNoiseEntry]:
        return [layer.to_entry() for layer in self._noise_layers]

    @property
    def selected_filter(self) -> FilterType:
        return FilterType(self.filter_combo.currentIndex())

    def get_filter_cutoff(self) -> float:
        return float(self.ed_fc.text())

    def get_filter_r(self) -> float:
        return float(self.ed_r.text())
