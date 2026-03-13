from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from dsp import SignalType
from filters import FilterType

SIGNAL_NAMES = ["Гармонический", "Гауссов импульс", "Пилообразный"]

FILTER_NAMES = [
    "Хеннинга",
    "Параболический",
    "Первого порядка",
    "НЧФ",
    "ВЧФ",
    "Полосовой",
    "Режекторный",
]


def _add(layout: QVBoxLayout, label: str, widget):
    layout.addWidget(QLabel(label))
    layout.addWidget(widget)


class ParamPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Параметры", parent)

        layout = QVBoxLayout(self)

        self.ed_T = QLineEdit("1.0")
        self.ed_dt = QLineEdit("0.01")
        self.ed_N = QLineEdit("100")
        self.ed_fd = QLineEdit("100.0")

        _add(layout, "Время T, с:", self.ed_T)
        _add(layout, "Шаг ΔT, с:", self.ed_dt)
        _add(layout, "Кол-во точек N:", self.ed_N)
        _add(layout, "Частота дискретизации Fd, Гц:", self.ed_fd)

        self.signal_combo = QComboBox()
        self.signal_combo.addItems(SIGNAL_NAMES)
        _add(layout, "Тип сигнала:", self.signal_combo)

        self.harmonics_table = QTableWidget(1, 3)
        self.harmonics_table.setHorizontalHeaderLabels(
            ["Частота, Гц", "Амплитуда", "Фаза, °"]
        )
        _add(layout, "Компоненты сигнала:", self.harmonics_table)

        self.btn_model = QPushButton("Моделировать")
        layout.addWidget(self.btn_model)

        self.btn_clear = QPushButton("Очистить таблицу")
        layout.addWidget(self.btn_clear)

        self.btn_add_row = QPushButton("Добавить строку")
        layout.addWidget(self.btn_add_row)

        layout.addWidget(QLabel("— Помеха —"))
        self.cb_noise = QCheckBox("Добавить помеху")
        layout.addWidget(self.cb_noise)

        self.ed_amp_min = QLineEdit("0")
        self.ed_amp_max = QLineEdit("1")
        self.ed_noise_freq = QLineEdit("50")

        _add(layout, "Мин. амплитуда:", self.ed_amp_min)
        _add(layout, "Макс. амплитуда:", self.ed_amp_max)
        _add(layout, "Частота помехи, Гц:", self.ed_noise_freq)

        layout.addWidget(QLabel("— Фильтр —"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTER_NAMES)
        layout.addWidget(self.filter_combo)

        self.ed_fc = QLineEdit("10")
        self.ed_r = QLineEdit("0.9")

        _add(layout, "Частота среза Fc, Гц:", self.ed_fc)
        _add(layout, "Коэффициент r:", self.ed_r)

        self.btn_filter = QPushButton("Применить фильтр")
        layout.addWidget(self.btn_filter)

    def clear_table(self):
        self.harmonics_table.setRowCount(1)
        for col in range(3):
            self.harmonics_table.setItem(0, col, None)

    @property
    def selected_signal(self) -> SignalType:
        return SignalType(self.signal_combo.currentIndex())

    @property
    def selected_filter(self) -> FilterType:
        return FilterType(self.filter_combo.currentIndex())

    def add_row(self):
        current_rows = self.harmonics_table.rowCount()
        self.harmonics_table.insertRow(current_rows)
