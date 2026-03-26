# src/widgets/param_panel.py
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from dsp import SignalType

SIGNAL_NAMES = [
    "Гармонический",
    "Гауссов импульс",
    "Пилообразный",
    "Импульсный",
    "Экспоненциальный импульс",
]

# Заголовки столбцов таблицы для каждого типа сигнала
SIGNAL_TABLE_HEADERS = {
    SignalType.HARMONIC: ["Частота, Гц", "Амплитуда", "Фаза, °"],
    SignalType.GAUSSIAN: ["Центр, с", "Амплитуда", "Сигма, с"],
    SignalType.SAWTOOTH: ["Частота, Гц", "Амплитуда", "-"],
    SignalType.IMPULSE: ["Длительность, с", "Амплитуда", "Период повт., с"],
    SignalType.EXPONENTIAL_IMPULSE: ["Коэф. затухания α", "Амплитуда", "Задержка, с"],
}


def _add(layout: QVBoxLayout, label: str, widget):
    layout.addWidget(QLabel(label))
    layout.addWidget(widget)


class ParamPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Параметры сигнала", parent)

        layout = QVBoxLayout(self)

        self.signal_combo = QComboBox()
        self.signal_combo.addItems(SIGNAL_NAMES)
        self.signal_combo.currentIndexChanged.connect(self._on_signal_type_changed)
        _add(layout, "Тип сигнала:", self.signal_combo)

        self.harmonics_table = QTableWidget(1, 3)
        self.harmonics_table.setHorizontalHeaderLabels(
            SIGNAL_TABLE_HEADERS[SignalType.HARMONIC]
        )
        _add(layout, "Компоненты сигнала:", self.harmonics_table)

        self.btn_model = QPushButton("Моделировать")
        layout.addWidget(self.btn_model)

        self.btn_clear = QPushButton("Очистить таблицу")
        layout.addWidget(self.btn_clear)

        self.btn_add_row = QPushButton("Добавить строку")
        layout.addWidget(self.btn_add_row)

    def _on_signal_type_changed(self, index: int):
        """Обновляет заголовки таблицы при смене типа сигнала."""
        sig_type = SignalType(index)
        headers = SIGNAL_TABLE_HEADERS.get(
            sig_type, ["Частота, Гц", "Амплитуда", "Фаза, °"]
        )
        self.harmonics_table.setHorizontalHeaderLabels(headers)

    def clear_table(self):
        self.harmonics_table.setRowCount(1)
        for col in range(3):
            self.harmonics_table.setItem(0, col, None)

    @property
    def selected_signal(self) -> SignalType:
        return SignalType(self.signal_combo.currentIndex())

    def add_row(self):
        current_rows = self.harmonics_table.rowCount()
        self.harmonics_table.insertRow(current_rows)
