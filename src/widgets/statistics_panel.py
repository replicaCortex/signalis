from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import SignalStatistics


def _fmt(value: float) -> str:
    """Форматирует число до 2 знаков после запятой."""
    return f"{value:.2f}"


class StatisticsPanel(QGroupBox):
    """Панель характеристик сигнала (левая вкладка)."""

    export_wav_requested = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__("Характеристики", parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        export_group = QGroupBox("Экспорт в WAV")
        export_layout = QVBoxLayout(export_group)

        export_desc = QLabel("Сохранить текущий сигнал в WAV файл (16-bit PCM, моно).")
        export_desc.setStyleSheet("color: #666; font-size: 10px;")
        export_desc.setWordWrap(True)
        export_layout.addWidget(export_desc)

        signal_row = QHBoxLayout()
        signal_row.addWidget(QLabel("Сигнал:"))
        self.combo_export_signal = QComboBox()
        self.combo_export_signal.addItems(
            [
                "Базовый",
                "Результирующий",
                "Отфильтрованный",
            ]
        )
        self.combo_export_signal.setCurrentIndex(1)
        signal_row.addWidget(self.combo_export_signal)
        export_layout.addLayout(signal_row)

        self.btn_export_wav = QPushButton("Экспорт в WAV")
        self.btn_export_wav.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        self.btn_export_wav.clicked.connect(self._on_export_clicked)
        export_layout.addWidget(self.btn_export_wav)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setStyleSheet("font-size: 10px;")
        self.lbl_export_status.setWordWrap(True)
        export_layout.addWidget(self.lbl_export_status)

        scroll_layout.addWidget(export_group)

        power_group = QGroupBox("Мощность сигнала")
        power_layout = QVBoxLayout(power_group)
        self.lbl_total_power = QLabel("Полная мощность: —")
        self.lbl_mean_power = QLabel("Средняя мощность: —")
        power_layout.addWidget(self.lbl_total_power)
        power_layout.addWidget(self.lbl_mean_power)
        scroll_layout.addWidget(power_group)

        amp_group = QGroupBox("Амплитуды")
        amp_layout = QVBoxLayout(amp_group)
        self.lbl_min_amp = QLabel("Минимальная амплитуда: —")
        self.lbl_max_amp = QLabel("Максимальная амплитуда: —")
        amp_layout.addWidget(self.lbl_min_amp)
        amp_layout.addWidget(self.lbl_max_amp)
        scroll_layout.addWidget(amp_group)

        signal_group = QGroupBox("Статистики сигнала")
        signal_layout = QVBoxLayout(signal_group)
        self.lbl_signal_mean = QLabel("Математическое ожидание: —")
        self.lbl_signal_var = QLabel("Дисперсия: —")
        self.lbl_signal_std = QLabel("СКО: —")
        signal_layout.addWidget(self.lbl_signal_mean)
        signal_layout.addWidget(self.lbl_signal_var)
        signal_layout.addWidget(self.lbl_signal_std)
        scroll_layout.addWidget(signal_group)

        acf_group = QGroupBox("Оценки АКФ")
        acf_layout = QVBoxLayout(acf_group)
        self.lbl_acf_mean = QLabel("Математическое ожидание: —")
        self.lbl_acf_var = QLabel("Дисперсия: —")
        self.lbl_acf_std = QLabel("СКО: —")
        acf_layout.addWidget(self.lbl_acf_mean)
        acf_layout.addWidget(self.lbl_acf_var)
        acf_layout.addWidget(self.lbl_acf_std)
        scroll_layout.addWidget(acf_group)

        psd_group = QGroupBox("Оценки СПМ")
        psd_layout = QVBoxLayout(psd_group)
        self.lbl_psd_mean = QLabel("Математическое ожидание: —")
        self.lbl_psd_var = QLabel("Дисперсия: —")
        self.lbl_psd_std = QLabel("СКО: —")
        psd_layout.addWidget(self.lbl_psd_mean)
        psd_layout.addWidget(self.lbl_psd_var)
        psd_layout.addWidget(self.lbl_psd_std)
        scroll_layout.addWidget(psd_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _on_export_clicked(self):
        """Открывает диалог сохранения и эмитирует сигнал экспорта."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить WAV файл",
            "signal.wav",
            "WAV файлы (*.wav);;Все файлы (*.*)",
        )
        if path:
            if not path.lower().endswith(".wav"):
                path += ".wav"
            signal_idx = self.combo_export_signal.currentIndex()
            self.export_wav_requested.emit(path, signal_idx)

    def set_export_status(self, message: str, is_error: bool = False):
        """Устанавливает статус экспорта."""
        color = "#D32F2F" if is_error else "#388E3C"
        self.lbl_export_status.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.lbl_export_status.setText(message)

    def update_statistics(self, stats: SignalStatistics):
        self.lbl_total_power.setText(f"Полная мощность: {_fmt(stats.total_power)}")
        self.lbl_mean_power.setText(f"Средняя мощность: {_fmt(stats.mean_power)}")

        self.lbl_min_amp.setText(f"Минимальная амплитуда: {_fmt(stats.min_amplitude)}")
        self.lbl_max_amp.setText(f"Максимальная амплитуда: {_fmt(stats.max_amplitude)}")

        self.lbl_signal_mean.setText(
            f"Математическое ожидание: {_fmt(stats.signal_mean)}"
        )
        self.lbl_signal_var.setText(f"Дисперсия: {_fmt(stats.signal_variance)}")
        self.lbl_signal_std.setText(f"СКО: {_fmt(stats.signal_std)}")

        self.lbl_acf_mean.setText(f"Математическое ожидание: {_fmt(stats.acf_mean)}")
        self.lbl_acf_var.setText(f"Дисперсия: {_fmt(stats.acf_variance)}")
        self.lbl_acf_std.setText(f"СКО: {_fmt(stats.acf_std)}")

        self.lbl_psd_mean.setText(f"Математическое ожидание: {_fmt(stats.psd_mean)}")
        self.lbl_psd_var.setText(f"Дисперсия: {_fmt(stats.psd_variance)}")
        self.lbl_psd_std.setText(f"СКО: {_fmt(stats.psd_std)}")
