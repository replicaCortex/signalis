import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import SignalData, SignalParams


class PlotTab(QWidget):
    """Одна вкладка с графиком."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()


class PlotPanel(QWidget):
    """4 вкладки графиков + чекбоксы."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Чекбоксы
        self.cb_base = QCheckBox("Базовый сигнал")
        self.cb_base.setChecked(True)
        self.cb_noise = QCheckBox("Помеха")
        self.cb_colored_noise = QCheckBox("Цветной шум")
        self.cb_combined = QCheckBox("Результирующий")
        self.cb_combined.setChecked(True)
        self.cb_filtered = QCheckBox("Отфильтрованный")

        cb_layout = QHBoxLayout()
        for cb in [
            self.cb_base,
            self.cb_noise,
            self.cb_colored_noise,
            self.cb_combined,
            self.cb_filtered,
        ]:
            cb_layout.addWidget(cb)
        layout.addLayout(cb_layout)

        # Вкладки
        self.tabs = QTabWidget()

        self.tab_signals = PlotTab()
        self.tab_amplitude = PlotTab()
        self.tab_phase = PlotTab()
        self.tab_freq_resp = PlotTab()

        self.tabs.addTab(self.tab_signals, "Сигналы")
        self.tabs.addTab(self.tab_amplitude, "Амплитудный спектр")
        self.tabs.addTab(self.tab_phase, "Фазовый спектр")
        self.tabs.addTab(self.tab_freq_resp, "АЧХ фильтра")

        layout.addWidget(self.tabs)

    def update_plots(
        self,
        params: SignalParams,
        data: SignalData,
        show_filter: bool,
        signal_label: str = "Базовый",
    ):
        n = data.n
        if n == 0:
            return

        time = np.arange(n) * params.dt
        freq = np.arange(n) * params.df
        bar_w = 0.8 * params.df

        # --- Сигналы ---
        ax = self.tab_signals.ax
        ax.clear()

        traces = [
            (self.cb_base, data.base, signal_label),
            (self.cb_noise, data.noise, "Помеха"),
            (self.cb_colored_noise, data.colored_noise, "Цветной шум"),
            (self.cb_combined, data.combined, "Результирующий"),
            (self.cb_filtered, data.filtered, "Отфильтрованный"),
        ]
        for cb, sig, label in traces:
            if cb.isChecked() and len(sig) == n:
                ax.plot(time, sig, label=label)

        ax.set(xlabel="Время, с", ylabel="Амплитуда")
        ax.legend()
        ax.grid(True)
        self.tab_signals.refresh()

        # --- Амплитудный спектр ---
        ax = self.tab_amplitude.ax
        ax.clear()
        if len(data.spectrum_mag) == n:
            ax.bar(freq, data.spectrum_mag, width=bar_w, label="Исходный")
        if show_filter and len(data.filtered_spectrum_mag) == n:
            ax.bar(
                freq,
                data.filtered_spectrum_mag,
                width=bar_w * 0.5,
                label="Отфильтрованный",
            )
        ax.set(xlabel="Частота, Гц", ylabel="Амплитуда")
        ax.legend()
        ax.grid(True)
        self.tab_amplitude.refresh()

        # --- Фазовый спектр ---
        ax = self.tab_phase.ax
        ax.clear()
        if len(data.spectrum_phase) == n:
            ax.bar(freq, data.spectrum_phase, width=bar_w)
        ax.set(xlabel="Частота, Гц", ylabel="Фаза, °")
        ax.grid(True)
        self.tab_phase.refresh()

        # --- АЧХ ---
        ax = self.tab_freq_resp.ax
        ax.clear()
        if show_filter and len(data.freq_response) == n:
            ax.bar(freq, data.freq_response, width=bar_w)
            ax.set(xlabel="Частота, Гц", ylabel="Коэффициент передачи")
            ax.grid(True)
        self.tab_freq_resp.refresh()
