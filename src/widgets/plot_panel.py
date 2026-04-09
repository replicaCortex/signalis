import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import SignalData, SignalParams


class PlotTab(QWidget):
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


class SpectrumTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Метод:"))
        self.combo = QComboBox()
        self.combo.addItems(["DFT", "FFT"])
        selector_layout.addWidget(self.combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()

    @property
    def is_fft(self) -> bool:
        return self.combo.currentIndex() == 1


class PhasePsdTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Отображение:"))
        self.combo = QComboBox()
        self.combo.addItems(["Фазовый спектр", "СПМ"])
        selector_layout.addWidget(self.combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()

    @property
    def is_psd(self) -> bool:
        return self.combo.currentIndex() == 1


SEGMENT_COLORS = {
    "Гарм": "#1f77b4",
    "Гаусс": "#ff7f0e",
    "Пила": "#2ca02c",
    "Импульс": "#d62728",
    "Эксп": "#9467bd",
    "Речь": "#8c564b",
    "АМ": "#e377c2",
}


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

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

        self.tabs = QTabWidget()

        self.tab_noise = PlotTab()
        self.tab_confidence = ConfidencePlotTab()
        self.tab_signals = PlotTab()
        self.tab_spectrum = SpectrumTab()
        self.tab_phase_psd = PhasePsdTab()
        self.tab_acf = PlotTab()
        self.tab_freq_resp = PlotTab()

        self.tabs.addTab(self.tab_signals, "Сигналы")
        self.tabs.addTab(self.tab_spectrum, "Амплитудный спектр")
        self.tabs.addTab(self.tab_phase_psd, "Фаза / СПМ")
        self.tabs.addTab(self.tab_acf, "АКФ")
        self.tabs.addTab(self.tab_freq_resp, "АЧХ фильтра")
        self.tabs.addTab(self.tab_noise, "Шум")
        self.tabs.addTab(self.tab_confidence, "Классификация")

        layout.addWidget(self.tabs)

        self.tab_spectrum.combo.currentIndexChanged.connect(self._on_spectrum_switch)
        self.tab_phase_psd.combo.currentIndexChanged.connect(self._on_phase_psd_switch)

        self._last_params = None
        self._last_data = None
        self._last_show_filter = False
        self._last_label = "Базовый"

    def _on_spectrum_switch(self):
        if self._last_data is not None:
            self._draw_spectrum()

    def _on_phase_psd_switch(self):
        if self._last_data is not None:
            self._draw_phase_psd()

    def _draw_spectrum(self):
        params = self._last_params
        data = self._last_data
        show_filter = self._last_show_filter

        n = data.n
        freq = np.arange(n) * params.df
        bar_w = 0.8 * params.df

        ax = self.tab_spectrum.ax
        ax.clear()

        if self.tab_spectrum.is_fft:
            if len(data.fft_mag) == n:
                half_n = n // 2
                ax.bar(freq[:half_n], data.fft_mag[:half_n], width=bar_w, label="FFT")
            ax.set(xlabel="Частота, Гц", ylabel="Амплитуда (FFT)")
        else:
            if len(data.spectrum_mag) == n:
                ax.bar(freq, data.spectrum_mag, width=bar_w, label="DFT")
            if show_filter and len(data.filtered_spectrum_mag) == n:
                ax.bar(
                    freq,
                    data.filtered_spectrum_mag,
                    width=bar_w * 0.5,
                    label="Отфильтрованный",
                )
            ax.set(xlabel="Частота, Гц", ylabel="Амплитуда (DFT)")

        ax.legend()
        ax.grid(True)
        self.tab_spectrum.refresh()

    def _draw_phase_psd(self):
        params = self._last_params
        data = self._last_data

        n = data.n
        freq = np.arange(n) * params.df
        bar_w = 0.8 * params.df

        ax = self.tab_phase_psd.ax
        ax.clear()

        if self.tab_phase_psd.is_psd:
            if len(data.psd) == n:
                half_n = n // 2
                ax.semilogy(freq[:half_n], data.psd[:half_n], label="СПМ")
            ax.set(xlabel="Частота, Гц", ylabel="СПМ (лог. шкала)")
        else:
            if data.has_noise and len(data.phase_psd) == n:
                ax.bar(freq, data.phase_psd, width=bar_w, label="СПМ фазы")
                ax.set(xlabel="Частота, Гц", ylabel="СПМ фазы")
            elif len(data.spectrum_phase) == n:
                ax.bar(freq, data.spectrum_phase, width=bar_w, label="Фаза")
                ax.set(xlabel="Частота, Гц", ylabel="Фаза, °")

        ax.legend()
        ax.grid(True)
        self.tab_phase_psd.refresh()

    def _draw_segment_boundaries(self, ax, data, params, x_axis):
        """Рисует границы сегментов с уникальными цветами для каждого варианта."""
        if not data.segment_boundaries:
            return

        drawn_labels = set()

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Создаем уникальную цветовую палитру для всех вариантов
        unique_labels = list(dict.fromkeys(data.segment_labels))  # сохраняем порядок
        n_unique = len(unique_labels)

        # Генерируем различимые цвета
        if n_unique <= 10:
            colormap = cm.get_cmap("tab10")
        elif n_unique <= 20:
            colormap = cm.get_cmap("tab20")
        else:
            colormap = cm.get_cmap("hsv")

        color_map = {}
        for i, label in enumerate(unique_labels):
            if n_unique <= 20:
                color_map[label] = colormap(i)
            else:
                color_map[label] = colormap(i / n_unique)

        for (start, end), label in zip(data.segment_boundaries, data.segment_labels):
            x_start = x_axis[start] if start < len(x_axis) else x_axis[-1]
            x_end = x_axis[min(end, len(x_axis) - 1)]
            color = color_map.get(label, "#999999")

            show_label = label not in drawn_labels
            ax.axvspan(
                x_start,
                x_end,
                alpha=0.2,  # Увеличена прозрачность для лучшей видимости
                color=color,
                label=f"{label}" if show_label else None,
            )
            drawn_labels.add(label)

            ax.axvline(x_start, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

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

        self._last_params = params
        self._last_data = data
        self._last_show_filter = show_filter
        self._last_label = signal_label

        freq = np.arange(n) * params.df
        bar_w = 0.8 * params.df

        # Определяем ось X для графика сигнала
        if data.x_axis_mode == "sigma" and data.gauss_sigma > 0:
            time = np.arange(n) * params.dt
            x_axis = (time - data.gauss_center) / data.gauss_sigma
            x_label = "σ (отклонение от мат. ожидания)"
        else:
            x_axis = np.arange(n) * params.dt
            x_label = "Время, с"

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
                ax.plot(x_axis, sig, label=label)

        self._draw_segment_boundaries(ax, data, params, x_axis)

        ax.set(xlabel=x_label, ylabel="Амплитуда")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True)
        self.tab_signals.refresh()

        self._draw_spectrum()
        self._draw_phase_psd()

        ax = self.tab_acf.ax
        ax.clear()
        if len(data.acf) == n:
            lags = np.arange(n) * params.dt
            ax.plot(lags, data.acf, label="АКФ")
        ax.set(xlabel="Лаг, с", ylabel="АКФ")
        ax.legend()
        ax.grid(True)
        self.tab_acf.refresh()

        ax = self.tab_freq_resp.ax
        ax.clear()
        if show_filter and len(data.freq_response) == n:
            ax.bar(freq, data.freq_response, width=bar_w)
            ax.set(xlabel="Частота, Гц", ylabel="Коэффициент передачи")
            ax.grid(True)
        self.tab_freq_resp.refresh()

        ax_noise = self.tab_noise.ax
        ax_noise.clear()
        if len(data.colored_noise) == data.n and np.any(data.colored_noise):
            ax_noise.plot(x_axis, data.colored_noise, label="Цветной шум", alpha=0.7)
            ax_noise.set(xlabel=x_label, ylabel="Амплитуда")
            ax_noise.legend()
            ax_noise.grid(True)

        if data.classification_confidences:
            self.tab_confidence.plot_confidence(
                data.classification_boundaries,
                data.classification_confidences,
                x_axis,
                params.dt,
            )
        else:
            self.tab_confidence.clear()
        self.tab_confidence.refresh()


class ConfidencePlotTab(QWidget):
    """Вкладка для отображения уверенности классификации."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)

        # Информация о формате
        info_label = QLabel(
            "График показывает уверенность классификатора для каждого сегмента.\n"
            "Высота столбца = вероятность (от 0 до 1, т.е. 0% до 100%)."
        )
        info_label.setStyleSheet("color: #666; font-size: 9px; padding: 3px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addWidget(self.canvas)

    def plot_confidence(self, boundaries, confidences, x_axis, dt):
        """Рисует график уверенности классификации."""
        self.ax.clear()

        if not confidences:
            self.ax.text(
                0.5,
                0.5,
                "Нет данных классификации",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self.canvas.draw()
            return

        # Собираем все уникальные метки
        all_labels = set()
        for conf_dict in confidences:
            all_labels.update(conf_dict.keys())
        all_labels = sorted(all_labels)

        # Создаем цветовую карту
        import matplotlib.cm as cm

        n_labels = len(all_labels)
        if n_labels <= 10:
            colormap = cm.get_cmap("tab10")
        elif n_labels <= 20:
            colormap = cm.get_cmap("tab20")
        else:
            colormap = cm.get_cmap("hsv")

        colors = {}
        for i, label in enumerate(all_labels):
            if n_labels <= 20:
                colors[label] = colormap(i)
            else:
                colors[label] = colormap(i / n_labels)

        # Для каждого сегмента рисуем столбчатую диаграмму уверенности
        n_segments = len(boundaries)

        for i, ((start, end), conf_dict) in enumerate(zip(boundaries, confidences)):
            center = (start + end) // 2
            time_center = center * dt

            # ПРОВЕРКА: сумма вероятностей должна быть 1.0
            total_prob = sum(conf_dict.values())
            if not np.isclose(total_prob, 1.0):
                print(f"ВНИМАНИЕ: сегмент {i}, сумма вероятностей = {total_prob:.4f}")

            # Накопленная высота для stacked bar
            bottom = 0.0
            for label in all_labels:
                prob = conf_dict.get(label, 0.0)
                if prob > 0.001:  # Показываем только значимые вероятности (>0.1%)
                    width = dt * (end - start) * 0.8
                    self.ax.bar(
                        time_center,
                        prob,
                        bottom=bottom,
                        color=colors[label],
                        width=width,
                        label=label if i == 0 else "",
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=0.5,
                    )

                    # Добавляем текст с процентами для значимых вероятностей (>10%)
                    if prob > 0.1:
                        text_y = bottom + prob / 2
                        self.ax.text(
                            time_center,
                            text_y,
                            f"{prob * 100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if prob > 0.3 else "black",
                            weight="bold",
                        )

                    bottom += prob

        self.ax.set_xlabel("Время, с")
        self.ax.set_ylabel("Вероятность")
        self.ax.set_title("Уверенность классификации по сегментам")
        self.ax.set_ylim(0, 1.0)  # ИСПРАВЛЕНО: максимум 1.0

        # Добавляем горизонтальные линии для удобства чтения
        for y in [0.25, 0.5, 0.75, 1.0]:
            self.ax.axhline(y, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

        # Форматируем ось Y в процентах
        from matplotlib.ticker import FuncFormatter

        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

        self.ax.grid(True, alpha=0.3, axis="x")

        # Легенда без дубликатов
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

        self.canvas.draw()

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()
