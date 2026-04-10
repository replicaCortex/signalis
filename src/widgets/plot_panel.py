import numpy as np
from catppuccin import PALETTE
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

MOCHA = PALETTE.mocha.colors

# Catppuccin Mocha цвета для matplotlib
DARK_BG = MOCHA.crust.hex
DARK_AXES = MOCHA.surface0.hex
DARK_TEXT = MOCHA.text.hex
DARK_GRID = MOCHA.surface1.hex
DARK_SUBTEXT = MOCHA.subtext0.hex


def _apply_dark_style(ax, title=None):
    """Применяет тёмную тему Catppuccin к оси matplotlib."""
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors=DARK_TEXT)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    if title:
        ax.set_title(title, color=DARK_TEXT)
    ax.grid(True, color=DARK_GRID, alpha=0.3)

    # Исправление: тёмный фон легенды
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(DARK_AXES)
        legend.get_frame().set_edgecolor(DARK_GRID)
        for text in legend.get_texts():
            text.set_color(DARK_TEXT)


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

    def refresh(self):
        self.canvas.draw()


class SpectrumTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

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
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

    def refresh(self):
        self.canvas.draw()

    @property
    def is_fft(self) -> bool:
        return self.combo.currentIndex() == 1


class PhasePsdTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

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
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

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
        self.cb_colored_noise = QCheckBox("Цветной шум")
        self.cb_combined = QCheckBox("Результирующий")
        self.cb_combined.setChecked(True)
        self.cb_filtered = QCheckBox("Отфильтрованный")
        self.cb_filtered.setChecked(True)

        cb_layout = QHBoxLayout()
        for cb in [
            self.cb_base,
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
        self.tab_classification = ClassificationTab()

        self.tabs.addTab(self.tab_signals, "Сигналы")
        self.tabs.addTab(self.tab_spectrum, "Амплитудный спектр")
        self.tabs.addTab(self.tab_phase_psd, "Фаза / СПМ")
        self.tabs.addTab(self.tab_acf, "АКФ")
        self.tabs.addTab(self.tab_freq_resp, "АЧХ фильтра")
        self.tabs.addTab(self.tab_noise, "Шум")
        self.tabs.addTab(self.tab_classification, "Классификация")

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
        _apply_dark_style(ax)
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
        _apply_dark_style(ax)
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

        # Вычисляем массив частот
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

        # Вкладка "Сигналы"
        ax = self.tab_signals.ax
        ax.clear()

        traces = [
            (self.cb_base, data.base, signal_label),
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
        _apply_dark_style(ax)
        self.tab_signals.refresh()

        # Вкладка "Амплитудный спектр"
        self._draw_spectrum()

        # Вкладка "Фаза / СПМ"
        self._draw_phase_psd()

        # Вкладка "АКФ"
        ax = self.tab_acf.ax
        ax.clear()
        if len(data.acf) == n:
            lags = np.arange(n) * params.dt
            ax.plot(lags, data.acf, label="АКФ")
        ax.set(xlabel="Лаг, с", ylabel="АКФ")
        ax.legend()
        _apply_dark_style(ax)
        self.tab_acf.refresh()

        # Вкладка "АЧХ фильтра" - ИСПРАВЛЕНО
        ax = self.tab_freq_resp.ax
        ax.clear()
        ax.set_facecolor(DARK_AXES)
        ax.tick_params(colors=DARK_TEXT)

        # Проверяем, есть ли данные для АЧХ
        if show_filter and len(data.freq_response) == n:
            # Отображаем только положительные частоты до частоты Найквиста
            half_n = n // 2
            ax.bar(
                freq[:half_n],
                data.freq_response[:half_n],
                width=bar_w,
                color=MOCHA.mauve.hex,
            )
            ax.set_xlabel("Частота, Гц", color=DARK_TEXT)
            ax.set_ylabel("Коэффициент передачи", color=DARK_TEXT)
            ax.set_title("Амплитудно-частотная характеристика фильтра", color=DARK_TEXT)
            ax.grid(True, alpha=0.3, color=DARK_GRID)
            # Устанавливаем пределы осей для лучшего отображения
            ax.set_ylim(0, 1.1)
            if half_n > 0:
                ax.set_xlim(0, freq[half_n - 1])
        else:
            # Если данных нет, показываем сообщение
            ax.text(
                0.5,
                0.5,
                "Нет данных АЧХ\n(примените фильтр)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=DARK_TEXT,
            )

        _apply_dark_style(ax)
        self.tab_freq_resp.refresh()

        # Вкладка "Шум"
        ax_noise = self.tab_noise.ax
        ax_noise.clear()
        if len(data.colored_noise) == data.n and np.any(data.colored_noise):
            ax_noise.plot(x_axis, data.colored_noise, label="Цветной шум", alpha=0.7)
            ax_noise.set(xlabel=x_label, ylabel="Амплитуда")
            ax_noise.legend()
            _apply_dark_style(ax_noise)
        else:
            ax_noise.text(
                0.5,
                0.5,
                "Нет данных шума",
                ha="center",
                va="center",
                transform=ax_noise.transAxes,
                color=DARK_TEXT,
            )
            _apply_dark_style(ax_noise)
        self.tab_noise.refresh()

        # Вкладка уверенности
        if data.classification_confidences:
            self.tab_confidence.plot_confidence(
                data.classification_boundaries,
                data.classification_confidences,
                x_axis,
                params.dt,
            )
        else:
            self.tab_confidence.clear()

        # Вкладка классификации
        if data.classification_confidences:
            self.tab_classification.plot_classification(
                data.combined,
                data.classification_boundaries,
                data.classification_labels,
                data.classification_confidences,
                x_axis,
                params.dt,
            )
        else:
            self.tab_classification.clear()
        self.tab_classification.refresh()


class ConfidencePlotTab(QWidget):
    """Вкладка для отображения уверенности классификации."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

        layout = QVBoxLayout(self)

        # Информация о формате
        info_label = QLabel(
            "График показывает уверенность классификатора для каждого сегмента.\n"
            "Высота столбца = вероятность (от 0 до 1, т.е. 0% до 100%)."
        )
        info_label.setStyleSheet(
            f"color: {DARK_SUBTEXT}; font-size: 9px; padding: 3px;"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addWidget(self.canvas)

    def plot_confidence(self, boundaries, confidences, x_axis, dt):
        """Рисует график уверенности классификации."""
        self.ax.clear()
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

        if not confidences:
            self.ax.text(
                0.5,
                0.5,
                "Нет данных классификации",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color=DARK_TEXT,
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

        self.ax.set_xlabel("Время, с", color=DARK_TEXT)
        self.ax.set_ylabel("Вероятность", color=DARK_TEXT)
        self.ax.set_title("Уверенность классификации по сегментам", color=DARK_TEXT)
        self.ax.set_ylim(0, 1.0)

        # Добавляем горизонтальные линии для удобства чтения
        for y in [0.25, 0.5, 0.75, 1.0]:
            self.ax.axhline(y, color=DARK_GRID, linestyle=":", linewidth=0.5, alpha=0.5)

        # Форматируем ось Y в процентах
        from matplotlib.ticker import FuncFormatter

        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

        self.ax.grid(True, alpha=0.3, axis="x", color=DARK_GRID)
        self.ax.tick_params(colors=DARK_TEXT)
        self.ax.set_facecolor(DARK_AXES)

        # Исправление: тёмная легенда
        legend = self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if legend:
            legend.get_frame().set_facecolor(DARK_AXES)
            legend.get_frame().set_edgecolor(DARK_GRID)
            for text in legend.get_texts():
                text.set_color(DARK_TEXT)

        self.canvas.draw()

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()


class ClassificationTab(QWidget):
    """Вкладка для отображения классификации с сигналом и уверенностью."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Создаем фигуру с двумя подграфиками
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)

        # Два графика: сигнал сверху, уверенность снизу
        self.ax_signal = self.figure.add_subplot(211)
        self.ax_signal.set_facecolor(DARK_AXES)
        self.ax_signal.tick_params(colors=DARK_TEXT)
        self.ax_confidence = self.figure.add_subplot(212)
        self.ax_confidence.set_facecolor(DARK_AXES)
        self.ax_confidence.tick_params(colors=DARK_TEXT)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def plot_classification(self, signal, boundaries, labels, confidences, x_axis, dt):
        """Рисует сигнал с классификацией и график уверенности."""
        self.ax_signal.clear()
        self.ax_confidence.clear()
        self.ax_signal.set_facecolor(DARK_AXES)
        self.ax_signal.tick_params(colors=DARK_TEXT)
        self.ax_confidence.set_facecolor(DARK_AXES)
        self.ax_confidence.tick_params(colors=DARK_TEXT)

        if not confidences:
            self.ax_signal.text(
                0.5,
                0.5,
                "Нет данных классификации",
                ha="center",
                va="center",
                transform=self.ax_signal.transAxes,
                color=DARK_TEXT,
            )
            self.ax_confidence.text(
                0.5,
                0.5,
                "Нет данных классификации",
                ha="center",
                va="center",
                transform=self.ax_confidence.transAxes,
                color=DARK_TEXT,
            )
            self.canvas.draw()
            return

        # === График 1: Сигнал с цветовой разметкой ===

        # Рисуем весь сигнал
        self.ax_signal.plot(
            x_axis, signal, color="gray", alpha=0.3, linewidth=0.5, label="Сигнал"
        )

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

        # Рисуем сегменты с цветами по предсказанной метке
        drawn_labels = set()
        for (start, end), label in zip(boundaries, labels):
            x_start = x_axis[start] if start < len(x_axis) else x_axis[-1]
            x_end = x_axis[min(end, len(x_axis) - 1)]

            color = colors.get(label, "#999999")

            # Закрашиваем фон
            show_label = label not in drawn_labels
            self.ax_signal.axvspan(
                x_start,
                x_end,
                alpha=0.15,
                color=color,
                label=label if show_label else None,
            )
            drawn_labels.add(label)

            # Рисуем сам сигнал в этом сегменте
            segment_indices = slice(start, min(end, len(signal)))
            self.ax_signal.plot(
                x_axis[segment_indices],
                signal[segment_indices],
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

        self.ax_signal.set_ylabel("Амплитуда", color=DARK_TEXT)
        self.ax_signal.set_title("Классифицированный сигнал", color=DARK_TEXT)

        # Исправление: тёмная легенда для ax_signal
        legend_signal = self.ax_signal.legend(
            loc="upper right", fontsize=8, framealpha=0.9
        )
        if legend_signal:
            legend_signal.get_frame().set_facecolor(DARK_AXES)
            legend_signal.get_frame().set_edgecolor(DARK_GRID)
            for text in legend_signal.get_texts():
                text.set_color(DARK_TEXT)

        self.ax_signal.grid(True, alpha=0.3, color=DARK_GRID)
        self.ax_signal.tick_params(colors=DARK_TEXT)
        self.ax_signal.set_facecolor(DARK_AXES)

        # === График 2: Уверенность классификации ===

        for i, ((start, end), conf_dict) in enumerate(zip(boundaries, confidences)):
            center = (start + end) // 2
            time_center = center * dt

            # Накопленная высота для stacked bar
            bottom = 0.0
            for label in all_labels:
                prob = conf_dict.get(label, 0.0)
                if prob > 0.001:  # Показываем только значимые вероятности (>0.1%)
                    width = dt * (end - start) * 0.8
                    self.ax_confidence.bar(
                        time_center,
                        prob,
                        bottom=bottom,
                        color=colors[label],
                        width=width,
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=0.5,
                        label=label if i == 0 else "",  # Добавляем label для легенды
                    )

                    # Добавляем текст с процентами для значимых вероятностей (>15%)
                    if prob > 0.15:
                        text_y = bottom + prob / 2
                        self.ax_confidence.text(
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

        self.ax_confidence.set_xlabel("Время, с", color=DARK_TEXT)
        self.ax_confidence.set_ylabel("Вероятность", color=DARK_TEXT)
        self.ax_confidence.set_title("Уверенность классификации", color=DARK_TEXT)
        self.ax_confidence.set_ylim(0, 1.0)

        # Добавляем горизонтальные линии для удобства чтения
        for y in [0.25, 0.5, 0.75, 1.0]:
            self.ax_confidence.axhline(
                y, color=DARK_GRID, linestyle=":", linewidth=0.5, alpha=0.5
            )

        # Форматируем ось Y в процентах
        from matplotlib.ticker import FuncFormatter

        self.ax_confidence.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
        )

        # Исправление: тёмная легенда для ax_confidence
        legend_conf = self.ax_confidence.legend(
            loc="upper right", fontsize=8, framealpha=0.9
        )
        if legend_conf:
            legend_conf.get_frame().set_facecolor(DARK_AXES)
            legend_conf.get_frame().set_edgecolor(DARK_GRID)
            for text in legend_conf.get_texts():
                text.set_color(DARK_TEXT)

        self.ax_confidence.grid(True, alpha=0.3, axis="x", color=DARK_GRID)
        self.ax_confidence.tick_params(colors=DARK_TEXT)
        self.ax_confidence.set_facecolor(DARK_AXES)

        self.canvas.draw()

    def clear(self):
        self.ax_signal.clear()
        self.ax_confidence.clear()

    def refresh(self):
        self.canvas.draw()


class ConfidencePlotTab(QWidget):
    """Вкладка для отображения уверенности классификации."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(tight_layout=True)
        self.figure.patch.set_facecolor(DARK_BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

        layout = QVBoxLayout(self)

        # Информация о формате
        info_label = QLabel(
            "График показывает уверенность классификатора для каждого сегмента.\n"
            "Высота столбца = вероятность (от 0 до 1, т.е. 0% до 100%)."
        )
        info_label.setStyleSheet(
            f"color: {DARK_SUBTEXT}; font-size: 9px; padding: 3px;"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addWidget(self.canvas)

    def plot_confidence(self, boundaries, confidences, x_axis, dt):
        """Рисует график уверенности классификации."""
        self.ax.clear()
        self.ax.set_facecolor(DARK_AXES)
        self.ax.tick_params(colors=DARK_TEXT)

        if not confidences:
            self.ax.text(
                0.5,
                0.5,
                "Нет данных классификации",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color=DARK_TEXT,
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

        self.ax.set_xlabel("Время, с", color=DARK_TEXT)
        self.ax.set_ylabel("Вероятность", color=DARK_TEXT)
        self.ax.set_title("Уверенность классификации по сегментам", color=DARK_TEXT)
        self.ax.set_ylim(0, 1.0)

        # Добавляем горизонтальные линии для удобства чтения
        for y in [0.25, 0.5, 0.75, 1.0]:
            self.ax.axhline(y, color=DARK_GRID, linestyle=":", linewidth=0.5, alpha=0.5)

        # Форматируем ось Y в процентах
        from matplotlib.ticker import FuncFormatter

        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

        self.ax.grid(True, alpha=0.3, axis="x", color=DARK_GRID)
        self.ax.tick_params(colors=DARK_TEXT)
        self.ax.set_facecolor(DARK_AXES)

        # Исправление: тёмная легенда
        legend = self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if legend:
            legend.get_frame().set_facecolor(DARK_AXES)
            legend.get_frame().set_edgecolor(DARK_GRID)
            for text in legend.get_texts():
                text.set_color(DARK_TEXT)

        self.canvas.draw()

    def clear(self):
        self.ax.clear()

    def refresh(self):
        self.canvas.draw()
