# src/main_window.py
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QTabWidget, QWidget

from app import (
    ExponentialComponent,
    FilterParams,
    GaussianComponent,
    Harmonic,
    ImpulseComponent,
    NoiseParams,
    SignalData,
    SignalParams,
    apply_filter_to_data,
    generate_segmented_signal,
    generate_signal,
)
from dsp import SignalType
from widgets.noise_filter_panel import NoiseFilterPanel
from widgets.param_panel import SIGNAL_NAMES, ParamPanel
from widgets.plot_panel import PlotPanel
from widgets.segmented_panel import SegmentedPanel
from widgets.simulation_panel import SimulationPanel
from widgets.statistics_panel import StatisticsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FooBarPrime")
        self.setGeometry(100, 100, 1400, 800)

        self.data = SignalData()
        self.params = SignalParams()

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Левая часть: вкладки
        self.left_tabs = QTabWidget()

        self.simulation_panel = SimulationPanel()
        self.param_panel = ParamPanel()
        self.noise_filter_panel = NoiseFilterPanel()
        self.segmented_panel = SegmentedPanel()
        self.statistics_panel = StatisticsPanel()

        self.left_tabs.addTab(self.simulation_panel, "Параметры симуляции")
        self.left_tabs.addTab(self.param_panel, "Сигнал")
        self.left_tabs.addTab(self.noise_filter_panel, "Шумы и фильтры")
        self.left_tabs.addTab(self.segmented_panel, "Сегментированный")
        self.left_tabs.addTab(self.statistics_panel, "Характеристики")

        # Правая часть: графики
        self.plot_panel = PlotPanel()

        layout.addWidget(self.left_tabs, stretch=1)
        layout.addWidget(self.plot_panel, stretch=3)

    def _connect_signals(self):
        pp = self.param_panel
        pp.btn_model.clicked.connect(self._on_model)
        pp.btn_clear.clicked.connect(pp.clear_table)
        pp.btn_add_row.clicked.connect(pp.add_row)

        # Панель шумов/фильтров
        nfp = self.noise_filter_panel
        nfp.filter_requested.connect(self._on_filter_from_noise_panel)
        nfp.model_requested.connect(self._on_model)

        # Панель сегментированного сигнала
        self.segmented_panel.generate_requested.connect(self._on_segmented_generate)

        # Чекбоксы отображения
        for cb in [
            self.plot_panel.cb_base,
            self.plot_panel.cb_noise,
            self.plot_panel.cb_colored_noise,
            self.plot_panel.cb_combined,
            self.plot_panel.cb_filtered,
        ]:
            cb.stateChanged.connect(self._refresh_plots)

    def _read_table_row(self, row: int) -> tuple[float, float, float] | None:
        table = self.param_panel.harmonics_table
        try:
            v0 = float(table.item(row, 0).text())
            v1 = float(table.item(row, 1).text())
            v2 = float(table.item(row, 2).text())
            return v0, v1, v2
        except (AttributeError, ValueError, TypeError):
            return None

    def _read_harmonics(self) -> list[Harmonic]:
        table = self.param_panel.harmonics_table
        result = []
        for row in range(table.rowCount()):
            vals = self._read_table_row(row)
            if vals is not None:
                result.append(Harmonic(freq=vals[0], amp=vals[1], phase_deg=vals[2]))
        return result

    def _read_impulse_components(self) -> list[ImpulseComponent]:
        table = self.param_panel.harmonics_table
        result = []
        for row in range(table.rowCount()):
            vals = self._read_table_row(row)
            if vals is not None:
                result.append(
                    ImpulseComponent(width=vals[0], amp=vals[1], distance=vals[2])
                )
        return result

    def _read_gaussian_components(self) -> list[GaussianComponent]:
        table = self.param_panel.harmonics_table
        result = []
        for row in range(table.rowCount()):
            vals = self._read_table_row(row)
            if vals is not None:
                result.append(
                    GaussianComponent(center=vals[0], amp=vals[1], sigma=vals[2])
                )
        return result

    def _read_exponential_components(self) -> list[ExponentialComponent]:
        table = self.param_panel.harmonics_table
        result = []
        for row in range(table.rowCount()):
            vals = self._read_table_row(row)
            if vals is not None:
                result.append(
                    ExponentialComponent(alpha=vals[0], amp=vals[1], delay=vals[2])
                )
        return result

    def _read_params(self) -> SignalParams:
        sp = self.simulation_panel
        return SignalParams(
            duration=sp.get_duration(),
            sample_rate=sp.get_sample_rate(),
        )

    def _read_noise(self) -> NoiseParams:
        return NoiseParams(enabled=False)

    def _read_filter_from_noise_panel(self) -> FilterParams:
        nfp = self.noise_filter_panel
        return FilterParams(
            filter_type=nfp.selected_filter,
            cutoff=nfp.get_filter_cutoff(),
            r=nfp.get_filter_r(),
        )

    def _on_model(self):
        self.params = self._read_params()
        sig_type = self.param_panel.selected_signal

        harmonics = None
        impulse_components = None
        gaussian_components = None
        exponential_components = None

        if sig_type == SignalType.IMPULSE:
            impulse_components = self._read_impulse_components()
        elif sig_type == SignalType.GAUSSIAN:
            gaussian_components = self._read_gaussian_components()
        elif sig_type == SignalType.EXPONENTIAL_IMPULSE:
            exponential_components = self._read_exponential_components()
        else:
            harmonics = self._read_harmonics()

        noise = self._read_noise()
        colored_entries = self.noise_filter_panel.get_noise_entries()

        self.data = generate_signal(
            self.params,
            signal_type=sig_type,
            harmonics=harmonics,
            impulse_components=impulse_components,
            gaussian_components=gaussian_components,
            exponential_components=exponential_components,
            noise_params=noise,
            colored_noise_entries=colored_entries,
        )
        self._refresh_plots()

    def _on_segmented_generate(self):
        self.params = self._read_params()

        pool = self.segmented_panel.get_pool_entries()
        seg_min = self.segmented_panel.get_seg_min_duration()
        seg_max = self.segmented_panel.get_seg_max_duration()

        colored_entries = self.noise_filter_panel.get_noise_entries()

        self.data = generate_segmented_signal(
            self.params,
            pool=pool,
            seg_min_duration=seg_min,
            seg_max_duration=seg_max,
            colored_noise_entries=colored_entries,
        )
        self._refresh_plots()

    def _on_filter_from_noise_panel(self):
        if self.data.n == 0:
            return
        fp = self._read_filter_from_noise_panel()
        noise_on = len(self.noise_filter_panel.get_noise_entries()) > 0
        self.data = apply_filter_to_data(
            self.data, fp, self.params.sample_rate, noise_on
        )
        self._refresh_plots()

    def _refresh_plots(self):
        show_filter = self.plot_panel.cb_filtered.isChecked()
        sig_idx = self.param_panel.signal_combo.currentIndex()
        label = SIGNAL_NAMES[sig_idx]

        if self.data.segment_labels:
            label = "Сегментированный"

        self.plot_panel.update_plots(
            self.params,
            self.data,
            show_filter,
            signal_label=label,
        )

        self.statistics_panel.update_statistics(self.data.statistics)
