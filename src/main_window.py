from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from app import (
    FilterParams,
    Harmonic,
    NoiseParams,
    SignalData,
    SignalParams,
    apply_filter_to_data,
    generate_signal,
)
from dsp import SignalType
from widgets.param_panel import SIGNAL_NAMES, ParamPanel
from widgets.plot_panel import PlotPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FooBarPrime")
        self.setGeometry(100, 100, 1200, 800)

        self.data = SignalData()
        self.params = SignalParams()

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.param_panel = ParamPanel()
        self.plot_panel = PlotPanel()

        layout.addWidget(self.param_panel, stretch=1)
        layout.addWidget(self.plot_panel, stretch=3)

    def _connect_signals(self):
        pp = self.param_panel
        pp.btn_model.clicked.connect(self._on_model)
        pp.btn_clear.clicked.connect(pp.clear_table)
        pp.btn_filter.clicked.connect(self._on_filter)
        pp.btn_add_row.clicked.connect(pp.add_row)

        for cb in [
            self.plot_panel.cb_base,
            self.plot_panel.cb_noise,
            self.plot_panel.cb_combined,
            self.plot_panel.cb_filtered,
        ]:
            cb.stateChanged.connect(self._refresh_plots)

    def _read_harmonics(self) -> list[Harmonic]:
        table = self.param_panel.harmonics_table
        result = []
        for row in range(table.rowCount()):
            try:
                f = float(table.item(row, 0).text())
                a = float(table.item(row, 1).text())
                p = float(table.item(row, 2).text())
                result.append(Harmonic(freq=f, amp=a, phase_deg=p))
            except (AttributeError, ValueError, TypeError):
                continue
        return result

    def _read_params(self) -> SignalParams:
        pp = self.param_panel
        params = SignalParams(
            duration=float(pp.ed_T.text()),
            sample_rate=float(pp.ed_fd.text()),
        )
        pp.ed_dt.setText(f"{params.dt:.6f}")
        pp.ed_N.setText(str(params.n_samples))
        return params

    def _read_noise(self) -> NoiseParams:
        pp = self.param_panel
        return NoiseParams(
            enabled=pp.cb_noise.isChecked(),
            amp_min=int(pp.ed_amp_min.text()),
            amp_max=int(pp.ed_amp_max.text()),
            freq=float(pp.ed_noise_freq.text()),
        )

    def _read_filter(self) -> FilterParams:
        pp = self.param_panel
        return FilterParams(
            filter_type=pp.selected_filter,
            cutoff=float(pp.ed_fc.text()),
            r=float(pp.ed_r.text()),
        )

    def _on_model(self):
        self.params = self._read_params()
        harmonics = self._read_harmonics()
        noise = self._read_noise()
        sig_type = self.param_panel.selected_signal

        self.data = generate_signal(
            self.params,
            harmonics,
            noise,
            signal_type=sig_type,
        )
        self._refresh_plots()

    def _on_filter(self):
        if self.data.n == 0:
            return

        fp = self._read_filter()
        noise_on = self.param_panel.cb_noise.isChecked()

        self.data = apply_filter_to_data(
            self.data,
            fp,
            self.params.sample_rate,
            noise_on,
        )
        self._refresh_plots()

    def _refresh_plots(self):
        show_filter = self.plot_panel.cb_filtered.isChecked()
        sig_idx = self.param_panel.signal_combo.currentIndex()
        label = SIGNAL_NAMES[sig_idx]

        self.plot_panel.update_plots(
            self.params,
            self.data,
            show_filter,
            signal_label=label,
        )
