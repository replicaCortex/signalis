# src/widgets/simulation_panel.py
from PyQt5.QtWidgets import (
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


def _add(layout: QVBoxLayout, label: str, widget):
    layout.addWidget(QLabel(label))
    layout.addWidget(widget)


class SimulationPanel(QGroupBox):
    """Панель параметров симуляции: T, dt, N, Fd."""

    def __init__(self, parent=None):
        super().__init__("Параметры симуляции", parent)

        layout = QVBoxLayout(self)

        self.ed_T = QLineEdit("1.0")
        self.ed_dt = QLineEdit("0.01")
        self.ed_N = QLineEdit("100")
        self.ed_fd = QLineEdit("100.0")

        # dt и N — автовычисляемые, только для чтения
        self.ed_dt.setReadOnly(True)
        self.ed_N.setReadOnly(True)
        self.ed_dt.setStyleSheet("background-color: #f0f0f0;")
        self.ed_N.setStyleSheet("background-color: #f0f0f0;")

        _add(layout, "Время T, с:", self.ed_T)
        _add(layout, "Шаг ΔT, с (авто):", self.ed_dt)
        _add(layout, "Кол-во точек N (авто):", self.ed_N)
        _add(layout, "Частота дискретизации Fd, Гц:", self.ed_fd)

        # Автообновление при изменении T или Fd
        self.ed_T.textChanged.connect(self._update_derived_params)
        self.ed_fd.textChanged.connect(self._update_derived_params)

        layout.addStretch()

        # Инициализируем
        self._update_derived_params()

    def _update_derived_params(self):
        """Обновляет dt и N на основе T и Fd."""
        try:
            T = float(self.ed_T.text())
            fd = float(self.ed_fd.text())
            if fd > 0 and T > 0:
                dt = 1.0 / fd
                N = int(round(T / dt))
                self.ed_dt.setText(f"{dt:.6f}")
                self.ed_N.setText(str(N))
        except (ValueError, ZeroDivisionError):
            pass

    def get_duration(self) -> float:
        try:
            return float(self.ed_T.text())
        except ValueError:
            return 1.0

    def get_sample_rate(self) -> float:
        try:
            return float(self.ed_fd.text())
        except ValueError:
            return 100.0
