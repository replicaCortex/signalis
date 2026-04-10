from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSimulationWidget(QWidget):
    """Компактный сворачиваемый виджет параметров симуляции."""

    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setText("Параметры симуляции")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setToolButtonStyle(2)  # Qt.ToolButtonTextOnly
        self._toggle_btn.toggled.connect(self._on_toggle)
        outer_layout.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content.setObjectName("simulationContent")
        # Убираем явный фон, делаем прозрачным
        self._content.setStyleSheet("background-color: transparent;")

        form = QFormLayout(self._content)
        form.setContentsMargins(6, 4, 6, 4)
        form.setSpacing(4)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.001, 100000.0)
        self.spin_T.setValue(5.0)
        self.spin_T.setDecimals(4)
        self.spin_T.setSingleStep(0.1)
        self.spin_T.setSuffix(" с")
        self.spin_T.valueChanged.connect(self._update_derived)
        form.addRow("Время T:", self.spin_T)

        self.spin_fd = QDoubleSpinBox()
        self.spin_fd.setRange(1.0, 1000000.0)
        self.spin_fd.setValue(100.0)
        self.spin_fd.setDecimals(1)
        self.spin_fd.setSingleStep(10.0)
        self.spin_fd.setSuffix(" Гц")
        self.spin_fd.valueChanged.connect(self._update_derived)
        form.addRow("Частота Fd:", self.spin_fd)

        self.lbl_dt = QLabel("0.01с")
        form.addRow("Шаг ΔT:", self.lbl_dt)

        self.lbl_N = QLabel("100")
        form.addRow("Точек N:", self.lbl_N)

        self._content.setVisible(False)
        outer_layout.addWidget(self._content)

        self._update_derived()

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self._toggle_btn.setText("Параметры симуляции")

    def _update_derived(self):
        T = self.spin_T.value()
        fd = self.spin_fd.value()
        if fd > 0 and T > 0:
            dt = 1.0 / fd
            N = int(round(T / dt))
            self.lbl_dt.setText(f"{dt:.6f} с")
            self.lbl_N.setText(str(N))
        self.params_changed.emit()

    def get_duration(self) -> float:
        return self.spin_T.value()

    def get_sample_rate(self) -> float:
        return self.spin_fd.value()
