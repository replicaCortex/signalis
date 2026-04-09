from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import SegmentPoolEntry
from dsp import SignalType
from widgets.simulation_panel import CollapsibleSimulationWidget

SEGMENT_TYPE_NAMES = [
    "Гармонический",
    "Гауссов импульс",
    "Пилообразный",
    "Импульсный",
    "Экспоненциальный",
    "Речевой сигнал",
    "Амплитудная модуляция",
]

SEGMENT_TYPE_MAP = {
    0: SignalType.HARMONIC,
    1: SignalType.GAUSSIAN,
    2: SignalType.SAWTOOTH,
    3: SignalType.IMPULSE,
    4: SignalType.EXPONENTIAL_IMPULSE,
    5: SignalType.SPEECH,
    6: SignalType.AM_MODULATED,
}


class SegmentEntryWidget(QGroupBox):
    """Виджет одного элемента пула сегментов."""

    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, index: int = 0, parent=None):
        super().__init__(f"Сегмент #{index + 1}", parent)
        self._index = index
        self._build_ui()
        self._on_type_changed(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Включение
        self.cb_enabled = QCheckBox("Включён")
        self.cb_enabled.setChecked(True)
        self.cb_enabled.stateChanged.connect(self.changed)
        layout.addWidget(self.cb_enabled)

        # Тип сигнала
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Тип:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(SEGMENT_TYPE_NAMES)
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        row_type.addWidget(self.combo_type)
        layout.addLayout(row_type)

        # === Параметры для гармонического ===
        self.harmonic_group = QWidget()
        hg_layout = QVBoxLayout(self.harmonic_group)
        hg_layout.setContentsMargins(0, 0, 0, 0)

        row_freq = QHBoxLayout()
        row_freq.addWidget(QLabel("Частота, Гц:"))
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.1, 50000.0)
        self.spin_freq.setValue(5.0)
        self.spin_freq.setDecimals(2)
        self.spin_freq.setSingleStep(1.0)
        row_freq.addWidget(self.spin_freq)
        hg_layout.addLayout(row_freq)

        row_amp = QHBoxLayout()
        row_amp.addWidget(QLabel("Амплитуда:"))
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.001, 10000.0)
        self.spin_amp.setValue(1.0)
        self.spin_amp.setDecimals(3)
        self.spin_amp.setSingleStep(0.1)
        row_amp.addWidget(self.spin_amp)
        hg_layout.addLayout(row_amp)

        row_phase = QHBoxLayout()
        self.lbl_phase = QLabel("Фаза, °:")
        row_phase.addWidget(self.lbl_phase)
        self.spin_phase = QDoubleSpinBox()
        self.spin_phase.setRange(-360.0, 360.0)
        self.spin_phase.setValue(0.0)
        self.spin_phase.setDecimals(1)
        self.spin_phase.setSingleStep(15.0)
        row_phase.addWidget(self.spin_phase)
        hg_layout.addLayout(row_phase)

        layout.addWidget(self.harmonic_group)

        # === Параметры для пилообразного ===
        self.sawtooth_group = QWidget()
        sw_layout = QVBoxLayout(self.sawtooth_group)
        sw_layout.setContentsMargins(0, 0, 0, 0)

        row_saw_freq = QHBoxLayout()
        row_saw_freq.addWidget(QLabel("Частота, Гц:"))
        self.spin_saw_freq = QDoubleSpinBox()
        self.spin_saw_freq.setRange(0.1, 50000.0)
        self.spin_saw_freq.setValue(5.0)
        self.spin_saw_freq.setDecimals(2)
        self.spin_saw_freq.setSingleStep(1.0)
        row_saw_freq.addWidget(self.spin_saw_freq)
        sw_layout.addLayout(row_saw_freq)

        row_saw_amp = QHBoxLayout()
        row_saw_amp.addWidget(QLabel("Амплитуда:"))
        self.spin_saw_amp = QDoubleSpinBox()
        self.spin_saw_amp.setRange(0.001, 10000.0)
        self.spin_saw_amp.setValue(1.0)
        self.spin_saw_amp.setDecimals(3)
        self.spin_saw_amp.setSingleStep(0.1)
        row_saw_amp.addWidget(self.spin_saw_amp)
        sw_layout.addLayout(row_saw_amp)

        row_saw_phase = QHBoxLayout()
        row_saw_phase.addWidget(QLabel("Фаза, °:"))
        self.spin_saw_phase = QDoubleSpinBox()
        self.spin_saw_phase.setRange(-360.0, 360.0)
        self.spin_saw_phase.setValue(0.0)
        self.spin_saw_phase.setDecimals(1)
        self.spin_saw_phase.setSingleStep(15.0)
        row_saw_phase.addWidget(self.spin_saw_phase)
        sw_layout.addLayout(row_saw_phase)

        layout.addWidget(self.sawtooth_group)

        # === Параметры для гауссова импульса ===
        self.gauss_group = QWidget()
        gg_layout = QVBoxLayout(self.gauss_group)
        gg_layout.setContentsMargins(0, 0, 0, 0)

        row_gcenter = QHBoxLayout()
        row_gcenter.addWidget(QLabel("Мат. ожидание (центр), с:"))
        self.spin_gauss_center = QDoubleSpinBox()
        self.spin_gauss_center.setRange(-100000.0, 100000.0)
        self.spin_gauss_center.setValue(0.5)
        self.spin_gauss_center.setDecimals(4)
        self.spin_gauss_center.setSingleStep(0.1)
        row_gcenter.addWidget(self.spin_gauss_center)
        gg_layout.addLayout(row_gcenter)

        row_gamp = QHBoxLayout()
        row_gamp.addWidget(QLabel("Амплитуда:"))
        self.spin_gauss_amp = QDoubleSpinBox()
        self.spin_gauss_amp.setRange(0.001, 10000.0)
        self.spin_gauss_amp.setValue(1.0)
        self.spin_gauss_amp.setDecimals(3)
        row_gamp.addWidget(self.spin_gauss_amp)
        gg_layout.addLayout(row_gamp)

        row_sigma = QHBoxLayout()
        row_sigma.addWidget(QLabel("Сигма (σ), с:"))
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.001, 100.0)
        self.spin_gauss_sigma.setValue(0.05)
        self.spin_gauss_sigma.setDecimals(4)
        self.spin_gauss_sigma.setSingleStep(0.01)
        row_sigma.addWidget(self.spin_gauss_sigma)
        gg_layout.addLayout(row_sigma)

        layout.addWidget(self.gauss_group)

        # === Параметры для импульсного ===
        self.impulse_group = QWidget()
        ig_layout = QVBoxLayout(self.impulse_group)
        ig_layout.setContentsMargins(0, 0, 0, 0)

        row_iw = QHBoxLayout()
        row_iw.addWidget(QLabel("Длительность, с:"))
        self.spin_imp_width = QDoubleSpinBox()
        self.spin_imp_width.setRange(0.001, 100.0)
        self.spin_imp_width.setValue(0.02)
        self.spin_imp_width.setDecimals(4)
        self.spin_imp_width.setSingleStep(0.01)
        row_iw.addWidget(self.spin_imp_width)
        ig_layout.addLayout(row_iw)

        row_ia = QHBoxLayout()
        row_ia.addWidget(QLabel("Амплитуда:"))
        self.spin_imp_amp = QDoubleSpinBox()
        self.spin_imp_amp.setRange(0.001, 10000.0)
        self.spin_imp_amp.setValue(1.0)
        self.spin_imp_amp.setDecimals(3)
        row_ia.addWidget(self.spin_imp_amp)
        ig_layout.addLayout(row_ia)

        row_ip = QHBoxLayout()
        row_ip.addWidget(QLabel("Период повт., с:"))
        self.spin_imp_period = QDoubleSpinBox()
        self.spin_imp_period.setRange(0.0, 100.0)
        self.spin_imp_period.setValue(0.1)
        self.spin_imp_period.setDecimals(4)
        self.spin_imp_period.setSingleStep(0.01)
        row_ip.addWidget(self.spin_imp_period)
        ig_layout.addLayout(row_ip)

        row_id = QHBoxLayout()
        row_id.addWidget(QLabel("Задержка (сдвиг), с:"))
        self.spin_imp_delay = QDoubleSpinBox()
        self.spin_imp_delay.setRange(0.0, 100000.0)
        self.spin_imp_delay.setValue(0.0)
        self.spin_imp_delay.setDecimals(4)
        self.spin_imp_delay.setSingleStep(0.01)
        row_id.addWidget(self.spin_imp_delay)
        ig_layout.addLayout(row_id)

        layout.addWidget(self.impulse_group)

        # === Параметры для экспоненциального ===
        self.exp_group = QWidget()
        eg_layout = QVBoxLayout(self.exp_group)
        eg_layout.setContentsMargins(0, 0, 0, 0)

        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Коэф. α:"))
        self.spin_exp_alpha = QDoubleSpinBox()
        self.spin_exp_alpha.setRange(-1000.0, 1000.0)
        self.spin_exp_alpha.setValue(-5.0)
        self.spin_exp_alpha.setDecimals(2)
        self.spin_exp_alpha.setSingleStep(1.0)
        row_alpha.addWidget(self.spin_exp_alpha)
        eg_layout.addLayout(row_alpha)

        row_ea = QHBoxLayout()
        row_ea.addWidget(QLabel("Амплитуда:"))
        self.spin_exp_amp = QDoubleSpinBox()
        self.spin_exp_amp.setRange(0.001, 10000.0)
        self.spin_exp_amp.setValue(1.0)
        self.spin_exp_amp.setDecimals(3)
        row_ea.addWidget(self.spin_exp_amp)
        eg_layout.addLayout(row_ea)

        row_ed = QHBoxLayout()
        row_ed.addWidget(QLabel("Задержка, с:"))
        self.spin_exp_delay = QDoubleSpinBox()
        self.spin_exp_delay.setRange(0.0, 100000.0)
        self.spin_exp_delay.setValue(0.0)
        self.spin_exp_delay.setDecimals(4)
        self.spin_exp_delay.setSingleStep(0.1)
        row_ed.addWidget(self.spin_exp_delay)
        eg_layout.addLayout(row_ed)

        layout.addWidget(self.exp_group)

        # === Параметры для речевого сигнала ===
        self.speech_group = QWidget()
        sg_layout = QVBoxLayout(self.speech_group)
        sg_layout.setContentsMargins(0, 0, 0, 0)

        row_file = QHBoxLayout()
        row_file.addWidget(QLabel("WAV файл:"))
        self.ed_speech_file = QLineEdit()
        self.ed_speech_file.setPlaceholderText("Выберите файл...")
        self.ed_speech_file.setReadOnly(True)
        row_file.addWidget(self.ed_speech_file)
        self.btn_browse = QPushButton("📂")
        self.btn_browse.setFixedWidth(36)
        self.btn_browse.clicked.connect(self._browse_wav)
        row_file.addWidget(self.btn_browse)
        sg_layout.addLayout(row_file)

        row_sa = QHBoxLayout()
        row_sa.addWidget(QLabel("Амплитуда:"))
        self.spin_speech_amp = QDoubleSpinBox()
        self.spin_speech_amp.setRange(0.001, 10000.0)
        self.spin_speech_amp.setValue(1.0)
        self.spin_speech_amp.setDecimals(3)
        self.spin_speech_amp.setSingleStep(0.1)
        row_sa.addWidget(self.spin_speech_amp)
        sg_layout.addLayout(row_sa)

        self.lbl_file_info = QLabel("")
        self.lbl_file_info.setStyleSheet("color: #888; font-size: 10px;")
        sg_layout.addWidget(self.lbl_file_info)

        layout.addWidget(self.speech_group)

        self.am_group = QWidget()
        am_layout = QVBoxLayout(self.am_group)
        am_layout.setContentsMargins(0, 0, 0, 0)

        row_am_cf = QHBoxLayout()
        row_am_cf.addWidget(QLabel("Несущая частота, Гц:"))
        self.spin_am_carrier_freq = QDoubleSpinBox()
        self.spin_am_carrier_freq.setRange(0.1, 50000.0)
        self.spin_am_carrier_freq.setValue(50.0)
        self.spin_am_carrier_freq.setDecimals(2)
        self.spin_am_carrier_freq.setSingleStep(10.0)
        row_am_cf.addWidget(self.spin_am_carrier_freq)
        am_layout.addLayout(row_am_cf)

        row_am_ca = QHBoxLayout()
        row_am_ca.addWidget(QLabel("Амплитуда несущей:"))
        self.spin_am_carrier_amp = QDoubleSpinBox()
        self.spin_am_carrier_amp.setRange(0.001, 10000.0)
        self.spin_am_carrier_amp.setValue(1.0)
        self.spin_am_carrier_amp.setDecimals(3)
        self.spin_am_carrier_amp.setSingleStep(0.1)
        row_am_ca.addWidget(self.spin_am_carrier_amp)
        am_layout.addLayout(row_am_ca)

        row_am_mf = QHBoxLayout()
        row_am_mf.addWidget(QLabel("Частота модуляции, Гц:"))
        self.spin_am_mod_freq = QDoubleSpinBox()
        self.spin_am_mod_freq.setRange(0.1, 1000.0)
        self.spin_am_mod_freq.setValue(5.0)
        self.spin_am_mod_freq.setDecimals(2)
        self.spin_am_mod_freq.setSingleStep(1.0)
        row_am_mf.addWidget(self.spin_am_mod_freq)
        am_layout.addLayout(row_am_mf)

        row_am_md = QHBoxLayout()
        row_am_md.addWidget(QLabel("Глубина модуляции:"))
        self.spin_am_mod_depth = QDoubleSpinBox()
        self.spin_am_mod_depth.setRange(0.0, 1.0)
        self.spin_am_mod_depth.setValue(0.8)
        self.spin_am_mod_depth.setDecimals(2)
        self.spin_am_mod_depth.setSingleStep(0.1)
        row_am_md.addWidget(self.spin_am_mod_depth)
        am_layout.addLayout(row_am_md)

        layout.addWidget(self.am_group)

        # Метка минимальной длительности
        self.lbl_min_duration = QLabel("Мин. длительность: —")
        self.lbl_min_duration.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.lbl_min_duration)

        # Кнопка удаления
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(self.btn_remove)

        # Обновляем мин. длительность при изменении параметров
        for spin in [
            self.spin_freq,
            self.spin_amp,
            self.spin_phase,
            self.spin_saw_freq,
            self.spin_saw_amp,
            self.spin_saw_phase,
            self.spin_gauss_center,
            self.spin_gauss_amp,
            self.spin_gauss_sigma,
            self.spin_imp_width,
            self.spin_imp_amp,
            self.spin_imp_period,
            self.spin_imp_delay,
            self.spin_exp_alpha,
            self.spin_exp_amp,
            self.spin_exp_delay,
            self.spin_speech_amp,
            self.spin_am_carrier_freq,  # НОВОЕ
            self.spin_am_carrier_amp,
            self.spin_am_mod_freq,
            self.spin_am_mod_depth,
        ]:
            spin.valueChanged.connect(self._update_min_duration_label)

    def _browse_wav(self):
        """Открывает диалог выбора WAV файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите WAV файл", "", "WAV файлы (*.wav);;Все файлы (*.*)"
        )
        if path:
            self.ed_speech_file.setText(path)
            self._show_wav_info(path)
            self.changed.emit()

    def _show_wav_info(self, path: str):
        """Показывает информацию о WAV файле."""
        import wave

        try:
            with wave.open(path, "rb") as wf:
                ch = wf.getnchannels()
                sr = wf.getframerate()
                n = wf.getnframes()
                dur = n / sr
                self.lbl_file_info.setText(
                    f"Каналы: {ch} | Частота: {sr} Гц | "
                    f"Длительность: {dur:.2f} с | Сэмплы: {n}"
                )
        except Exception as e:
            self.lbl_file_info.setText(f"Ошибка: {e}")

    def _on_type_changed(self, index: int):
        """Показываем/скрываем группы параметров."""
        sig_type = SEGMENT_TYPE_MAP.get(index, SignalType.HARMONIC)

        self.harmonic_group.setVisible(sig_type == SignalType.HARMONIC)
        self.sawtooth_group.setVisible(sig_type == SignalType.SAWTOOTH)
        self.gauss_group.setVisible(sig_type == SignalType.GAUSSIAN)
        self.impulse_group.setVisible(sig_type == SignalType.IMPULSE)
        self.exp_group.setVisible(sig_type == SignalType.EXPONENTIAL_IMPULSE)
        self.speech_group.setVisible(sig_type == SignalType.SPEECH)
        self.am_group.setVisible(sig_type == SignalType.AM_MODULATED)

        self._update_min_duration_label()
        self.changed.emit()

    def _update_min_duration_label(self):
        entry = self.to_entry()
        min_dur = entry.min_duration_for_one_period()
        self.lbl_min_duration.setText(f"Мин. длительность (1 период): {min_dur:.4f} с")

    def to_entry(self) -> SegmentPoolEntry:
        idx = self.combo_type.currentIndex()
        sig_type = SEGMENT_TYPE_MAP.get(idx, SignalType.HARMONIC)

        # Для пилообразного используем отдельные спинбоксы
        if sig_type == SignalType.SAWTOOTH:
            freq = self.spin_saw_freq.value()
            amp = self.spin_saw_amp.value()
            saw_phase = self.spin_saw_phase.value()
        else:
            freq = self.spin_freq.value()
            amp = self.spin_amp.value()
            saw_phase = 0.0

        return SegmentPoolEntry(
            signal_type=sig_type,
            enabled=self.cb_enabled.isChecked(),
            freq=freq,
            amp=amp,
            phase_deg=self.spin_phase.value(),
            saw_phase_deg=saw_phase,
            gauss_amp=self.spin_gauss_amp.value(),
            gauss_sigma=self.spin_gauss_sigma.value(),
            gauss_center=self.spin_gauss_center.value(),
            impulse_width=self.spin_imp_width.value(),
            impulse_amp=self.spin_imp_amp.value(),
            impulse_period=self.spin_imp_period.value(),
            impulse_delay=self.spin_imp_delay.value(),
            exp_alpha=self.spin_exp_alpha.value(),
            exp_amp=self.spin_exp_amp.value(),
            exp_delay=self.spin_exp_delay.value(),
            speech_file=self.ed_speech_file.text(),
            speech_amp=self.spin_speech_amp.value(),
            am_carrier_freq=self.spin_am_carrier_freq.value(),
            am_carrier_amp=self.spin_am_carrier_amp.value(),
            am_mod_freq=self.spin_am_mod_freq.value(),
            am_mod_depth=self.spin_am_mod_depth.value(),
        )

    def set_index(self, idx: int):
        self._index = idx
        self.setTitle(f"Сегмент #{idx + 1}")


class SegmentedPanel(QGroupBox):
    """Панель настройки сигнала с встроенными параметрами симуляции."""

    generate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Сигнал", parent)
        self._entries: list[SegmentEntryWidget] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Параметры симуляции (сворачиваемый блок) ──
        self.sim_widget = CollapsibleSimulationWidget()
        main_layout.addWidget(self.sim_widget)

        # ── Режим заполнения ──
        mode_group = QGroupBox("Режим генерации")
        mode_layout = QVBoxLayout(mode_group)

        self.cb_single_type = QCheckBox("Заполнить одним типом сигнала")
        self.cb_single_type.setToolTip(
            "Первый включённый сигнал из пула заполнит\n"
            "весь временной интервал без сегментации."
        )
        self.cb_single_type.stateChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.cb_single_type)

        # Настройки сегментации
        self.seg_settings_widget = QWidget()
        seg_layout = QVBoxLayout(self.seg_settings_widget)
        seg_layout.setContentsMargins(0, 0, 0, 0)

        row_min = QHBoxLayout()
        row_min.addWidget(QLabel("Мин. длительность, с:"))
        self.spin_seg_min = QDoubleSpinBox()
        self.spin_seg_min.setRange(0.001, 100.0)
        self.spin_seg_min.setValue(0.05)
        self.spin_seg_min.setDecimals(4)
        self.spin_seg_min.setSingleStep(0.01)
        row_min.addWidget(self.spin_seg_min)
        seg_layout.addLayout(row_min)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Макс. длительность, с:"))
        self.spin_seg_max = QDoubleSpinBox()
        self.spin_seg_max.setRange(0.001, 100.0)
        self.spin_seg_max.setValue(0.3)
        self.spin_seg_max.setDecimals(4)
        self.spin_seg_max.setSingleStep(0.05)
        row_max.addWidget(self.spin_seg_max)
        seg_layout.addLayout(row_max)

        row_periods = QHBoxLayout()
        row_periods.addWidget(QLabel("Мин. периодов в сегменте:"))
        self.spin_min_periods = QSpinBox()
        self.spin_min_periods.setRange(1, 100)
        self.spin_min_periods.setValue(5)  # НОВОЕ: по умолчанию 2 периода
        row_periods.addWidget(self.spin_min_periods)
        seg_layout.addLayout(row_periods)

        self.lbl_info = QLabel(
            "Мин. длительность будет автоматически увеличена\n"
            "для сегментов, которым нужен полный период."
        )
        self.lbl_info.setStyleSheet("color: #888; font-size: 10px;")
        seg_layout.addWidget(self.lbl_info)

        mode_layout.addWidget(self.seg_settings_widget)
        main_layout.addWidget(mode_group)

        # ── Пул сегментов ──
        pool_group = QGroupBox("Пул сигналов")
        pool_outer = QVBoxLayout(pool_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._pool_layout = QVBoxLayout(scroll_widget)
        self._pool_layout.addStretch()
        scroll.setWidget(scroll_widget)
        pool_outer.addWidget(scroll)

        # Кнопки
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Добавить сигнал")
        self.btn_add.clicked.connect(self._add_entry)
        btn_row.addWidget(self.btn_add)

        self.btn_add_all = QPushButton("Добавить все типы")
        self.btn_add_all.clicked.connect(self._add_all_types)
        btn_row.addWidget(self.btn_add_all)
        pool_outer.addLayout(btn_row)

        self.btn_clear = QPushButton("Очистить пул")
        self.btn_clear.clicked.connect(self._clear_all)
        pool_outer.addWidget(self.btn_clear)

        main_layout.addWidget(pool_group)

        # ── Кнопка генерации ──
        self.btn_generate = QPushButton("Сгенерировать сигнал")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; font-size: 13px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.btn_generate.clicked.connect(self.generate_requested)
        main_layout.addWidget(self.btn_generate)

    def _on_mode_changed(self):
        is_single = self.cb_single_type.isChecked()
        self.seg_settings_widget.setVisible(not is_single)

    def _add_entry(self, type_idx: int = 0) -> SegmentEntryWidget:
        idx = len(self._entries)
        entry = SegmentEntryWidget(index=idx)
        entry.combo_type.setCurrentIndex(type_idx)
        entry.removed.connect(self._remove_entry)

        self._entries.append(entry)
        self._pool_layout.insertWidget(self._pool_layout.count() - 1, entry)
        return entry

    def get_min_periods(self) -> int:
        return self.spin_min_periods.value()

    def _remove_entry(self, entry: SegmentEntryWidget):
        if entry in self._entries:
            self._entries.remove(entry)
            self._pool_layout.removeWidget(entry)
            entry.deleteLater()
            self._reindex()

    def _clear_all(self):
        for entry in list(self._entries):
            self._pool_layout.removeWidget(entry)
            entry.deleteLater()
        self._entries.clear()

    def _add_all_types(self):
        for i in range(len(SEGMENT_TYPE_NAMES)):
            self._add_entry(type_idx=i)

    def _reindex(self):
        for i, entry in enumerate(self._entries):
            entry.set_index(i)

    # ── Публичный API ──

    @property
    def is_single_type(self) -> bool:
        return self.cb_single_type.isChecked()

    def get_pool_entries(self) -> list[SegmentPoolEntry]:
        return [e.to_entry() for e in self._entries]

    def get_first_enabled_entry(self) -> SegmentPoolEntry | None:
        for e in self._entries:
            entry = e.to_entry()
            if entry.enabled:
                return entry
        return None

    def get_seg_min_duration(self) -> float:
        return self.spin_seg_min.value()

    def get_seg_max_duration(self) -> float:
        return self.spin_seg_max.value()

    def get_duration(self) -> float:
        return self.sim_widget.get_duration()

    def get_sample_rate(self) -> float:
        return self.sim_widget.get_sample_rate()
