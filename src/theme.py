"""Тёмная тема Catppuccin Mocha для приложения."""

from catppuccin import PALETTE

MOCHA = PALETTE.mocha.colors


def get_qss_theme() -> str:
    """Возвращает QSS stylesheet с темой Catppuccin Mocha."""
    crust = MOCHA.crust.hex  # #1e1e2e - основной фон
    mantle = MOCHA.mantle.hex  # #181825
    base = MOCHA.base.hex  # #1e1e2e
    surface0 = MOCHA.surface0.hex  # #313244 - фон виджетов
    surface1 = MOCHA.surface1.hex  # #45475a - границы, кнопки
    surface2 = MOCHA.surface2.hex  # #585b70 - ховер
    text = MOCHA.text.hex  # #cdd6f4 - текст
    subtext0 = MOCHA.subtext0.hex  # #a6adc8 - подсказки
    lavender = MOCHA.lavender.hex  # #b4befe - заголовки групп
    mauve = MOCHA.mauve.hex  # #cba6f7 - акцент

    return f"""
        /* === Общие стили === */
        QWidget {{
            background-color: {crust};
            color: {text};
            font-size: 13px;
        }}
        QMainWindow, QTabWidget::pane {{
            background-color: {crust};
        }}
        
        /* === Левая панель (вкладки) - единый фон === */
        QTabWidget {{
            background-color: {crust};
        }}
        QTabWidget::pane {{
            background-color: {crust};
            border: none;
        }}
        
        /* === Скролл-области - прозрачный фон === */
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background-color: transparent;
        }}
        QScrollArea > QWidget {{
            background-color: transparent;
        }}
        
        /* === ВСЕ контейнеры внутри скролла делаем прозрачными === */
        QScrollArea QWidget {{
            background-color: transparent;
        }}
        
        /* === Группы (GroupBox) === */
        QGroupBox {{
            background-color: {surface0};
            border: 1px solid {surface1};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: bold;
            color: {lavender};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            top: 0px;
            padding: 0 5px;
            color: {lavender};
            background-color: transparent;
        }}
        
        /* === Внутри GroupBox делаем фон прозрачным для вложенных виджетов === */
        QGroupBox QWidget {{
            background-color: transparent;
        }}

        /* === Вкладки табов === */
        QTabBar::tab {{
            background-color: {surface0};
            color: {text};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {crust};
            color: {mauve};
            border-bottom: 2px solid {mauve};
        }}
        QTabBar::tab:hover {{
            background-color: {surface1};
        }}

        /* === Кнопки === */
        QPushButton {{
            background-color: {surface1};
            color: {text};
            border: 1px solid {surface2};
            border-radius: 4px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: {surface2};
            border-color: {subtext0};
        }}
        QPushButton:pressed {{
            background-color: {surface0};
        }}

        /* === ToolButton (для сворачиваемой панели) === */
        QToolButton {{
            background-color: {surface0};
            color: {text};
            border: 1px solid {surface1};
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
            text-align: left;
        }}
        QToolButton:checked {{
            background-color: {surface1};
            border-color: {mauve};
        }}
        QToolButton:hover {{
            background-color: {surface2};
        }}

        /* === SpinBox === */
        QSpinBox, QDoubleSpinBox {{
            background-color: {surface0};
            color: {text};
            border: 1px solid {surface1};
            border-radius: 4px;
            padding: 3px 6px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {mauve};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {surface1};
            border: none;
            border-radius: 2px;
            width: 16px;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {surface2};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid {text};
            width: 0;
            height: 0;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {text};
            width: 0;
            height: 0;
        }}

        /* === ComboBox === */
        QComboBox {{
            background-color: {surface0};
            color: {text};
            border: 1px solid {surface1};
            border-radius: 4px;
            padding: 3px 6px;
        }}
        QComboBox:focus {{
            border-color: {mauve};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {text};
            width: 0;
            height: 0;
        }}
        QComboBox QAbstractItemView {{
            background-color: {surface0};
            color: {text};
            selection-background-color: {surface1};
            border: 1px solid {surface1};
        }}

        /* === LineEdit === */
        QLineEdit {{
            background-color: {surface0};
            color: {text};
            border: 1px solid {surface1};
            border-radius: 4px;
            padding: 3px 6px;
        }}
        QLineEdit:focus {{
            border-color: {mauve};
        }}

        /* === CheckBox === */
        QCheckBox {{
            color: {text};
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            background-color: {surface0};
            border: 1px solid {surface1};
            border-radius: 3px;
        }}
        QCheckBox::indicator:checked {{
            background-color: {mauve};
            border-color: {mauve};
        }}
        QCheckBox::indicator:hover {{
            border-color: {mauve};
        }}

        /* === Скроллбары === */
        QScrollBar:vertical {{
            background-color: transparent;
            width: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {surface1};
            border-radius: 6px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {surface2};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background-color: transparent;
            height: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {surface1};
            border-radius: 6px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {surface2};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        /* === Label - ВАЖНО: прозрачный фон для всех надписей === */
        QLabel {{
            color: {text};
            background-color: transparent;
            border: none;
        }}
        
        /* === Дополнительно: для FormLayout явно делаем прозрачным === */
        QFormLayout QLabel {{
            background-color: transparent;
        }}
        
        /* === Для всех контейнеров внутри GroupBox === */
        QGroupBox QLabel {{
            background-color: transparent;
        }}
        
        /* === Для информационных надписей (например, в segmented_panel) === */
        QLabel[infoLabel="true"] {{
            color: {subtext0};
            background-color: transparent;
            font-size: 10px;
            font-style: italic;
        }}
    """
