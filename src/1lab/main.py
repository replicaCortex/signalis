import time

import numpy as np
import pyray as rl
import sounddevice as sd
from scipy.io import wavfile

# --- НАСТРОЙКИ ---
SAMPLE_RATE = 8000
DURATION = 1.0  # Длительность генерации в секундах
WIDTH, HEIGHT = 1000, 700
BG_COLOR = rl.GRAY

# Глобальное состояние (хранилище данных)
state = {
    "signal": np.zeros(int(SAMPLE_RATE * DURATION)),  # Пустой сигнал
    "is_recording": False,
    "recorded_frames": [],
    "stats": {},
    "params": {"amp": 0.5, "freq": 10.0, "noise": 0.1},
}

# --- ЛОГИКА (ГЕНЕРАТОРЫ И МАТЕМАТИКА) ---


def generate_signal(sig_type):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    amp = state["params"]["amp"]
    freq = state["params"]["freq"]
    noise = state["params"]["noise"]

    if sig_type == "Polyharmonic":
        # Формула: A * sin(wt) + 0.5 * A * sin(2wt) + шум
        clean = amp * (
            np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
        )
        noise_arr = noise * np.random.uniform(-1, 1, len(t))
        state["signal"] = clean + noise_arr

    elif sig_type == "AM Noise":
        # АМ Белого шума
        carrier = 1 + 0.5 * np.sin(2 * np.pi * freq * t)
        state["signal"] = amp * carrier * np.random.uniform(-1, 1, len(t)) * noise

    elif sig_type == "Pulse":
        # Импульс (Гауссов)
        mu = 0.5  # Центр
        sigma = 0.05
        state["signal"] = amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    calculate_stats()


def calculate_stats():
    sig = state["signal"]
    if len(sig) == 0:
        return

    state["stats"] = {
        "Max": np.max(sig),
        "Min": np.min(sig),
        "Mean": np.mean(sig),
        "Disperition": np.var(sig),
        "Power": np.mean(sig**2),
    }


def toggle_record():
    if not state["is_recording"]:
        state["is_recording"] = True
        state["recorded_frames"] = []
        # Запускаем поток записи
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
        # Начинаем запись (в фоне)
        state["stream"] = sd.InputStream(callback=audio_callback)
        state["stream"].start()
    else:
        state["is_recording"] = False
        state["stream"].stop()
        state["stream"].close()
        # Превращаем список кусочков в один массив
        if state["recorded_frames"]:
            full_sig = np.concatenate(state["recorded_frames"])
            state["signal"] = full_sig.flatten()
            calculate_stats()


def audio_callback(indata, frames, time, status):
    if state["is_recording"]:
        state["recorded_frames"].append(indata.copy())


def save_wav():
    # Нормализуем для сохранения (wav любит формат float32)
    wavfile.write("output.wav", SAMPLE_RATE, state["signal"].astype(np.float32))


# --- ИНТЕРФЕЙС (UI) ---


def draw_ui():
    # 1. Кнопки (Простые прямоугольники)
    # Формат: (Rectangle, Текст, Функция при нажатии, Аргумент)
    buttons = [
        (
            rl.Rectangle(20, 20, 140, 40),
            "Polyharmonic",
            generate_signal,
            "Polyharmonic",
        ),
        (rl.Rectangle(170, 20, 140, 40), "AM Noise", generate_signal, "AM Noise"),
        (rl.Rectangle(320, 20, 140, 40), "Pulse", generate_signal, "Pulse"),
        (rl.Rectangle(500, 20, 140, 40), "STOP/REC", toggle_record, None),
        (rl.Rectangle(650, 20, 140, 40), "Save WAV", save_wav, None),
    ]

    mouse_pos = rl.get_mouse_position()
    mouse_pressed = rl.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON)

    for rect, text, func, arg in buttons:
        color = (
            rl.RED if (text == "STOP/REC" and state["is_recording"]) else rl.LIGHTGRAY
        )
        # Если мышка над кнопкой -> темнее
        if rl.check_collision_point_rec(mouse_pos, rect):
            color = rl.GRAY
            if mouse_pressed:
                if arg:
                    func(arg)
                else:
                    func()

        rl.draw_rectangle_rec(rect, color)
        rl.draw_rectangle_lines_ex(rect, 2, rl.DARKGRAY)
        rl.draw_text(text, int(rect.x + 10), int(rect.y + 10), 20, rl.BLACK)

    # 2. Ползунки (Sliders) - упрощенная эмуляция клавишами
    rl.draw_text(
        f"Amplitude (UP/DOWN): {state['params']['amp']:.2f}", 20, 80, 20, rl.DARKGRAY
    )
    rl.draw_text(
        f"Frequency (LEFT/RIGHT): {state['params']['freq']:.2f}",
        20,
        110,
        20,
        rl.DARKGRAY,
    )

    # Управление с клавиатуры
    if rl.is_key_down(rl.KEY_UP):
        state["params"]["amp"] += 0.01
    if rl.is_key_down(rl.KEY_DOWN):
        state["params"]["amp"] -= 0.01
    if rl.is_key_down(rl.KEY_RIGHT):
        state["params"]["freq"] += 0.5
    if rl.is_key_down(rl.KEY_LEFT):
        state["params"]["freq"] -= 0.5

    # 3. Статистика
    y_stat = 550
    rl.draw_text("Statistics:", 20, y_stat, 20, rl.BLACK)
    for k, v in state["stats"].items():
        y_stat += 25
        rl.draw_text(f"{k}: {v:.5f}", 20, y_stat, 18, rl.DARKBLUE)


def draw_signal():
    # Область графика
    plot_rect = rl.Rectangle(50, 150, 900, 350)
    rl.draw_rectangle_rec(plot_rect, rl.WHITE)
    rl.draw_rectangle_lines_ex(plot_rect, 2, rl.BLACK)

    # Рисуем центральную ось
    center_y = plot_rect.y + plot_rect.height / 2
    rl.draw_line(
        int(plot_rect.x),
        int(center_y),
        int(plot_rect.x + plot_rect.width),
        int(center_y),
        rl.LIGHTGRAY,
    )

    sig = state["signal"]
    if len(sig) < 2:
        return

    # Оптимизация: рисуем не все точки, если их слишком много
    step = max(1, len(sig) // int(plot_rect.width))

    prev_x, prev_y = plot_rect.x, center_y - (sig[0] * 100)  # 100 - масштаб по Y

    for i in range(0, len(sig), step):
        # Масштабирование координат
        x = plot_rect.x + (i / len(sig)) * plot_rect.width
        y = center_y - (sig[i] * 100)  # Инверсия Y, т.к. в Raylib Y растет вниз

        # Обрезаем, чтобы не вылезало за рамку (клиппинг)
        y = max(plot_rect.y, min(y, plot_rect.y + plot_rect.height))

        rl.draw_line(int(prev_x), int(prev_y), int(x), int(y), rl.BLUE)
        prev_x, prev_y = x, y


# --- ГЛАВНЫЙ ЦИКЛ ---


def main():
    rl.init_window(WIDTH, HEIGHT, "Signal Lab - Python + Raylib")
    rl.set_target_fps(60)

    # Генерируем начальный сигнал
    generate_signal("Polyharmonic")

    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(BG_COLOR)

        draw_ui()
        draw_signal()

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
