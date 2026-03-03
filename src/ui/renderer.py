from dataclasses import dataclass

import numpy as np
import raylib as rl


@dataclass
class Button:
    x: int
    y: int
    w: int
    h: int
    text: str

    def draw(self, active: bool = False):
        color = rl.DARKGRAY if active else rl.LIGHTGRAY
        rl.DrawRectangle(self.x, self.y, self.w, self.h, color)
        rl.DrawRectangleLines(self.x, self.y, self.w, self.h, rl.BLACK)
        rl.DrawText(self.text.encode(), self.x + 5, self.y + 8, 16, rl.BLACK)

    def clicked(self) -> bool:
        if not rl.IsMouseButtonPressed(0):
            return False
        mx, my = rl.GetMouseX(), rl.GetMouseY()
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h


class App:
    def __init__(self, width: int = 1200, height: int = 800):
        self.width, self.height = width, height
        self.signal = np.zeros(1024)
        self.fs = 1000
        self.t = np.linspace(0, 1, 1024)
        self.signal_type = 0
        self.stats = None
        self.acf = None
        self.psd_f = self.psd_p = None

        self.buttons = [
            Button(10, 10, 120, 30, "Gaussian"),
            Button(140, 10, 120, 30, "Sawtooth"),
            Button(270, 10, 120, 30, "Polyharmonic"),
            Button(400, 10, 120, 30, "+Noise"),
            Button(530, 10, 120, 30, "Load WAV"),
            Button(660, 10, 120, 30, "Save WAV"),
            Button(790, 10, 120, 30, "Analyze"),
            Button(920, 10, 120, 30, "Record"),
        ]

    def draw_graph(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        color=rl.BLUE,
    ):
        rl.DrawRectangle(x, y, w, h, rl.WHITE)
        rl.DrawRectangleLines(x, y, w, h, rl.BLACK)
        rl.DrawText(title.encode(), x + 5, y + 5, 14, rl.DARKGRAY)

        if len(data) < 2:
            return

        n = len(data)
        dmin, dmax = np.min(data), np.max(data)
        rng = dmax - dmin if dmax != dmin else 1

        margin = 20
        gx, gy, gw, gh = (
            x + margin,
            y + margin + 15,
            w - 2 * margin,
            h - 2 * margin - 15,
        )

        # Ось X (нулевая линия)
        zero_y = int(gy + gh - (0 - dmin) / rng * gh)
        rl.DrawLine(gx, zero_y, gx + gw, zero_y, rl.LIGHTGRAY)

        # Ресемплинг данных под ширину графика
        indices = np.linspace(0, n - 1, gw).astype(int)

        for i in range(gw - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            y1 = int(gy + gh - (data[idx1] - dmin) / rng * gh)
            y2 = int(gy + gh - (data[idx2] - dmin) / rng * gh)
            rl.DrawLine(gx + i, y1, gx + i + 1, y2, color)

    def draw_stats_table(self, x: int, y: int):
        if not self.stats:
            return
        rl.DrawRectangle(x, y, 300, 200, rl.WHITE)
        rl.DrawRectangleLines(x, y, 300, 200, rl.BLACK)
        rl.DrawText(b"Statistics", x + 10, y + 5, 16, rl.DARKGRAY)

        rows = [
            f"Total Power:  {self.stats.total_power:.4f}",
            f"Mean Power:   {self.stats.mean_power:.4f}",
            f"Min Amp:      {self.stats.min_amp:.4f}",
            f"Max Amp:      {self.stats.max_amp:.4f}",
            f"Mean:         {self.stats.mean:.4f}",
            f"Variance:     {self.stats.variance:.4f}",
            f"Std:          {self.stats.std:.4f}",
        ]
        for i, row in enumerate(rows):
            rl.DrawText(row.encode(), x + 10, y + 30 + i * 22, 16, rl.BLACK)

    def run(
        self,
        generate_signal,
        compute_stats,
        compute_acf,
        compute_psd,
        add_noise,
        read_wav,
        write_wav,
        record_audio,
    ):
        rl.InitWindow(self.width, self.height, b"Signal Analyzer")
        rl.SetTargetFPS(60)

        while not rl.WindowShouldClose():
            # Обработка кнопок
            if self.buttons[0].clicked():
                self.signal = generate_signal("gaussian", self.t)
                self.signal_type = 0
            elif self.buttons[1].clicked():
                self.signal = generate_signal("sawtooth", self.t)
                self.signal_type = 1
            elif self.buttons[2].clicked():
                self.signal = generate_signal("polyharmonic", self.t)
                self.signal_type = 2
            elif self.buttons[3].clicked():
                self.signal = add_noise(self.signal, 10)
            elif self.buttons[4].clicked():
                try:
                    self.signal, self.fs = read_wav("input.wav")
                    self.t = np.linspace(
                        0, len(self.signal) / self.fs, len(self.signal)
                    )
                except:
                    pass
            elif self.buttons[5].clicked():
                write_wav("output.wav", self.signal, self.fs)
            elif self.buttons[6].clicked():
                self.stats = compute_stats(self.signal)
                self.acf = compute_acf(self.signal)
                self.psd_f, self.psd_p = compute_psd(self.signal, self.fs)
            elif self.buttons[7].clicked():
                self.signal = record_audio(3.0)
                self.fs = 44100
                self.t = np.linspace(0, len(self.signal) / self.fs, len(self.signal))

            # Отрисовка
            rl.BeginDrawing()
            rl.ClearBackground(rl.RAYWHITE)

            for i, btn in enumerate(self.buttons):
                btn.draw(i == self.signal_type and i < 3)

            # Графики
            self.draw_graph(self.signal, 10, 60, 580, 220, "Signal", rl.BLUE)
            self.draw_graph(
                self.acf if self.acf is not None else np.array([]),
                600,
                60,
                580,
                220,
                "ACF",
                rl.GREEN,
            )
            self.draw_graph(
                self.psd_p if self.psd_p is not None else np.array([]),
                10,
                300,
                580,
                220,
                "PSD",
                rl.RED,
            )

            # Таблица статистики
            self.draw_stats_table(600, 300)

            rl.EndDrawing()

        rl.CloseWindow()
