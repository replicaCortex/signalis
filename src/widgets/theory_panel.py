from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class TheoryPanel(QGroupBox):
    """Панель с теоретическими сведениями и формулами."""

    def __init__(self, parent=None):
        super().__init__("Теоретические сведения", parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        sections = [
            self._section_signals(),
            self._section_signal_formulas(),
            self._section_discretization(),
            self._section_statistics(),
            self._section_correlation(),
            self._section_spectrum(),
            self._section_dft_fft(),
            self._section_psd(),
            self._section_noise(),
            self._section_filters(),
            self._section_segmentation(),
        ]

        for section in sections:
            scroll_layout.addWidget(section)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _make_section(self, title: str, content: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel(content)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setStyleSheet("font-size: 11px; line-height: 1.4;")
        layout.addWidget(label)
        return group

    def _section_signals(self) -> QGroupBox:
        return self._make_section(
            "1. Основные положения теории сигналов",
            """
            <b>Сигнал</b> — материальный носитель информации, представляемый
            функцией времени y = x(t).<br><br>

            <b>Классификация:</b><br>
            • <b>Детерминированные</b> — полностью определены математически
            (периодические и непериодические)<br>
            • <b>Случайные</b> — значение в любой момент времени является
            случайной величиной (стационарные, нестационарные)<br><br>

            <b>Периодический сигнал:</b> x(t + kT) = x(t), где k — целое число, T — период.<br><br>

            <b>Аналоговый сигнал</b> — непрерывен по времени и амплитуде.<br>
            <b>Дискретный сигнал</b> — определён в дискретные моменты t_n = n·Δt.<br>
            <b>Цифровой сигнал</b> — дискретный + квантованный по амплитуде.
            """,
        )

    def _section_signal_formulas(self) -> QGroupBox:
        return self._make_section(
            "2. Формулы типов сигналов",
            """
            <b>Гармонический сигнал:</b><br>
            x(t) = A·cos(2πft + φ)<br>
            где A — амплитуда, f — частота (Гц), φ — начальная фаза (рад).<br><br>

            <b>Полигармонический сигнал (ряд Фурье):</b><br>
            x(t) = A₀ + Σ Aₖ·cos(kω*t + φₖ), ω* = 2π/T<br><br>

            <b>Гауссов импульс:</b><br>
            x(t) = A·exp(−(t − t₀)² / (2σ²))<br>
            где t₀ — центр импульса, σ — ширина (СКО).<br><br>

            <b>Пилообразный сигнал:</b><br>
            x(t) = A·(2·((t mod T)/T) − 1)<br>
            где T = 1/f — период.<br><br>

            <b>Прямоугольный (импульсный) сигнал:</b><br>
            x(t) = A при (t mod T_period) ≤ τ, иначе 0<br>
            где τ — длительность импульса, T_period — период повторения.<br><br>

            <b>Экспоненциальный импульс:</b><br>
            x(t) = A·exp(α·(t − t_delay)), при t ≥ t_delay<br>
            где α — коэффициент затухания (α &lt; 0 — затухающий).<br><br>

            <b>Экспоненциальная последовательность:</b><br>
            x(nT) = c<sup>n</sup>, где |c| &lt; 1 для затухания.
            """,
        )

    def _section_discretization(self) -> QGroupBox:
        return self._make_section(
            "3. Дискретизация сигналов",
            """
            <b>Период дискретизации:</b> Δt = 1/f_d<br>
            <b>Частота дискретизации:</b> f_d = 1/Δt (Гц)<br>
            <b>Число отсчётов:</b> N = ⌊T/Δt⌋<br><br>

            <b>Теорема Котельникова (Найквиста):</b><br>
            Δt ≤ 1/(2·f_max)<br>
            Сигнал с максимальной частотой f_max может быть без потерь
            восстановлен из дискретных отсчётов, взятых с частотой
            f_d ≥ 2·f_max.<br><br>

            <b>Частота Найквиста:</b> f_N = f_d/2 = 1/(2·Δt)<br><br>

            <b>АЦП</b> выполняет:<br>
            1. Дискретизацию по времени<br>
            2. Квантование по амплитуде
            """,
        )

    def _section_statistics(self) -> QGroupBox:
        return self._make_section(
            "4. Статистические характеристики",
            """
            <b>Характеристики детерминированных сигналов:</b><br><br>

            Мин./макс. амплитуда:<br>
            A_min = min{x[n]}, A_max = max{x[n]}<br><br>

            Мгновенная мощность: p[n] = x[n]²<br><br>

            <b>Средняя мощность:</b><br>
            P_ср = (1/N)·Σ x[n]² , n = 0..N−1<br><br>

            <b>Энергия сигнала:</b><br>
            E = Σ x[n]² , n = 0..N−1<br><br>

            <hr>
            <b>Характеристики стационарных эргодических сигналов:</b><br><br>

            <b>Математическое ожидание:</b><br>
            M = (1/(N+1))·Σ x(n), n = 0..N<br><br>

            <b>Дисперсия:</b><br>
            D = (1/(N+1))·Σ [x[n] − M]², n = 0..N<br><br>

            <b>СКО (среднее квадратическое отклонение):</b><br>
            σ = √D
            """,
        )

    def _section_correlation(self) -> QGroupBox:
        return self._make_section(
            "5. Корреляционная функция (АКФ)",
            """
            <b>Корреляционная функция</b> показывает степень сходства
            между сигналом и его сдвинутой копией.<br><br>

            <b>Непрерывная КФ:</b><br>
            R(τ) = ∫ x(t)·x(t−τ) dt<br><br>

            <b>КФ эргодического процесса:</b><br>
            R(τ) = lim(T→∞) (1/T)·∫ x(t)·x(t−τ) dt − m²_x<br><br>

            <b>Дискретная АКФ:</b><br>
            R_x(r) = (1/(N−r))·Σ x[n]·x[n+r], n = 0..N−r−1<br>
            r = 0..p (p &lt; N — макс. сдвиг)<br><br>

            Свойства АКФ:<br>
            • R(0) = максимум (полная мощность)<br>
            • R(τ) = R(−τ) — чётная функция<br>
            • |R(τ)| ≤ R(0)
            """,
        )

    def _section_spectrum(self) -> QGroupBox:
        return self._make_section(
            "6. Спектральный анализ",
            """
            <b>Ряд Фурье (комплексная форма):</b><br>
            x(t) = Σ X(k)·exp(jkω*t), ω* = 2π/T<br>
            X(k) = (1/T)·∫ x(t)·exp(−jkω*t) dt<br><br>

            <b>Вещественная форма:</b><br>
            x(t) = A₀ + Σ Aₖ·cos(kω*t + φₖ)<br><br>

            <b>Амплитудный спектр:</b><br>
            |X(k)| = √(Re²[X(k)] + Im²[X(k)])<br><br>

            <b>Фазовый спектр:</b><br>
            φₖ = −arctg(Im[X(k)] / Re[X(k)])<br><br>

            <b>Преобразование Фурье (непериодический сигнал):</b><br>
            X(jω) = ∫ x(t)·exp(−jωt) dt<br>
            x(t) = (1/2π)·∫ X(jω)·exp(jωt) dω<br><br>

            <b>АЧХ:</b> |X(jω)| — амплитудно-частотная хар-ка<br>
            <b>ФЧХ:</b> φ(jω) — фазо-частотная хар-ка
            """,
        )

    def _section_dft_fft(self) -> QGroupBox:
        return self._make_section(
            "7. ДПФ и БПФ",
            """
            <b>Дискретное преобразование Фурье (ДПФ):</b><br><br>

            <b>Прямое ДПФ:</b><br>
            X(k) = Σ x(n)·exp(−j·2πnk/N), n = 0..N−1<br><br>

            <b>Обратное ДПФ:</b><br>
            x(n) = (1/N)·Σ X(k)·exp(j·2πnk/N), k = 0..N−1<br><br>

            Частоты: ωₖ = 2πk/N, fₖ = k/(N·Δt)<br><br>

            <b>Свойства спектра дискретного сигнала:</b><br>
            1. Непрерывность<br>
            2. Периодичность (период = f_d)<br>
            3. Четность модуля, нечётность аргумента<br>
            4. Линейность<br><br>

            <b>Равенство Парсеваля:</b><br>
            Σ|x(n)|² = (Δt/2π)·∫|X(e^jωΔt)| dω<br><br>

            <b>БПФ</b> — алгоритмически оптимизированная версия ДПФ,
            даёт идентичный результат, но за O(N·log N) операций
            вместо O(N²).
            """,
        )

    def _section_psd(self) -> QGroupBox:
        return self._make_section(
            "8. Спектральная плотность мощности (СПМ)",
            """
            <b>СПМ (метод 1 — периодограмма):</b><br>
            G(k) = (1/N)·|X(k)|², k = 0..N−1<br>
            где X(k) — ДПФ реализации сигнала.<br><br>

            <b>Метод Бартлетта:</b><br>
            Сигнал разбивается на M сегментов длиной N/M.<br>
            G(k) = (1/M)·Σ Gⁱ(k), i = 1..M<br><br>

            <b>Метод Уэлча:</b><br>
            Усовершенствование: оконные функции + перекрывающиеся
            фрагменты.<br><br>

            <b>Соотношение Винера-Хинчина:</b><br>
            G(jω) = ∫ R_x(τ)·exp(−jωτ) dτ<br>
            R_x(τ) = (1/2π)·∫ G(jω)·exp(jωτ) dω<br><br>

            СПМ через АКФ (дискретный вариант):<br>
            G(k) = Σ R_x(r)·exp(−j2πrk/N), r = 0..N−1
            """,
        )

    def _section_noise(self) -> QGroupBox:
        return self._make_section(
            "9. Типы шумов",
            """
            <b>Равномерный шум:</b><br>
            Значения равномерно распределены в [−A, A].<br><br>

            <b>Белый (гауссовский) шум:</b><br>
            Значения ~ N(0, σ²), плоский спектр: G(f) = const.<br><br>

            <b>Розовый шум (1/f):</b><br>
            G(f) ∝ 1/f — спектральная плотность обратно
            пропорциональна частоте.<br><br>

            <b>Красный / Броуновский шум (1/f²):</b><br>
            G(f) ∝ 1/f² — интеграл белого шума (случайное блуждание).<br><br>

            <b>Синий шум (f):</b><br>
            G(f) ∝ f — спектральная плотность растёт с частотой.<br><br>

            <b>Фиолетовый шум (f²):</b><br>
            G(f) ∝ f² — производная белого шума.<br><br>

            <b>Экспоненциальный шум:</b><br>
            Значения с экспоненциальным распределением
            (центрированные, нормализованные).
            """,
        )

    def _section_filters(self) -> QGroupBox:
        return self._make_section(
            "10. Цифровые фильтры",
            """
            <b>Линейная дискретная система (ЛДС):</b><br>
            y(n) = Σ h(k)·x(n−k) — свёртка<br><br>

            <b>Разностное уравнение фильтра:</b><br>
            Σ aₘ·y(nT−mT) = Σ bₖ·x(nT−kT)<br>
            y(n) = −Σ aₘ·y(n−m) + Σ bₖ·x(n−k)<br><br>

            <b>КИХ-фильтр</b> (нерекурсивный): y(n) = Σ bₖ·x(n−k)<br>
            <b>БИХ-фильтр</b> (рекурсивный): зависит и от предыдущих y.<br><br>

            <b>Передаточная функция:</b><br>
            H(z) = Y(z)/X(z) = Σbₖz⁻ᵏ / (1 + Σaₘz⁻ᵐ)<br><br>

            <b>АЧХ фильтра:</b><br>
            |H(e^jωT)| — модуль частотной характеристики<br><br>

            <hr>
            <b>Реализованные фильтры:</b><br><br>

            <b>Фильтр Хеннинга (сглаживающий):</b><br>
            y(n) = 0.25·x(n) + 0.5·x(n−1) + 0.25·x(n−2)<br>
            Коэф.: a₀=0.25, a₁=0.5, a₂=0.25, b₁=0, b₂=0<br><br>

            <b>Параболический (сглаживание 5 точек):</b><br>
            y(n) = (−3·x(n−2) + 12·x(n−1) + 17·x(n) + 12·x(n+1) − 3·x(n+2)) / 35<br><br>

            <b>Фильтр первого порядка:</b><br>
            y(n) = x(n) − r·y(n−1)<br>
            Коэф.: a₀=1, b₁=−r<br><br>

            <b>НЧФ (Low-Pass):</b><br>
            y(n) = x(n) + 2·x(n−1) + x(n−2) + 2r·cos(ωc)·y(n−1) − r²·y(n−2)<br>
            ωc = 2π·fc/fd<br><br>

            <b>ВЧФ (High-Pass):</b><br>
            y(n) = x(n) − 2·x(n−1) + x(n−2) + 2r·cos(ωc)·y(n−1) − r²·y(n−2)<br><br>

            <b>Полосовой (Band-Pass):</b><br>
            y(n) = x(n) − x(n−2) + 2r·cos(ωc)·y(n−1) − r²·y(n−2)<br><br>

            <b>Режекторный (Band-Stop):</b><br>
            y(n) = x(n) − 2·cos(ωc)·x(n−1) + x(n−2) + r·cos(ωc)·y(n−1) − r²·y(n−2)<br><br>

            <b>Устойчивость:</b> все полюсы H(z) внутри единичного круга z-плоскости.
            """,
        )

    def _section_segmentation(self) -> QGroupBox:
        return self._make_section(
            "11. Структурный анализ и сегментация",
            """
            <b>Задача сегментации</b> — разбиение сигнала на стыкующиеся
            фрагменты, каждый из которых соответствует определённой
            фазе процесса.<br><br>

            <b>Два класса сигналов:</b><br>
            1. С повторяющимися характерными параметрами формы (ЭКС)<br>
            2. Шумоподобные с квазистационарными фрагментами (речь)<br><br>

            <b>Алгоритм сегментации по эталонам:</b><br>
            1. Для окна ωₙ вычислить вектор параметров pₙ<br>
            2. j = argmin μ(pₙ, p*_l), 1 ≤ l ≤ L<br>
            3. Сдвинуть окно: n = n+1<br>
            4. Повторять пока n ≤ N−1<br><br>

            <b>Адаптивная сегментация:</b><br>
            Два окна: неподвижное (настроечное) τ* и скользящее τₙ.<br>
            Вычисляется мера расхождения μ(p*, pₙ).<br>
            Пока μ ≤ δ — сигнал квазистационарен.<br>
            При μ &gt; δ — обнаружена граница сегмента.
            """,
        )
