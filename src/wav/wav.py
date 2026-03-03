import struct
import wave

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Чтение WAV файла"""
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        frames = w.readframes(w.getnframes())
        dtype = np.int16 if w.getsampwidth() == 2 else np.int32
        data = np.frombuffer(frames, dtype=dtype).astype(np.float64)
        data /= np.iinfo(dtype).max
    return data, fs


def write_wav(path: str, signal: np.ndarray, fs: int = 44100):
    """Запись WAV файла"""
    data = (signal * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data.tobytes())


def record_audio(duration: float, fs: int = 44100) -> np.ndarray:
    """Запись с микрофона (заглушка - raylib не поддерживает напрямую)"""
    print(f"[Запись {duration}с не поддерживается в raylib]")
    return np.zeros(int(duration * fs))
