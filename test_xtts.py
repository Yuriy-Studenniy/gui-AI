import os
import time
from TTS.api import TTS
import torch
import pygame

REF = "reference_v.wav"
OUT = "test_output.wav"

if not os.path.exists(REF):
    print(f"❌ Не найден референс: {REF}")
    print("   Положи файл reference_v.wav в папку со скриптом.") 
    exit(1)

print("⏳ Загрузка XTTS v2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)

print("✅ Готово. Пиши текст и жми Enter.")
print("   Команды:")
print("     /exit          — выход")
print("     /file имя.wav  — сменить референс\n")

while True:
    try:
        text = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not text:
        continue
    if text.lower() in ("/exit", "/quit", "exit", "quit"):
        break
    if text.lower().startswith("/file "):
        new_ref = text[6:].strip()
        if os.path.exists(new_ref):
            REF = new_ref
            print(f"📝 Новый референс: {REF}")
        else:
            print(f"⚠️ Файл не найден: {new_ref}")
        continue

    print("🎙️ Генерация...")
    try:
        model.tts_to_file(
            text=text,
            speaker_wav=REF,
            language="ru",
            file_path=OUT
        )
        print("🔊 Воспроизведение...")
        pygame.mixer.music.load(OUT)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

print("👋 Пока.")