import pyperclip
import pyautogui
import subprocess
import time
import pygetwindow as gw

# Шаг 1: Копируем текст
pyperclip.copy("Тест 123")

# Шаг 2: Открываем Блокнот
subprocess.Popen("notepad.exe", shell=True)
time.sleep(2)  # ждём открытия

# Шаг 3: Находим окно
windows = gw.getWindowsWithTitle('Блокнот')
if windows:
    win = windows[0]
    win.activate()  # или win.minimize(); win.restore()
    time.sleep(0.5)
else:
    print("Окно не найдено")

# Шаг 4: Вставляем
pyautogui.hotkey('ctrl', 'v')