import webbrowser
import subprocess
import time
import pygetwindow as gw
import re
import shutil
import os

ALLOWED_APPS = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "cmd": "cmd.exe",
    "explorer": "explorer.exe",
    "chrome": "chrome.exe",
    "waterfox": "waterfox.exe",
    "discord": "Discord.exe",
    "vscode": "Code.exe",
}

# Глобальный кэш последнего списка окон
_last_window_list = []

def open_app(name: str) -> str:
    """Открывает приложение по имени (поиск в PATH) или по полному пути."""
    if name.startswith("http://") or name.startswith("https://"):
        webbrowser.open(name)
        return f"Открыла ссылку: {name}"
    
    # 1. Если путь абсолютный — открываем напрямую
    if os.path.exists(name):
        try:
            subprocess.Popen(name, shell=True)
            return f"Открыла: {name}"
        except Exception as e:
            return f"Не удалось открыть {name}: {e}"
    
    # 2. Ищем в системном PATH (через which или where)
    exe_path = shutil.which(name)
    if exe_path:
        try:
            subprocess.Popen(exe_path, shell=True)
            return f"Открыла: {name}"
        except Exception as e:
            return f"Не удалось открыть {name}: {e}"
    
    # 3. Если не найден — пытаемся как есть (может сработать для ассоциированных файлов)
    try:
        subprocess.Popen(name, shell=True)
        return f"Открыла: {name} (через ассоциацию)"
    except Exception as e:
        return f"Не удалось найти или открыть '{name}'. Проверь имя или путь."

def list_windows() -> str:
    """Возвращает список открытых окон и сохраняет его в кэш."""
    global _last_window_list
    windows = gw.getAllTitles()
    windows = [w for w in windows if w.strip()]
    _last_window_list = windows
    if not windows:
        return "Нет открытых окон"
    return "Открытые окна:\n" + "\n".join(f"- {w}" for w in windows[:10])

def close_app(name: str) -> str:
    """
    Закрывает окна, в заголовке которых содержится переданное имя.
    Сначала ищет по точному совпадению в текущих окнах,
    затем по частичному совпадению в текущих окнах,
    затем по частичному совпадению в кэшированном списке (если окно уже не найдено).
    """
    if not name:
        return "Не указано имя для закрытия"

    # 1. Ищем все окна, в заголовке которых есть это слово (без учёта регистра)
    all_windows = gw.getAllWindows()
    windows_to_close = []
    for win in all_windows:
        title = win.title
        if title and name.lower() in title.lower():
            windows_to_close.append(win)

    if windows_to_close:
        closed = 0
        for win in windows_to_close:
            try:
                win.close()
                closed += 1
            except:
                pass
        if closed > 0:
            return f"Закрыла {closed} окон, связанных с '{name}'"
        else:
            # Если закрыть не удалось, пробуем taskkill
            exe = ALLOWED_APPS.get(name.lower())
            if exe:
                try:
                    subprocess.run(f"taskkill /IM {exe} /F", shell=True, capture_output=True)
                    return f"Закрыла {name} через taskkill"
                except:
                    pass
            return f"Не удалось закрыть окна с '{name}'"

    # 2. Если окна не найдены, пытаемся найти в кэшированном списке подходящее имя
    global _last_window_list
    if _last_window_list:
        # Ищем лучшее совпадение: точное вхождение или частичное
        best_match = None
        for title in _last_window_list:
            if name.lower() in title.lower():
                best_match = title
                break
        if best_match:
            # Пробуем закрыть по найденному имени
            return close_app(best_match)  # рекурсивно, но с более точным именем

    # 3. Если ничего не помогло
    return f"Не удалось найти окна или процессы с именем '{name}'"

def execute_command(command: str) -> str:
    if command.startswith("open:"):
        return open_app(command[5:].strip())
    elif command.startswith("close:"):
        return close_app(command[6:].strip())
    elif command.startswith("list_windows"):
        return list_windows()
    else:
        return f"Неизвестная команда: {command}"

# ═══════════════════════════════════════
# ЛИЧНЫЙ ДНЕВНИК V
# ═══════════════════════════════════════
DIARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diary.txt")

def diary_read() -> str:
    if not os.path.exists(DIARY_PATH):
        return "(дневник ещё пуст)"
    try:
        with open(DIARY_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content.strip() else "(дневник пуст)"
    except Exception as e:
        return f"(ошибка чтения дневника: {e})"

def diary_write(text: str) -> str:
    try:
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        return "ok"
    except Exception as e:
        print(f"⚠️ diary_write: {e}")
        return f"error: {e}"

def diary_append(text: str) -> str:
    try:
        with open(DIARY_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        return "ok"
    except Exception as e:
        print(f"⚠️ diary_append: {e}")
        return f"error: {e}"

def diary_clear() -> str:
    return diary_write("")