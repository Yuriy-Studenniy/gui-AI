# ═══════════════════════════════════════
# ГОРЯЧИЕ КЛАВИШИ — меняйте только здесь
# ═══════════════════════════════════════

# Запись голоса (удерживать)
HOTKEY_RECORD = "right ctrl"

# Остановить озвучку
HOTKEY_STOP = 'f10'

# Переключение PTT / VAD
HOTKEY_MODE = "f4"

# Очистить память диалога
HOTKEY_CLEAR = 'f12'

# Выход из программы
HOTKEY_EXIT = 'ctrl+shift+q'

# Скриншот экрана
HOTKEY_SCREENSHOT = 'f6'

# ═══════════════════════════════════════
# VAD (голосовая активация)
# ═══════════════════════════════════════
VAD_THRESHOLD = 0.5       # Порог уверенности Silero VAD (0.0 — 1.0)
VAD_SILENCE_SEC = 3.0     # Секунд тишины перед отправкой фразы в Whisper
VAD_PREBUFFER_SEC = 1.0   # Секунд предзаписи до начала речи (чтобы не обрезать начало)

# ═══════════════════════════════════════
# DISCORD AUDIO BRIDGE
# ═══════════════════════════════════════
# ═══════════════════════════════════════
# DISCORD AUDIO BRIDGE
# ═══════════════════════════════════════
DISCORD_WAKE_WORDS = ["в", "трактор", "виктория", "эй в", ]
DISCORD_INPUT_DEVICE = "CABLE Output (VB-Audio Virtual Cable)"
DISCORD_OUTPUT_DEVICE = "CABLE Input (VB-Audio Virtual Cable)"
DISCORD_VAD_THRESHOLD = 0.5
DISCORD_SILENCE_SEC = 2.0
DISCORD_ECHO_COOLDOWN_SEC = 1.5
DISCORD_WAKE_WINDOW_SEC = 3.0
DISCORD_WAKE_STEP_SEC = 1.5
DISCORD_WAKE_MIN_SPEECH_RATIO = 0.3
# ═══════════════════════════════════════
# АВТО-РЕПЛИКИ (бот сам заговорит, если молчишь)
# ═══════════════════════════════════════

# Секунд молчания перед тем как бот сам заговорит
AUTO_IDLE_TIMEOUT = 90

# Минимум секунд между авто-репликами (чтобы не задолбала)
AUTO_MIN_INTERVAL = 120

# Шанс что бот сделает скриншот и прокомментирует (0.0 - никогда, 1.0 - всегда)
AUTO_SCREENSHOT_CHANCE = 0.30

