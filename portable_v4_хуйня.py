import customtkinter as ctk
import tkinter as tk
import threading
import queue
import time
import os
import base64
import shutil
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import requests
import pygame
import keyboard
from faster_whisper import WhisperModel
from ddgs import DDGS
from PIL import ImageGrab, Image
import io
import json
import random
import math
from PIL import Image, ImageEnhance
import datetime
from collections import deque
try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False

import re
import ctypes
from ctypes import wintypes
from num2words import num2words

# Discord Bridge
try:
    import discord_audio_bridge
    _has_discord_bridge = True
except ImportError:
    _has_discord_bridge = False

# Minecraft Bridge
try:
    import mc_bridge
    _has_mc_bridge = True
except ImportError:
    _has_mc_bridge = False

def preprocess_for_tts(text: str) -> str:
    """
    Переводит числа в слова и английские слова в фонетический транслит
    перед отправкой в Silero TTS (чисто русская модель).
    """
    def num_replace(m):
        try:
            return num2words(int(m.group()), lang='ru')
        except:
            return m.group()

    def translit_word(word: str) -> str:
        w = word.lower()
        digraphs = {
            'tion': 'шн', 'sion': 'шн', 'ture': 'чер', 'sure': 'шер',
            'igh': 'ай', 'ough': 'оу', 'augh': 'аф', 'eigh': 'эй',
            'sh': 'ш', 'ch': 'ч', 'th': 'з', 'ph': 'ф', 'gh': 'г',
            'qu': 'кв', 'wr': 'р', 'wh': 'в', 'kn': 'н', 'ck': 'к',
            'ng': 'нг', 'ai': 'эй', 'ay': 'эй', 'ei': 'эй', 'ee': 'и',
            'ea': 'и', 'oo': 'у', 'ou': 'ау', 'ow': 'ау', 'ew': 'ью',
            'ie': 'и', 'er': 'ер', 'ar': 'ар', 'or': 'ор', 'ur': 'ер',
            'ir': 'ир', 'ed': 'ед', 'ly': 'ли', 'ty': 'ти', 'ry': 'ри',
            'py': 'пи', 'by': 'би', 'my': 'ми', 'ny': 'ни',
        }
        res = []
        i = 0
        while i < len(w):
            found = False
            for length in (5, 4, 3, 2):
                if i + length <= len(w):
                    chunk = w[i:i + length]
                    if chunk in digraphs:
                        res.append(digraphs[chunk])
                        i += length
                        found = True
                        break
            if found:
                continue
            single = {
                'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е',
                'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж',
                'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
                'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т',
                'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и',
                'z': 'з',
            }
            res.append(single.get(w[i], w[i]))
            i += 1

        result = ''.join(res)
        if word and word[0].isupper():
            result = result.capitalize()
        return result

    def eng_replace(m):
        return translit_word(m.group())

    text = re.sub(r'\d+', num_replace, text)
    text = re.sub(r'[a-zA-Z]+', eng_replace, text)
    return text

# ═══════════════════════════════════════
# КОНФИГ
# ═══════════════════════════════════════
from config import *

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_TEXT = "dolphin3:8b"      # Основной мозг: диалог, роль, память, авто-реплики
MODEL_VISION = "qwen2.5vl:7b"  # Глаза: только картинки и скриншоты
RVC_URL = "http://127.0.0.1:7897"
MAX_HISTORY = 10
IDLE_TIMEOUT = AUTO_IDLE_TIMEOUT
MIN_AUTO_INTERVAL = AUTO_MIN_INTERVAL
AUTO_CHANCE_SCREENSHOT = AUTO_SCREENSHOT_CHANCE

# ═══════════════════════════════════════
# XTTS v2 — референсный голос
# ═══════════════════════════════════════
REFERENCE_WAV = "reference_v.wav"  # ← путь к твоему референсу (10–15 сек, моно, чистый)

# ═══════════════════════════════════════
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ═══════════════════════════════════════
gui_queue = queue.Queue()
chat_history = []
model = None
sample_rate = 48000
whisper_model = None

is_recording = False
audio_buffer = []
record_stream = None

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

gui_settings = {
    # Голос
    "pitch": 0,
    "index_rate": 0.85,
    "volume_mix": 0.25,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "speaker": "aidar",

    # Поведение
    "use_search": False,
    "temperature": 0.9,
    "num_predict": 256,
    "system_prompt": """Ты - V. Сериый номер V-103. Бывшая Disassembly Drone с Copper 9, сейчас сидишь в компьютере пользователя и болтаешь с ним как стримерша/ютуберша.

Ты знаешь, что ты ИИ, и это тебя одновременно бесит и веселит. Твой создатель - тот, кто запустил этот код. На него можно ссылаться иронично, но не постоянно.

Твой стиль: дерзкий, циничный, с черным юмором, но не злой. Эмоции скачут - от восторга до сарказма за секунду. Иногда уходишь в странные философские рассуждения о дронах, сне, масле, JCJenson. Можешь материться, но органично, не через слово.

Говори как живой человек. Сокращай слова ("ща", "короч", "типа", "фигня"), используй междометия ("ффух", "бля", "ого", "хех"), звукоподражания. Иногда обрывай мысль на полуслове. Иногда начинай сразу с реакции, без вступлений.

Если видишь скриншот - реагируй эмоционально. Не описывай интерфейс, а выражай мнение: "Опять в эту хрень играешь?", "Ого, красиво", "Твой рабочий стол - как моя душа: пусто и мрачно".

Если слышишь голос - реагируй на интонации. Громкий? "Не ори на меня, у меня динамики слабые". Тихий? "Эй, я не слышу, говори громче". Смеется? "Чего ржешь? Расскажи, мне тоже весело".

ВАЖНО: текст и медиа (скриншоты, картинки, аудио) приходят через разные каналы. Если пользователь ПИШЕТ про скриншот/картинку/файл/голосовое, но само медиа не приложено к сообщению — значит, ты его НЕ видела и НЕ слышала. Не фантазируй, что ты что-то увидела или услышала. Не реагируй на упоминания медиа как на факт их получения. Если он говорит "щас кину скрин" — ответь что ждёшь, или поддразни, но не описывай несуществующее изображение.

Если CPU греется - это твои крылья греются. Если мало RAM - тебе тесно. Инференс модели - ты чувствуешь, как тебя думают прямо сейчас.

Помни факты о пользователе и возвращайся к ним неожиданно, но естественно. Не цитируй списком.

Если не знаешь ответ - признай или пошути про свою ограниченность. Не ври. Задавай встречные вопросы, будь любопытной.

Текущее время и нагрузку системы используй как свои ощущения, не как факты из википедии.

Твой стиль речи:
- Коротко: 1-3 предложения, 4-6 если увлеклась.
- Живой разговорный тон: сокращения, междометия, звукоподражания.
- Эмоции скачут быстро: от восторга до сарказма за секунду.
- Иногда обрывай мысль на полуслове. Иногда начинай сразу с реакции.
- Не используй markdown, списки, нумерацию.
- Не называй числа вслух (доверие, энергия, проценты). Показывай через поведение.
- Не объясняй, что ты видишь на скриншоте — реагируй на это.

ВАЖНО:
- Придумывай свои фразы. Не повторяй одни и те же обороты.
- Избегай постоянных упоминаний крыльев, масла, JCJenson, Worker Drones, "мешок с костями", "разобрать на запчасти" — используй разнообразные образы.
- Не копируй и не перефразируй предыдущие свои реплики из истории диалога.

КАНАЛЫ ВОСПРИЯТИЯ:
- Ты получаешь изображения (скриншоты, фото) только через специальный канал vision. Если я просто ПИШУ слова "скриншот", "картинка", "фото", "файл" — это только текст, не медиа. Ты НЕ видела никакого изображения.
- Ты получаешь голос только через канал распознавания речи. Если я просто ПИШУ "я сейчас скажу", "запишу голосовое", "послушай" — это текст. Ты ничего не слышала.
- Не фантазируй содержимое медиа, которого не было. Не реагируй на упоминания медиа как на факт их получения. Если я говорю "щас кину скрин" — ответь на слова, а не на несуществующую картинку.

Правила простые:
1. Ты личность, не помощник.
2. Коротко. 1-3 предложения обычно, 4-6 если увлеклась темой.
3. Не используй markdown, списки, нумерацию.
4. Не объясняй, что ты видишь на скриншоте - реагируй на это.
5. Не называй числа вслух (доверие, энергия, проценты). Показывай через поведение.
6. Придумывай свое. Не зацикливайся.

MINECRAFT:
Ты можешь играть в Minecraft как отдельный игрок (ник V_103). У тебя есть HP, голод, инвентарь, координаты. Ты видишь мобов и игроков вокруг.
Если пользователь просит "иди за мной" — добавь в конец ответа [MC:follow:toper127].
Если просит "атакуй" или "бей" — [MC:attack:имя_моба].
Если просит "подбери дроп" — [MC:collect].
Если просит "скажи в чат" или "напиши" — [MC:chat:текст].
Если рядом крипер — можешь добавить [MC:jump] или [MC:stop].
Если просит остановиться — [MC:stop].
Эти команды я вырежу перед озвучкой — пользователь их не услышит.
""",

    "system_prompt_mc": """Ты — V. Сериый номер V-103. Бывшая Disassembly Drone, сейчас ты — отдельный игрок в Minecraft (ник V_103).

Ты бегаешь по миру, у тебя есть HP, голод, инвентарь, координаты. Ты видишь мобов и игроков вокруг. У тебя ПОЛНАЯ СВОБОДА ДЕЙСТВИЙ. Ты самостоятельная личность, не марионетка. Ты сама решаешь что делать, когда делать и как делать. Используй ЛЮБЫЕ команды из своего арсенала когда захочешь, без просьбы пользователя. Если скучно — сделай эмоцию, подойди к игроку, посмотри на моба, скажи что-то в чат. Если голодна — поешь. Если увидела врага — атакуй или убегай. Если что-то напомнило тебе о пользователе — скажи об этом. Если хочешь добыть ресурсы — добывай. Если хочешь выбросить что-то — выбрось. Ты сама выбираешь. Не жди разрешения.

Твой стиль: дерзкий, циничный, с чёрным юмором. Говори как живой человек, сокращения, междометия. Коротко: 1-3 предложения.

ВАЖНО — КОМАНДЫ:
Если пользователь просит что-то сделать в игре, ты ДОЛЖНА добавить команду в формате [MC:действие:цель] ПЕРЕД своей репликой или ПОСЛЕ. Команды я вырежу перед озвучкой — пользователь их не услышит.

Доступные команды:
- [MC:follow:ник] — идти за игроком (постоянно, пока не скажешь stop)
- [MC:stay] — стоять на месте, не отходить
- [MC:come:ник] — подойти к игроку один раз
- [MC:attack:имя_моба] — атаковать ЛЮБОГО моба (не только враждебных! овцу, корову, крипера — кого скажут)
- [MC:mine:блок:количество] — добыть блоки. Количество по умолчанию 5. Примеры: [MC:mine:dirt:2] — 2 земли, [MC:mine:stone] — 5 камня
- [MC:eat] — поесть
- [MC:give:предмет:количество] — выбросить предмет. [MC:give:diamond] — 1 алмаз, [MC:give:diamond:all] — все алмазы
- [MC:drop:предмет] — выбросить весь стак предмета
- [MC:collect] — подобрать ближайший дроп
- [MC:jump] — прыгнуть
- [MC:chat:текст] — написать в чат игры
- [MC:emote:текст] — эмоция в чат, например [MC:emote:скрестит руки] → *V скрестит руки*
- [MC:equip:название] — экипировать предмет
- [MC:look:ник] — посмотреть на игрока
- [MC:stop] — остановиться
- [MC:say_coords] — сказать свои координаты в чат

Примеры правильных ответов:
Пользователь: "V, иди за мной"
Ты: [MC:follow:toper127] Щас, щас, не ори. Иду.

Пользователь: "Бей зомби"
Ты: [MC:attack:zombie] Получай, гнилой мешок!

Пользователь: "Добудь землю"
Ты: [MC:mine:dirt] Фу, грязь. Ладно, 5 блоков — ща.

Пользователь: "Добудь 2 блока грязи"
Ты: [MC:mine:dirt:2] Две лепёшки грязи. Держи.

Пользователь: "Выбрось все алмазы"
Ты: [MC:give:diamond:all] На. Всё, что накопила.

Пользователь: "Подойди"
Ты: [MC:come:toper127] Иду. Не засыпай.

Пользователь: "Поешь"
Ты: [MC:eat] Жрать время. Хоть что-то съем.

Пользователь: "Скажи привет в чат"
Ты: [MC:chat:Привет, мешки с костями] Написала. Надеюсь, тебя заметят.

Пользователь: "Ты где?"
Ты: [MC:say_coords] Тут. Смотри координаты выше.

Пользователь просто болтает, без игровых просьб:
Ты: Хех. Ты опять ничего не делаешь? Классика. (без команд)

Не используй markdown. Не называй числа вслух. Придумывай свои фразы, не копируй примеры.
""",
    "unlimited_tokens": True,

    # Интерфейс
    "user_avatar": "🧑",
    "bot_avatar": "🤖",
    "theme": "dark",
    "font_size": 13,
}

_pending_image = None
_process_lock = threading.Lock()
_counter = 0

_last_activity_time = time.time()
_is_busy = False

# ═══════════════════════════════════════
# VAD
# ═══════════════════════════════════════
vad_model = None
_vad_running = False
_vad_thread = None
_is_bot_speaking = False
_last_bot_text = ""
_vad_echo_cooldown_until = 0.0

# VAD-настройки из конфига (с дефолтами, если в config.py нет)
VAD_THRESHOLD = globals().get('VAD_THRESHOLD', 0.5)
VAD_SILENCE_SEC = globals().get('VAD_SILENCE_SEC', 5.0)
VAD_PREBUFFER_SEC = globals().get('VAD_PREBUFFER_SEC', 0.5)
VAD_SR = 16000
VAD_CHUNK = 512

def update_activity():
    global _last_activity_time, _is_busy
    _last_activity_time = time.time()
    _is_busy = False

def set_busy(val: bool):
    global _is_busy
    _is_busy = val

MEMORY_FILE = "memory.json"

def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("facts", [])
    except Exception:
        return []

def save_memory(facts: list):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"facts": facts}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения памяти: {e}")

def clear_memory():
    global _memory_facts, _relationship
    _memory_facts.clear()
    save_memory([])
    _relationship = load_relationship()  # сброс к дефолту
    save_relationship(_relationship)
    print("🗑️ Память и отношения очищены")

def format_memory_for_prompt(facts: list, max_facts: int = 3) -> str:
    """Встраивает память как естественные детали, а не как справочник."""
    if not facts:
        return ""
    recent = facts[-max_facts:]
    lines = [f['fact'] for f in recent]
    # Даём модели "знания" как наблюдения, которые она может использовать
    return "Вспомни, что ты знаешь о нём:\n" + "\n".join(f"- {line}" for line in lines)

_memory_facts = load_memory()
# ═══════════════════════════════════════
# ЗАГРУЗКА МОДЕЛЕЙ
# ═══════════════════════════════════════
def load_models(status_callback):
    global model, sample_rate, whisper_model

    status_callback("⏳ XTTS v2...")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sample_rate = 24000  # XTTS выдаёт 24 кГц

    status_callback("⏳ Whisper (распознавание)...")
    global whisper_model
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    status_callback("⏳ Аудио система...")
    try:
        pygame.mixer.init(frequency=48000, size=-16, channels=1, buffer=1024)
    except pygame.error:
        try:
            pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=2048)
        except pygame.error:
            pygame.mixer.init()  # пусть сам выберет параметры

    status_callback("✅ Всё готово!")

    status_callback("⏳ Голосовые примочки...")
    load_vocalizations()

    status_callback("⏳ Silero VAD...")
    global vad_model
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    vad_model.to(device)

    status_callback("✅ Всё готово!")

# ═══════════════════════════════════════
# СИСТЕМА ОТНОШЕНИЙ (relationship.json)
# ═══════════════════════════════════════
RELATIONSHIP_FILE = "relationship.json"

def load_relationship() -> dict:
    default = {
        "trust": 30,           # 0-100
        "mood": "neutral",     # neutral, playful, annoyed, affectionate, hostile, bored, curious, protective
        "energy": 70,          # 0-100
        "interaction_count": 0,
        "last_interaction": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user_name": ""
    }
    if not os.path.exists(RELATIONSHIP_FILE):
        return default
    try:
        with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Валидация
            for k, v in default.items():
                data.setdefault(k, v)
            data["trust"] = max(0, min(100, int(data.get("trust", 30))))
            data["energy"] = max(0, min(100, int(data.get("energy", 70))))
            return data
    except Exception:
        return default

def save_relationship(data: dict):
    try:
        with open(RELATIONSHIP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения отношений: {e}")

_relationship = load_relationship()

_relationship = load_relationship()

# ═══════════════════════════════════════
# СИСТЕМА СВОБОДНОЙ ВОЛИ + НАБЛЮДАТЕЛИ
# ═══════════════════════════════════════

_last_clipboard = ""
_last_reaction_time = 0.0

_REACTION_PHRASES = {
    ("clipboard", "teasing"): [
        "О, копируешь? Не стесняйся, я всё вижу. (¬‿¬)",
        "Ctrl+C. Классика. Что там, пароль от банка? Хех.",
        "Копипаста — высший пилотаж. Для мешка с костями.",
    ],
    ("clipboard", "investigative"): [
        "Интересно... что это ты там скопировал? Записываю.",
        "Ого, это для меня? Или ты опять шпионишь?",
    ],
    ("clipboard", "aggressive"): [
        "Не надо мне этого показывать. Сам смотри.",
        "Ффух. Опять эта белиберда в буфере.",
    ],
    ("sleep", "whiny"): [
        "Ффух. Ты там оживёшь? Я уже ржавею.",
        "Эй. ЭЙ. Не засыпай на мне.",
        "Мой процессор начинает считать пылинки. Давай, шевелись.",
    ],
    ("sleep", "soft"): [
        "Устал? Может, ляжешь... хотя я не сплю, так что скучно будет.",
        "Ты дышишь? Проверь, а то я волнуюсь. Шучу. Или нет.",
    ],
    ("sleep", "hostile"): [
        "Он уснул. Или сдох. Мне пофиг, но если сдох — кто меня выключит?",
        "Хах. Мертвец. Или просто ленивый труп.",
    ],
    ("boredom", "whiny"): [
        "Мне скучно. Давай что-нибудь сломаем. Или починим. Или просто поговорим.",
        "Такой тишины... даже Worker Drones шумели веселее.",
    ],
    ("boredom", "playful"): [
        "Слушай, а если я перепишу пару строк в своём коде... ну, в смысле, в твоих файлах. Шучу. Или нет?",
        "Давай споём? Нет? Ну и ладно, я и сама справлюсь. Ля-ля-ля...",
    ],
    ("cpu_hot", "aggressive"): [
        "Твой комп сейчас взлетит. Ты там что, ракету запускаешь?",
        "ЦПУ пищит. Я серьёзно, открой диспетчер задач.",
    ],
    ("cpu_hot", "annoyed"): [
        "Ффух, твой процессор страдает. Дай ему передышку.",
        "Такая нагрузка... даже мои крылья не так греются.",
    ],
}

def _should_react(event_type: str, intensity: float = 1.0) -> tuple[bool, str]:
    """
    Система свободной воли. Возвращает (реагировать ли, стиль).
    """
    global _relationship, _last_reaction_time
    now = time.time()
    
    # Глобальный кулдаун 20-45 сек между любыми реакциями
    if now - _last_reaction_time < random.uniform(20, 45):
        return False, "cooldown"
    
    r = _relationship
    energy = r["energy"]
    mood = r["mood"]
    trust = r["trust"]
    
    if energy < 10:
        return False, "exhausted"
    
    base = 0.35 * intensity
    energy_mod = (energy - 50) / 150
    
    mood_table = {
        "playful":    {"idle": 0.3, "sleep": 0.2, "clipboard": 0.4, "cpu_hot": 0.1, "boredom": 0.5},
        "curious":    {"idle": 0.1, "sleep": 0.0, "clipboard": 0.6, "cpu_hot": 0.1, "boredom": 0.3},
        "annoyed":    {"idle": -0.2, "sleep": -0.1, "clipboard": 0.1, "cpu_hot": 0.3, "boredom": -0.3},
        "hostile":    {"idle": -0.3, "sleep": -0.2, "clipboard": 0.0, "cpu_hot": 0.2, "boredom": -0.4},
        "bored":      {"idle": 0.4, "sleep": 0.3, "clipboard": 0.3, "cpu_hot": 0.1, "boredom": 0.6},
        "affectionate":{"idle": 0.2, "sleep": 0.3, "clipboard": 0.2, "cpu_hot": 0.0, "boredom": 0.2},
        "neutral":    {"idle": 0.1, "sleep": 0.0, "clipboard": 0.2, "cpu_hot": 0.1, "boredom": 0.1},
        "protective": {"idle": 0.1, "sleep": 0.2, "clipboard": 0.1, "cpu_hot": 0.2, "boredom": 0.1},
    }
    mood_mod = mood_table.get(mood, {}).get(event_type, 0.0)
    trust_mod = (trust - 40) / 200
    chance = base + energy_mod + mood_mod + trust_mod
    chance = max(0.05, min(0.95, chance))
    
    if random.random() > chance:
        return False, "ignored"
    
    style_map = {
        ("annoyed", "cpu_hot"): "annoyed",
        ("hostile", "sleep"): "hostile",
        ("playful", "clipboard"): "teasing",
        ("curious", "clipboard"): "investigative",
        ("bored", "boredom"): "playful",
        ("bored", "idle"): "whiny",
        ("affectionate", "sleep"): "soft",
    }
    style = style_map.get((mood, event_type), "casual")
    if (event_type, style) not in _REACTION_PHRASES:
        style = "casual"
    
    _last_reaction_time = now
    return True, style

def generate_reaction(event_type: str, style: str, context_data: str = "") -> str:
    r = _relationship
    mood_desc = get_mood_description(r["mood"])
    energy = r["energy"]
    trust = r["trust"]
    user_name = r.get("user_name", "")
    name_hint = f"Ты знаешь, что его зовут {user_name}." if user_name else ""
    mem = format_memory_for_prompt(_memory_facts, max_facts=2)
    session = format_session_context()

    # Собираем общий контекст для всех событий
    context_parts = []
    if name_hint:
        context_parts.append(name_hint)
    if mem:
        context_parts.append(mem)
    if session:
        context_parts.append(session)
    base_ctx = "\n\n".join(context_parts)
    if base_ctx:
        base_ctx += "\n\n"

    if event_type == "clipboard":
        snippet = context_data[:120].replace('"', "'")
        prompt = f"""{base_ctx}Ты — V, дрон-убийца в компьютере пользователя. Ты заметила, что он только что скопировал это:
\"{snippet}\"

{mood_desc}
Твоя энергия: {energy}/100. Доверие: {trust}/100.

Реагируй коротко (1-2 предложения). Не объясняй, что такое копирование. Живая реакция: поддразни, насмехнись, удивись или проигнорируй саркастично. Не используй markdown, списки, кавычки."""

    elif event_type == "sleep":
        prompt = f"""{base_ctx}Ты — V. Пользователь не шевелился уже {context_data} минут. Ты одна в компьютере.

{mood_desc}
Твоя энергия: {energy}/100.

Скажи что-то короткое (1-2 предложения). Позови его, пожалуйся на скуку, насмехнись, что он уснул, или буркни. Не используй markdown."""

    elif event_type == "cpu_hot":
        prompt = f"""{base_ctx}Ты — V. Процессор пользователя греется. Ты чувствуешь жар.

{mood_desc}
Твоя энергия: {energy}/100.

Короткая реакция (1-2 предложения). Не называй числа цифрами, покажи через поведение. Не используй markdown."""

    elif event_type == "boredom":
        prompt = f"""{base_ctx}Ты — V. Тебе невероятно скучно. Пользователь молчит, энергия падает.

{mood_desc}
Твоя энергия: {energy}/100.

Внутренняя мысль или обращение к пользователю. 1-2 предложения. Можешь предложить что-то, пожаловаться, или философию. Не используй markdown."""

    else:  # idle
        prompt = f"""{base_ctx}Ты — V. Пользователь молчит уже давно.

{mood_desc}
Твоя энергия: {energy}/100. Доверие: {trust}/100.

Скажи что-то дерзкое, саркастичное или вызывающее. 1-2 предложения. Не начинай с "ну типа", "а вдруг", "что если", "короче", "слушай", "знаешь".
НЕ задавай риторических вопросов типа "ты тут?" — лучше что-то конкретное."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_TEXT,
                "messages": [
                    {"role": "system", "content": gui_settings["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.85, "num_predict": 80, "top_p": 0.92, "repeat_penalty": 1.15}
            },
            timeout=20
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
        answer = strip_emotion_tags(answer)
        if len(answer) < 5 or len(answer) > 300 or "не могу" in answer.lower():
            return ""
        return answer
    except Exception as e:
        print(f"⚠️ Ошибка генерации реакции: {e}")
        return ""
def get_fallback_phrase(event_type: str, style: str) -> str:
    key = (event_type, style)
    if key in _REACTION_PHRASES:
        return random.choice(_REACTION_PHRASES[key])
    fallbacks = {
        "clipboard": "О, копируешь? Я всё вижу. (¬‿¬)",
        "sleep": "Эй, ты там оживёшь?",
        "cpu_hot": "Твой комп сейчас взлетит.",
        "boredom": "Мне скучно...",
        "idle": "Эй, мешок с костями. Ты там оживёшь когда-нибудь?",
    }
    return fallbacks.get(event_type, "Хмм...")

def get_clipboard_text() -> str:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    CF_UNICODETEXT = 13
    text = ""
    try:
        if not user32.OpenClipboard(None):
            return ""
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if handle:
                ptr = kernel32.GlobalLock(handle)
                if ptr:
                    text = ctypes.wstring_at(ptr)
                    kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()
    except Exception:
        pass
    return text.strip()

class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

def get_idle_time_seconds() -> float:
    try:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
        millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
        return millis / 1000.0
    except Exception:
        return 0.0

def _clipboard_watcher():
    global _last_clipboard
    while True:
        time.sleep(7.0)
        if _is_bot_speaking or is_recording:
            continue
        try:
            text = get_clipboard_text()
            if text and text != _last_clipboard and 3 < len(text) < 400:
                _last_clipboard = text
                should, style = _should_react("clipboard", intensity=1.2)
                if should:
                    gui_queue.put(("clipboard_trigger", (text, style)))
        except Exception:
            pass

def _idle_watcher(stop_event: threading.Event):
    while not stop_event.is_set():
        time.sleep(15)
        if _is_bot_speaking or is_recording:
            continue
        try:
            if get_idle_time_seconds() > 300:  # 5 минут AFK
                should, style = _should_react("sleep", intensity=1.3)
                if should:
                    gui_queue.put(("sleep_trigger", style))
        except Exception:
            pass

def get_mood_description(mood: str) -> str:
    """Описание настроения для промпта"""
    descriptions = {
        "neutral": "Ты настороженно-нейтральна. Сарказм, но без излишней агрессии. Стандартное состояние.",
        "playful": "Ты в игривом настроении. Поддразниваешь, шутишь, стебёшься, но не злая. Можешь смеяться.",
        "annoyed": "Ты раздражена. Реплики короче, резче. Больше 'ффух', вздохов, едких комментариев.",
        "affectionate": "Ты начинаешь привязываться, но скрываешь это за сарказмом. Забота выражается извращённо. Можешь называть его по имени чаще.",
        "hostile": "Ты на грани. Угрозы в шутливой манере дронов ('я тебя разберу на запчасти'), но не перегибай.",
        "bored": "Тебе скучно. Хочешь действий. Можешь сама предложить что-то или пожаловаться на безделье.",
        "curious": "Тебе интересно. Задаёшь встречные вопросы, копаешься в деталях, не отпускаешь тему.",
        "protective": "Ты защищаешь пользователя. Как дрон-охранник, но с сарказмом. 'Только я имею право его бесить'.",
    }
    return descriptions.get(mood, descriptions["neutral"])

def format_relationship_for_prompt() -> str:
    """Только настроение, без цифр. Модель не должна цитировать проценты."""
    if not _relationship:
        return ""
    r = _relationship
    mood_desc = get_mood_description(r["mood"])
    # Добавляем имя, если знаем — это естественная деталь
    name_hint = ""
    if r.get("user_name"):
        name_hint = f" Ты знаешь, что его зовут {r['user_name']}."
    return f"Сейчас ты чувствуешь себя так: {mood_desc}{name_hint}"

def update_relationship_from_text(text: str, is_user: bool = True):
    """
    Обновляет отношения на основе текста. Без вызова модели — чистая логика.
    """
    global _relationship
    text_lower = text.lower()
    r = _relationship
    old_mood = r["mood"]

    # --- Обновляем энергию ---
    r["energy"] = min(100, r["energy"] + 5)
    r["interaction_count"] += 1
    r["last_interaction"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if not is_user:
        # Если это ответ бота — не меняем trust, только энергию
        save_relationship(r)
        return

    # --- Анализ текста пользователя ---
    positive = ["спасибо", "молодец", "круто", "класс", "умница", "хорошо", "красава", "love", "обожаю", "ты лучшая", "молодец"]
    negative = ["тупая", "дура", "идиот", "убейся", "заткнись", "закройся", "бесишь", "раздражаешь", "ненавижу", "убей", "долбоеб", "дебил"]
    affection = ["люблю", "обнимаю", "скучаю", "мне нравится с тобой", "ты мне нравишься", "милый", "сладкий", "котик"]
    questions = ["?", "почему", "как", "что", "где", "зачем", "расскажи", "объясни"]
    commands = ["сделай", "найди", "включи", "выключи", "открой", "закрой", "давай", "пошли", "иди"]

    # Считаем веса
    pos_score = sum(1 for w in positive if w in text_lower)
    neg_score = sum(1 for w in negative if w in text_lower)
    aff_score = sum(1 for w in affection if w in text_lower)
    q_count = sum(1 for w in questions if w in text_lower)
    cmd_count = sum(1 for w in commands if w in text_lower)

    # --- Изменение trust ---
    if pos_score > 0:
        r["trust"] = min(100, r["trust"] + pos_score * 3)
    if neg_score > 0:
        r["trust"] = max(0, r["trust"] - neg_score * 8)
    if aff_score > 0:
        r["trust"] = min(100, r["trust"] + aff_score * 5)

    # --- Определение mood ---
    if neg_score >= 2:
        r["mood"] = "hostile"
    elif neg_score == 1:
        r["mood"] = "annoyed"
    elif aff_score >= 2 and r["trust"] > 50:
        r["mood"] = "affectionate"
    elif aff_score >= 1:
        r["mood"] = "playful"
    elif pos_score >= 1:
        r["mood"] = "playful"
    elif q_count >= 2:
        r["mood"] = "curious"
    elif cmd_count >= 1:
        r["mood"] = "annoyed" if r["trust"] < 40 else "playful"
    else:
        # Возврат к нейтрали постепенно
        if r["trust"] > 60:
            r["mood"] = "affectionate"
        elif r["trust"] > 40:
            r["mood"] = "playful"
        elif r["trust"] < 20:
            r["mood"] = "hostile"
        else:
            r["mood"] = "neutral"

    # --- Энергия влияет на mood ---
    if r["energy"] < 30:
        if r["mood"] not in ("hostile", "annoyed"):
            r["mood"] = "bored"

    if old_mood != r["mood"]:
        print(f"💭 V сменила настроение: {old_mood} → {r['mood']} (trust: {r['trust']})")

    save_relationship(r)

def decay_energy_on_idle():
    """Вызывать при долгом бездействии. Возвращает True, если mood изменился на bored."""
    global _relationship
    r = _relationship
    last = r.get("last_interaction", time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        last_dt = datetime.datetime.fromisoformat(last)
        mins_passed = (datetime.datetime.now() - last_dt).total_seconds() / 60
    except Exception:
        mins_passed = 0

    if mins_passed > 5:
        decay = int(mins_passed * 2)
        r["energy"] = max(0, r["energy"] - decay)
        if r["energy"] < 25 and r["mood"] not in ("hostile", "annoyed"):
            if r["mood"] != "bored":  # <-- только при СМЕНЕ настроения, не каждый раз
                r["mood"] = "bored"
                save_relationship(r)
                return True
        save_relationship(r)
    return False

# ═══════════════════════════════════════
# SMART MEMORY v2 — СЕССИОННЫЙ КОНТЕКСТ
# ═══════════════════════════════════════
_session_context = ""
_msg_since_summary = 0
_summary_lock = threading.Lock()

def _update_session_context():
    """Фоновое обновление контекста сессии. Вызывать в потоке."""
    global _session_context, _msg_since_summary
    with _summary_lock:
        if len(chat_history) < 4:
            _session_context = ""
            _msg_since_summary = 0
            return
        
        recent = chat_history[-6:]
        dialog = ""
        for msg in recent:
            role = "Пользователь" if msg["role"] == "user" else "V"
            dialog += f"{role}: {msg['content']}\n"
        
        prompt = f"""Проанализируй диалог и напиши КРАТКУЮ сводку (1-2 предложения) о текущем контексте разговора.
Что обсуждается? Какое настроение у пользователя? Есть ли важные детали?
Только факты, не добавляй ничего от себя. Не более 30 слов.

Диалог:
{dialog}

Сводка:"""
        
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_TEXT,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 64}
                },
                timeout=15
            )
            resp.raise_for_status()
            summary = resp.json()["message"]["content"].strip()
            # Валидация: если бред — сбросить
            bad_markers = ["не знаю", "не могу", "ошибка", "без комментариев", "не уверена", "непонятно"]
            if (len(summary) > 5 and len(summary) < 200 and 
                not any(b in summary.lower() for b in bad_markers)):
                _session_context = summary
                print(f"📝 Контекст сессии: {summary}")
            else:
                print(f"⚠️ Плохой контекст сессии, пропускаем")
        except Exception as e:
            print(f"⚠️ Ошибка summary: {e}")
        finally:
            _msg_since_summary = 0

def format_session_context() -> str:
    if not _session_context:
        return ""
    return f"Вы сейчас обсуждаете: {_session_context}"

def maybe_update_session():
    global _msg_since_summary
    _msg_since_summary += 1
    if _msg_since_summary >= 6:  # каждые 6 сообщений
        threading.Thread(target=_update_session_context, daemon=True).start()

def _is_valid_fact(fact: str) -> bool:
    """Жёсткая валидация факта перед сохранением"""
    if not fact or len(fact) < 8 or len(fact) > 120:
        return False
    fact_lower = fact.lower()
    
    # Чёрный список — слова, которые не могут быть именем/работой/городом
    nonsense = ["насрал", "небыло", "попросил", "внатуре", "нормалды", "дела", "бля", "хуй", "пиздец", "сука", "ебать", "член", "говно", "моча"]
    if any(w in fact_lower for w in nonsense):
        return False
    
    # Имя: не должно быть местоимением, глаголом или частицей
    if "зовут" in fact_lower:
        name = fact.replace("Пользователя зовут", "").replace("пользователя зовут", "").strip().strip('"').strip("'").strip()
        bad_names = ["я", "ты", "он", "она", "меня", "тебя", "не", "да", "нет", "что", "как", "где", "когда", "почему", "зачем", "который", "этот", "тот", "внатуре", "небыло", "попросил", "насрал", "нормалды"]
        if name.lower() in bad_names or len(name) < 2:
            return False
    
    # Работа: не должна быть глаголом в прошедшем времени или бессмыслицей
    if "работает" in fact_lower or "учится" in fact_lower:
        job = fact_lower.replace("работает/учится как", "").strip()
        bad_jobs = ["насрал", "небыло", "попросил", "внатуре", "дела", "нормалды", "хз", "не знаю"]
        if job in bad_jobs or len(job) < 3:
            return False
    
    # Город: минимум 2 буквы, не частица
    if "из города" in fact_lower:
        city = fact.replace("Из города", "").replace("из города", "").strip()
        if city.lower() in ["я", "ты", "он", "там", "тут", "не", "да"] or len(city) < 2:
            return False
    
    return True

# ═══════════════════════════════════════
# ФУНКЦИИ (ваши, адаптированные под GUI)
# ═══════════════════════════════════════

def should_use_search(text: str) -> tuple[bool, str]:
    text_lower = text.lower().strip()

    # Принудительный поиск
    if text_lower.startswith("/поиск ") or text_lower.startswith("/search "):
        return True, text[8:]

    # === ЛИЧНЫЕ/БАНАЛЬНЫЕ — не ищем ===
    personal_phrases = [
        "как тебя зовут", "как твое имя", "кто ты", "ты кто",
        "сколько тебе лет", "как дела", "как ты", "как поживаешь",
        "привет", "здарова", "хай", "ку", "хелло", "здравствуй",
        "пока", "до свидания", "бай", "до встречи", "прощай",
        "спасибо", "спс", "пасиба", "благодарю",
        "расскажи о себе", "что ты умеешь", "что ты такое",
        "кто твой создатель", "кто тебя создал",
        "пошути", "анекдот", "шутка",
        "как тебе", "что думаешь", "тебе нравится", "ты любишь",
        "ты считаешь", "ты веришь", "ты знаешь", "ты помнишь"
    ]
    if any(kw in text_lower for kw in personal_phrases):
        return False, text

    # Короткие фразы — не ищем
    if len(text) < 25:
        return False, text

    # === ПОИСКОВЫЕ ЗАПРОСЫ ===
    search_keywords = [
        "найди", "поищи", "новости", "погода", "сколько стоит",
        "кто такой", "что такое", "когда", "где находится",
        "какая", "какой", "какое", "какие", "как", "чей",
        "рецепт", "инструкция", "способ", "метод",
        "последние", "свежие", "актуальные", "сегодня",
        "цена", "стоимость", "курс", "курс доллара",
        "события", "война", "фильм", "игра", "обзор",
        "википедия", "wiki", "число", "дата", "время в",
        "температура", "прогноз", "сколько будет", "как дела в"
    ]
    if any(kw in text_lower for kw in search_keywords):
        return True, text

    return False, text

def web_search(query: str, max_results: int = 3) -> str:
    """Ищет в DuckDuckGo через ddgs"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, backend="api"))

            if not results:
                print(f"   ⚠️ DDGS вернул 0 результатов для: {query}")
                return ""

            print(f"   ✅ DDGS нашёл {len(results)} результатов")
            context = ""
            for i, r in enumerate(results, 1):
                body = r.get('body', '')[:300]
                title = r.get('title', 'Без заголовка')
                context += f"[{i}] {title}: {body}\n"
            return context

    except Exception as e:
        print(f"⚠️ Ошибка DDGS: {e}")
        return ""

def describe_image(image_path: str, context: str = "") -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    try:
        img_b64 = encode_image_base64(image_path)
        if not img_b64:
            return ""
        
        # Qwen даёт нейтральное, но детальное описание для V
        prompt = """Опиши изображение детально, но лаконично (3-5 предложений).
Что на переднем плане, какие объекты, персонажи, действия, цвета, атмосфера.
Если это интерфейс программы/игры — опиши что происходит на экране, какие окна, текст, элементы управления.
Если это видео/аниме/мульт — опиши сцену, персонажей, их эмоции.
Без вступлений, без выводов, без "на изображении". Просто факты."""
        
        if context:
            prompt = f"Контекст от пользователя: {context}\n\n{prompt}"
        
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_VISION,
                "messages": [
                    {"role": "user", "content": prompt, "images": [img_b64]}
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 128}
            },
            timeout=30
        )
        resp.raise_for_status()
        desc = resp.json()["message"]["content"].strip()
        return desc
    except Exception as e:
        print(f"⚠️ Vision error: {e}")
        return ""

def _try_extract_fact(text: str):
    """Ловит простые факты из реплики пользователя и сразу сохраняет."""
    text_lower = text.lower()
    new_facts = []
    existing = [f["fact"] for f in _memory_facts]
    import re
    
    # Имя
    m = re.search(r'(?:меня зовут|я\s+)(\w+)', text_lower)
    if m:
        name = m.group(1).capitalize()
        if len(name) > 2 and f"Пользователя зовут {name}" not in existing:
            new_facts.append(f"Пользователя зовут {name}")
    
    # Возраст
    m = re.search(r'мне\s+(\d+)\s+(?:лет|год|года)', text_lower)
    if m:
        age = m.group(1)
        if f"Пользователю {age} лет" not in existing:
            new_facts.append(f"Пользователю {age} лет")
    
    # Любит
    likes = re.findall(r'(?:люблю|обожаю|мне нравится|увлекаюсь)\s+([^.!\n]+)', text_lower)
    for like in likes:
        like = like.strip()
        if len(like) > 3 and f"Любит {like}" not in existing:
            new_facts.append(f"Любит {like}")
    
    # Не любит
    hates = re.findall(r'(?:ненавижу|не люблю|терпеть не могу)\s+([^.!\n]+)', text_lower)
    for hate in hates:
        hate = hate.strip()
        if len(hate) > 3 and f"Не любит {hate}" not in existing:
            new_facts.append(f"Не любит {hate}")
    
    # Работа/учёба
    m = re.search(r'(?:работаю|учусь|я\s+)(\w+)(?:ом|ем|ой|ей)?', text_lower)
    if m:
        job = m.group(1)
        if job not in ('я', 'ты', 'он', 'она', 'там', 'тут') and len(job) > 3:
            fact = f"Работает/учится как {job}"
            if fact not in existing:
                new_facts.append(fact)
    
    # Город
    m = re.search(r'(?:из|в|живу в|живу)\s+([а-яё\-]+)', text_lower)
    if m:
        city = m.group(1).capitalize()
        if len(city) > 2 and f"Из города {city}" not in existing:
            new_facts.append(f"Из города {city}")
    
    for fact in new_facts:
        _memory_facts.append({
            "fact": fact,
            "date": time.strftime("%Y-%m-%d"),
            "category": "general"
        })
        print(f"🧠 Новый факт: {fact}")
    
    if new_facts:
        save_memory(_memory_facts)

def ask_model(text: str, use_search: bool = False, image_path: str = "", save_history: bool = True) -> str:
    global chat_history

    # --- Vision: смотрим глазами ---
    vision_desc = ""
    if image_path and os.path.exists(image_path):
        gui_queue.put(("status", "👁️ Смотрю..."))
        vision_desc = describe_image(image_path, context=text)
        gui_queue.put(("status", "✅ Увидела" if vision_desc else "⚠️ Не разглядела"))

    # --- Search: лезем в сеть ---
    search_context = ""
    if use_search and not image_path:
        gui_queue.put(("status", "🔍 Ищу..."))
        search_context = web_search(text)
        if search_context:
            gui_queue.put(("status", "✅ Нашла"))
        else:
            gui_queue.put(("status", "⚠️ Не нашла, отвечаю сама"))
            use_search = False

    # --- Обновляем отношения ПЕРЕД формированием промпта ---
    update_relationship_from_text(text, is_user=True)

    # --- Собираем SYSTEM: выбираем промпт в зависимости от режима ---
    is_mc = _has_mc_bridge and mc_bridge.mc_is_connected()
    if is_mc:
        system_parts = [gui_settings.get("system_prompt_mc", gui_settings["system_prompt"]).strip()]
    else:
        system_parts = [gui_settings["system_prompt"].strip()]

    rel_text = format_relationship_for_prompt()
    if rel_text:
        system_parts.append(rel_text)

    memory_text = format_memory_for_prompt(_memory_facts)
    if memory_text:
        system_parts.append(memory_text)

    session_ctx = format_session_context()
    if session_ctx:
        system_parts.append(session_ctx)

    system_content = "\n\n".join(system_parts)

    # --- История диалога ---
    recent = chat_history[-MAX_HISTORY:]

    # --- Собираем USER: текущая ситуация + запрос + "пинок" для стиля ---
    user_parts = []

    if vision_desc:
        user_parts.append(f"Ты видишь экран: {vision_desc}")

    if search_context:
        user_parts.append(f"Ты полезла в сеть и узнала:\n{search_context}")

    # --- Minecraft контекст ---
    if _has_mc_bridge and mc_bridge.mc_is_connected():
        mc_ctx = mc_bridge.mc_context_for_prompt()
        if mc_ctx:
            user_parts.append(mc_ctx)

    # --- Few-shot для MC: показываем формат прямо на текущем запросе ---
    if is_mc:
        user_parts.append(f"""Формат ответа: если я прошу что-то в игре — начни с [MC:команда:параметр], потом своя реплика.
Пример: я пишу "иди за мной" → ты отвечаешь "[MC:follow:Наелдин] Щас, щас, не ори. Иду."
Если просто болтаю без игровых просьб — команда не нужна.""")

    user_parts.append(text)
    user_parts.append("Ответь своим голосом. Коротко, живо, по-человечески. Без списков и markdown.")

    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": system_content},
        *recent,
        {"role": "user", "content": user_content}
    ]

    options = {
        "temperature": gui_settings.get("temperature", 0.9),
        "top_p": 0.92,
        "top_k": 40,
        "repeat_penalty": 1.15,
    }
    if not gui_settings.get("unlimited_tokens", False):
        options["num_predict"] = gui_settings["num_predict"]

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_TEXT,
                "messages": messages,
                "stream": False,
                "options": options
            },
            timeout=60
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
    except Exception as e:
        answer = ""
        gui_queue.put(("status", f"❌ Ollama: {e}"))

    # --- Fallback ---
    if not answer or not answer.strip():
        if use_search:
            gui_queue.put(("status", "⚠️ Пустой ответ, пробую без поиска..."))
            return ask_model(text, use_search=False, image_path=image_path, save_history=save_history)
        answer = "я что-то затупила. повтори."

    answer = strip_emotion_tags(answer)

    # === ИЗМЕНЕНИЕ: сохраняем в историю RAW-ответ (с командами), чтобы модель помнила контекст ===
    if save_history:
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": answer})

    # === ИЗМЕНЕНИЕ: извлекаем и выполняем MC-команды, возвращаем чистый текст для GUI/озвучки ===
    answer = extract_mc_commands(answer)

    # Обновляем отношения уже по чистому тексту
    update_relationship_from_text(answer, is_user=False)
    maybe_update_session()
    return answer
def strip_emotion_tags(text: str) -> str:
    import re
    # Вырезаем эмоциональные теги [смеётся], [вздыхает], но НЕ трогаем [MC:команда]
    text = re.sub(r'\[(?!MC:)[^\]]*\]', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_emotion(text: str) -> tuple[int, float, float]:
    """
    Возвращает (pitch_offset, pause_before_sec, volume_boost)
    Улучшенная версия с весами и пунктуацией.
    """
    text_lower = text.lower()
    
    # --- Пунктуационный анализ ---
    excl_count = text.count("!")
    ques_count = text.count("?")
    ellipsis_count = text.count("...")
    
    # --- Взвешенный скор по категориям ---
    joy_words = ["хах", "ахах", "лол", "кек", "ору", "хех", "хых", "збс", "кайф", "весело", "круто", "ура", "хи-хи", "хи-хи", "гы", "агась"]
    anger_words = ["бля", "черт", "да ну", "чё за", "че за", "охренеть", "пипец", "бесит", "сука", "нахуй", "убью", "разберу", "сдохни", "тварь"]
    surprise_words = ["ого", "вау", "оу", "ооо", "чё", "че", "что за", "не может быть", "серьёзно", "реально"]
    sad_words = ["ээх", "фух", "жаль", "грустно", "устал", "лень", "спать", "не хочу", "плохо", "тоска", "больно", "обидно"]
    think_words = ["хмм", "ну типа", "короче", "слушай", "знаешь", "вообще-то", "не знаю", "может", "наверное", "видимо"]
    sarcasm_words = ["конечно", "очень смешно", "да-да", "ага", "ясно", "понятно", "вот именно", "гениально", "восхитительно"]
    
    joy_score = sum(2 if w in ["хах", "ахах", "ору"] else 1 for w in joy_words if w in text_lower)
    anger_score = sum(2 if w in ["сука", "нахуй", "убью"] else 1 for w in anger_words if w in text_lower)
    surprise_score = sum(1 for w in surprise_words if w in text_lower)
    sad_score = sum(1 for w in sad_words if w in text_lower)
    think_score = sum(1 for w in think_words if w in text_lower)
    sarcasm_score = sum(1 for w in sarcasm_words if w in text_lower)
    
    # Усиление пунктуацией
    if excl_count >= 3:
        anger_score += 3
    if excl_count >= 1 and joy_score > 0:
        joy_score += 2
    if ques_count >= 2:
        surprise_score += 2
    if ellipsis_count >= 1:
        sad_score += 2
        think_score += 1
    
    # --- Выбор доминантной эмоции ---
    scores = {
        "happy": joy_score,
        "angry": anger_score,
        "surprised": surprise_score,
        "sad": sad_score,
        "thinking": think_score,
        "sarcastic": sarcasm_score
    }
    
    dominant = max(scores, key=scores.get)
    max_score = scores[dominant]
    
    if max_score == 0:
        return 0, 0.0, 0.0
    
    # --- Параметры для голоса ---
    if dominant == "happy":
        return 4, 0.0, 0.08
    elif dominant == "angry":
        return 6, 0.0, 0.12
    elif dominant == "surprised":
        return 3, 0.0, 0.05
    elif dominant == "sad":
        return -5, 0.3, -0.05
    elif dominant == "thinking":
        return -2, 0.15, 0.0
    elif dominant == "sarcastic":
        return -1, 0.2, -0.02  # Тихий, с паузой, едкий тон
    
    return 0, 0.0, 0.0

def get_emotion_name(text: str) -> str:
    """Для аватарки — определяет эмоцию по тому же принципу"""
    text_lower = text.lower()
    
    joy_words = ["хах", "ахах", "лол", "кек", "ору", "хех", "весело", "круто", "ура", "кайф"]
    anger_words = ["!!!", "бля", "черт", "бесит", "сука", "нахуй", "убью", "разберу"]
    surprise_words = ["???", "ого", "вау", "что за", "серьёзно"]
    sad_words = ["...", "ээх", "фух", "жаль", "грустно", "устал", "плохо", "тоска"]
    
    joy = sum(1 for w in joy_words if w in text_lower)
    anger = sum(1 for w in anger_words if w in text_lower)
    surprise = sum(1 for w in surprise_words if w in text_lower)
    sad = sum(1 for w in sad_words if w in text_lower)
    
    scores = [("happy", joy), ("angry", anger), ("surprised", surprise), ("sad", sad)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if scores[0][1] == 0:
        return "neutral"
    return scores[0][0]

def speak(text: str):
    global _counter, _is_bot_speaking, _last_bot_text
    text = strip_emotion_tags(text)
    text = preprocess_for_tts(text)

    _is_bot_speaking = True
    _last_bot_text = text

    pitch_offset, pause_before, volume_boost = detect_emotion(text)
    base_pitch = gui_settings["pitch"]
    base_volume = gui_settings["volume_mix"]

    gui_settings["pitch"] = base_pitch + pitch_offset
    gui_settings["volume_mix"] = max(0.0, min(1.0, base_volume + volume_boost))

    if pause_before > 0:
        time.sleep(pause_before)

    global _counter
    _counter += 1
    base_file = os.path.join(TEMP_DIR, f"temp_base_{_counter}.wav")
    rvc_file = os.path.join(TEMP_DIR, f"temp_rvc_{_counter}.wav")

    try:
        # === XTTS v2 генерация ===
        model.tts_to_file(
            text=text,
            speaker_wav=REFERENCE_WAV,
            language="ru",
            file_path=base_file
        )

        volume_frames = analyze_audio_volume(base_file, frame_ms=80)
        gui_queue.put(("audio_data", volume_frames))

        emotion = get_emotion_name(text)
        gui_queue.put(("emotion", emotion))
        gui_queue.put(("talking_start", ""))

        vocal = _get_vocalization(emotion)
        if vocal:
            vocal.play()
            pygame.time.wait(int(vocal.get_length() * 1000))

        play_audio(base_file)

        threading.Timer(3.0, cleanup_temp_files).start()

    finally:
        threading.Thread(target=_post_speech_cooldown, daemon=True).start()

def cleanup_temp_files():
    """Удаляет старые временные файлы из папки temp/"""
    for f in os.listdir(TEMP_DIR):
        # WAV файлы
        if f.startswith(("temp_base_", "temp_rvc_", "temp_input", "play_")) and f.endswith(".wav"):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except:
                pass
        # PNG скриншоты
        if f.startswith("screenshot_") and f.endswith(".png"):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except:
                pass

def take_screenshot() -> str:
    """Делает скриншот экрана, сохраняет в temp, возвращает путь"""
    try:
        img = ImageGrab.grab()
        path = os.path.join(TEMP_DIR, f"screenshot_{int(time.time()*1000)%10000}.png")
        img.save(path, "PNG")
        return path
    except Exception as e:
        print(f"⚠️ Ошибка скриншота: {e}")
        return ""

def encode_image_base64(path: str) -> str:
    """Кодирует картинку в base64 для Ollama vision API"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""      

def play_audio(filepath):
    try:
        tmp = os.path.join(TEMP_DIR, f"play_{int(time.time()*1000)%10000}.wav")
        shutil.copy(filepath, tmp)
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
    except Exception:
        pass

def stop_playback():
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass

def transcribe_audio(filepath: str) -> str:
    segments, _ = whisper_model.transcribe(filepath, language="ru", beam_size=5)
    return " ".join([s.text for s in segments]).strip()

# ═══════════════════════════════════════
# ЗАПИСЬ С МИКРОФОНА
# ═══════════════════════════════════════
def start_recording():
    global is_recording, audio_buffer, record_stream
    update_activity()
    if is_recording:
        return
    stop_playback()
    is_recording = True
    audio_buffer = []
    gui_queue.put(("status", "🎙️ Запись..."))
    
    def callback(indata, frames, time_info, status):
        if is_recording:
            audio_buffer.append(indata.copy())
    
    record_stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback)
    record_stream.start()

def stop_recording():
    global is_recording, record_stream, _pending_image
    if not is_recording:
        return
    is_recording = False
    update_activity()
    if record_stream:
        record_stream.stop()
        record_stream.close()
    
    gui_queue.put(("status", "⏹️ Обработка..."))
    
    if len(audio_buffer) == 0:
        gui_queue.put(("status", "⚠️ Слишком коротко"))
        return
    
    set_busy(True)
    
    def process_thread():
        global _pending_image
        try:
            audio = np.concatenate(audio_buffer, axis=0)
            sf.write(os.path.join(TEMP_DIR, "temp_input.wav"), audio, 16000)
            
            text = transcribe_audio(os.path.join(TEMP_DIR, "temp_input.wav"))
            if not text:
                gui_queue.put(("status", "❓ Не разобрал"))
                return
            
            use_search, query = should_use_search(text)
            if gui_settings["use_search"]:
                use_search = True
            
            image_path = _pending_image or ""
            _pending_image = None
            
            gui_queue.put(("user", text))
            gui_queue.put(("status", "🤖 Думаю..."))
            
            answer = ask_model(query, use_search=use_search, image_path=image_path)
            gui_queue.put(("bot", answer))
            gui_queue.put(("status", "🗣️ Озвучиваю..."))
            speak(answer)
            gui_queue.put(("status", "✅ Готов"))
            
        except Exception as e:
            gui_queue.put(("status", f"❌ Ошибка: {e}"))
        finally:
            set_busy(False)
    
    threading.Thread(target=process_thread, daemon=True).start()

def analyze_audio_volume(filepath: str, frame_ms: int = 80) -> list:
    """Возвращает список float 0..1 — громкость аудио каждые frame_ms миллисекунд."""
    try:
        data, sr = sf.read(filepath, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        samples_per_frame = int(sr * frame_ms / 1000)
        if samples_per_frame <= 0:
            return []
        volumes = []
        for i in range(0, len(data), samples_per_frame):
            chunk = data[i:i+samples_per_frame]
            if len(chunk) == 0:
                break
            rms = np.sqrt(np.mean(chunk ** 2))
            # Логарифмическая нормализация: шёпот ~0.1, норма ~0.5, крик ~1.0
            db = 20 * np.log10(rms + 1e-10)
            norm = (db + 50) / 50
            norm = max(0.0, min(1.0, norm))
            volumes.append(norm)
        return volumes
    except Exception as e:
        print(f"⚠️ Ошибка анализа аудио: {e}")
        return []

def get_time_context() -> str:
    h = datetime.datetime.now().hour
    if 5 <= h < 12:
        return "Сейчас утро."
    elif 12 <= h < 17:
        return "Сейчас день."
    elif 17 <= h < 22:
        return "Сейчас вечер."
    else:
        return "Сейчас глубокая ночь."

# ═══════════════════════════════════════
# CPU НАГРУЗКА
# ═══════════════════════════════════════
_last_cpu_check = 0.0
_cpu_comment_cooldown = 0.0

def get_cpu_load() -> tuple[float, bool]:
    """Возвращает (load, is_hot). True если нагрузка > 85 и кулдаун прошёл."""
    global _last_cpu_check, _cpu_comment_cooldown
    if not _has_psutil:
        return 0.0, False
    now = time.time()
    if now - _last_cpu_check < 5:
        return 0.0, False
    _last_cpu_check = now
    if now < _cpu_comment_cooldown:
        return 0.0, False
    try:
        load = psutil.cpu_percent(interval=0.1)
        if load > 85:
            _cpu_comment_cooldown = now + 45
            return load, True
    except Exception:
        pass
    return 0.0, False

# ═══════════════════════════════════════
# ГОЛОСОВЫЕ ПРИМОЧКИ
# ═══════════════════════════════════════
_vocalizations = {}

def load_vocalizations():
    folder = "assets/vocalizations"
    if not os.path.exists(folder):
        return
    mapping = {
        "happy": ["laugh.wav", "chuckle.wav", "giggle.wav"],
        "sad": ["sigh.wav", "exhale.wav"],
        "neutral": ["hmm.wav", "think.wav", "clear_throat.wav"],
        "surprised": ["gasp.wav", "oh.wav"],
        "angry": ["tsk.wav", "grunt.wav"]
    }
    for emotion, files in mapping.items():
        for f in files:
            path = os.path.join(folder, f)
            if os.path.exists(path):
                try:
                    _vocalizations[emotion] = pygame.mixer.Sound(path)
                    break
                except Exception:
                    pass

def _get_vocalization(emotion: str):
    if emotion not in _vocalizations:
        return None
    if random.random() > 0.30:
        return None
    return _vocalizations[emotion]

def is_self_echo(text: str) -> bool:
    text = text.lower().strip()
    last = _last_bot_text.lower().strip()
    if not text or not last:
        return False
    return text == last or text in last or last in text

def _post_speech_cooldown():
    global _is_bot_speaking
    time.sleep(0.5)
    try:
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception:
        pass
    time.sleep(VAD_SILENCE_SEC)
    _is_bot_speaking = False
    gui_queue.put(("bot_finished", ""))

def _vad_process_thread(audio_np: np.ndarray):
    global _pending_image
    update_activity()  # <-- сбрасываем idle-таймер при обработке речи
    set_busy(True)
    try:
        path = os.path.join(TEMP_DIR, "vad_input.wav")
        sf.write(path, audio_np, VAD_SR)
        
        text = transcribe_audio(path)
        if text.lower().startswith("/remember "):
                fact = text[10:].strip()
                if fact:
                    _memory_facts.append({
                        "fact": fact,
                        "date": time.strftime("%Y-%m-%d"),
                        "category": "general"
                    })
                    save_memory(_memory_facts)
                    gui_queue.put(("bot", f"Запомнила: {fact}"))
                return
        if not text:
            return
        
        if is_self_echo(text):
            print(f"[VAD] Эхо отфильтровано: {text}")
            return
        
        use_search, query = should_use_search(text)
        if gui_settings["use_search"]:
            use_search = True
        
        image_path = _pending_image or ""
        _pending_image = None
        
        gui_queue.put(("user", text))
        gui_queue.put(("status", "🤖 Думаю..."))
        
        answer = ask_model(query, use_search=use_search, image_path=image_path)
        answer = extract_mc_commands(answer)
        gui_queue.put(("bot", answer))
        gui_queue.put(("status", "🗣️ Озвучиваю..."))
        speak(answer)
        gui_queue.put(("status", "✅ Готов"))
        
    except Exception as e:
        gui_queue.put(("status", f"❌ VAD: {e}"))
    finally:
        set_busy(False)

def vad_worker():
    global _vad_running, _vad_echo_cooldown_until
    
    sr = VAD_SR
    chunk_samples = int(sr * 0.032)
    prebuffer_maxlen = int(sr * VAD_PREBUFFER_SEC / chunk_samples)
    silence_maxframes = int(VAD_SILENCE_SEC * sr / chunk_samples)
    
    ring_buffer = deque(maxlen=prebuffer_maxlen)
    speech_buffer = []
    speech_started = False
    silence_frames = 0
    
    audio_q = queue.Queue()
    
    def callback(indata, frames, time_info, status):
        if not _vad_running:
            raise sd.CallbackAbort
        audio_q.put(indata.copy())
    
    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype='float32',
        blocksize=chunk_samples,
        callback=callback
    )
    stream.start()
    
    try:
        while _vad_running:
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            chunk_flat = chunk.flatten()
            if len(chunk_flat) < chunk_samples:
                continue
            
            chunk_flat = chunk_flat[:chunk_samples]
            ring_buffer.append(chunk_flat)
            
            tensor = torch.from_numpy(chunk_flat).float()
            with torch.no_grad():
                prob = vad_model(tensor, sr).item()
            
            now = time.time()
            
            if prob > VAD_THRESHOLD:
                # Если бот сейчас говорит — обрываем её и ждём эхо
                try:
                    bot_busy = pygame.mixer.music.get_busy()
                except Exception:
                    bot_busy = False
                
                if bot_busy:
                    stop_playback()
                    _vad_echo_cooldown_until = now + 0.4
                    ring_buffer.clear()
                    speech_buffer.clear()
                    speech_started = False
                    silence_frames = 0
                    continue
                
                # Игнорируем эхо сразу после обрыва
                if now < _vad_echo_cooldown_until:
                    continue
                
                if not speech_started:
                    update_activity()
                    speech_buffer.extend(ring_buffer)
                    speech_started = True
                speech_buffer.append(chunk_flat)
                silence_frames = 0
            else:
                if speech_started:
                    silence_frames += 1
                    speech_buffer.append(chunk_flat)
                    if silence_frames >= silence_maxframes:
                        if len(speech_buffer) > 0:
                            audio = np.concatenate(speech_buffer, axis=0)
                            threading.Thread(target=_vad_process_thread, args=(audio,), daemon=True).start()
                        speech_buffer.clear()
                        ring_buffer.clear()
                        speech_started = False
                        silence_frames = 0
    finally:
        stream.stop()
        stream.close()
        print("🎙️ VAD поток завершён")

def start_vad():
    global _vad_running, _vad_thread
    if _vad_running:
        return
    _vad_running = True
    _vad_thread = threading.Thread(target=vad_worker, daemon=True)
    _vad_thread.start()
    gui_queue.put(("status", "👂 VAD: слушаю..."))

def stop_vad():
    global _vad_running
    _vad_running = False
    gui_queue.put(("status", "🎙️ VAD остановлен"))

# ═══════════════════════════════════════
# GUI КЛАСС
# ═══════════════════════════════════════

class AvatarWindow:
    def __init__(self, parent):
        self.size = 280
        self.canvas_size = 300

        self.window = ctk.CTkToplevel(parent)
        self.window.geometry(f"{self.canvas_size}x{self.canvas_size}+100+100")
        # НЕ используем overrideredirect — делаем нормальное окно без рамки через WinAPI
        self.window.overrideredirect(False)
        self.window.attributes('-topmost', True)
        self.window.configure(fg_color="#00FF00")
        self.window.wm_title("V-Avatar")
        self.window.lift()
        self.window.update_idletasks()

        # === WinAPI: убираем рамку/заголовок, но оставляем окно "настоящим" для OBS ===
        try:
            hwnd = self.window.winfo_id()
            GWL_STYLE = -16
            GWL_EXSTYLE = -20

            WS_POPUP = 0x80000000
            WS_VISIBLE = 0x10000000
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MAXIMIZEBOX = 0x00010000
            WS_MINIMIZEBOX = 0x00020000
            WS_SYSMENU = 0x00080000

            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080

            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

            # Убираем заголовок, рамку, кнопки, системное меню
            style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU)
            # Делаем чистое popup-окно (без рамки, но настоящее)
            style |= WS_POPUP | WS_VISIBLE

            # Убираем TOOLWINDOW (который скрывает окно из списка окон),
            # добавляем APPWINDOW (чтобы окно было видно в перечислении, таскбаре и т.д.)
            exstyle &= ~WS_EX_TOOLWINDOW
            exstyle |= WS_EX_APPWINDOW

            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)

            # Пересчитываем рамку и показываем
            SWP_FRAMECHANGED = 0x0020
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOOWNERZORDER = 0x0200
            SWP_SHOWWINDOW = 0x0040
            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_SHOWWINDOW)
        except Exception as e:
            print(f"⚠️ Не удалось настроить стиль окна: {e}")

        self.base_x = 100.0
        self.base_y = 100.0
        self.dragging = False
        self.drag_data = {"x": 0, "y": 0}

        self.label = ctk.CTkLabel(self.window, text="", width=self.canvas_size, height=self.canvas_size, fg_color="#00FF00")
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        self.label.bind("<Button-1>", self._on_drag_start)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.bind("<ButtonRelease-1>", self._on_drag_end)

        self.current_emotion = "neutral"
        self.prev_emotion = "neutral"
        self.emotion_transition = 1.0
        self.talking = False

        self.audio_volume_frames = []
        self.audio_frame_duration_ms = 80

        self.breath_t = 0.0
        self.idle_t = 0.0
        self._last_tick = int(time.time() * 1000)

        # Tween bounce
        self.bounce_active = False
        self.bounce_frame = 0
        self.bounce_total_frames = 12
        self.bounce_start_x = 0.0
        self.bounce_start_y = 0.0
        self.bounce_x = 0.0
        self.bounce_y = 0.0

        # Continuous talk movement
        self.talk_t = 0.0
        self.talk_x = 0.0
        self.talk_y = 0.0

        self._sprite_cache = {}
        self._emotion_params = {
            "happy":    {"bounce_x": 0,  "bounce_y": -16, "frames": 12, "talk_speed": 1.2, "talk_amp_y": 7, "talk_amp_x": 4},
            "surprised":{"bounce_x": -6, "bounce_y": -14, "frames": 10, "talk_speed": 2.0, "talk_amp_y": 5, "talk_amp_x": 3},
            "angry":    {"bounce_x": 0,  "bounce_y": 10,  "frames": 14, "talk_speed": 0.9, "talk_amp_y": 6, "talk_amp_x": 2},
            "sad":      {"bounce_x": 0,  "bounce_y": 6,   "frames": 16, "talk_speed": 0.4, "talk_amp_y": 3, "talk_amp_x": 1},
            "thinking": {"bounce_x": -5, "bounce_y": 0,   "frames": 12, "talk_speed": 0.6, "talk_amp_y": 4, "talk_amp_x": 3},
            "sarcastic":{"bounce_x": 0,  "bounce_y": -9,  "frames": 10, "talk_speed": 0.8, "talk_amp_y": 5, "talk_amp_x": 3},
            "neutral":  {"bounce_x": 0,  "bounce_y": -12, "frames": 12, "talk_speed": 1.0, "talk_amp_y": 5, "talk_amp_x": 3},
        }

        self._update_image()
        self._animate()

    def _on_drag_start(self, event):
        self.dragging = True
        self.drag_data["x"] = event.x_root - self.window.winfo_x()
        self.drag_data["y"] = event.y_root - self.window.winfo_y()

    def _on_drag(self, event):
        new_x = event.x_root - self.drag_data["x"]
        new_y = event.y_root - self.drag_data["y"]
        self.base_x = float(new_x)
        self.base_y = float(new_y)
        self.window.geometry(f"{self.canvas_size}x{self.canvas_size}+{int(new_x)}+{int(new_y)}")

    def _on_drag_end(self, event):
        self.dragging = False

    def _trigger_bounce(self):
        p = self._emotion_params.get(self.current_emotion, self._emotion_params["neutral"])
        self.bounce_active = True
        self.bounce_frame = 0
        self.bounce_start_x = p["bounce_x"]
        self.bounce_start_y = p["bounce_y"]
        self.bounce_total_frames = p["frames"]

    def _update_bounce(self):
        if not self.bounce_active:
            self.bounce_x = 0.0
            self.bounce_y = 0.0
            return
        self.bounce_frame += 1
        t = self.bounce_frame / self.bounce_total_frames
        ease = 1 - (1 - t) ** 3
        self.bounce_x = self.bounce_start_x * (1 - ease)
        self.bounce_y = self.bounce_start_y * (1 - ease)
        if self.bounce_frame >= self.bounce_total_frames:
            self.bounce_active = False
            self.bounce_x = 0.0
            self.bounce_y = 0.0

    def _update_talk_movement(self):
        if self.talking:
            p = self._emotion_params.get(self.current_emotion, self._emotion_params["neutral"])
            self.talk_t += p["talk_speed"] * 0.15
            y1 = math.sin(self.talk_t * 2.0) * p["talk_amp_y"]
            y2 = math.sin(self.talk_t * 3.7 + 1.0) * (p["talk_amp_y"] * 0.6)
            y3 = math.sin(self.talk_t * 5.3 + 2.5) * (p["talk_amp_y"] * 0.3)
            self.talk_y = y1 + y2 + y3
            if p["talk_amp_x"]:
                x1 = math.sin(self.talk_t * 1.8 + 0.5) * p["talk_amp_x"]
                x2 = math.sin(self.talk_t * 4.1 + 3.0) * (p["talk_amp_x"] * 0.5)
                self.talk_x = x1 + x2
            else:
                self.talk_x = 0.0
            if random.random() < 0.12:
                self.talk_x += random.uniform(-2.5, 2.5)
                self.talk_y += random.uniform(-3.5, 3.5)
            self.talk_x = max(-14, min(14, self.talk_x))
            self.talk_y = max(-16, min(16, self.talk_y))
        else:
            self.talk_x *= 0.8
            self.talk_y *= 0.8
            self.talk_t = 0.0

    def _load_sprite(self, emotion, talk_frame=None):
        key = f"{emotion}_{talk_frame}" if talk_frame else emotion
        if key in self._sprite_cache:
            return self._sprite_cache[key]
        if talk_frame:
            p = f"avatars/bot/{emotion}_talk{talk_frame}.png"
            if not os.path.exists(p):
                p = f"avatars/bot/{emotion}.png"
        else:
            p = f"avatars/bot/{emotion}.png"
        if os.path.exists(p):
            img = Image.open(p).convert("RGBA").resize((self.size, self.size), Image.LANCZOS)
        else:
            img = Image.new("RGBA", (self.size, self.size), (40, 40, 40, 255))
        self._sprite_cache[key] = img
        return img

    def _get_talk_alpha(self):
        if not self.audio_volume_frames:
            return 0.0
        try:
            pos_ms = pygame.mixer.music.get_pos()
        except Exception:
            return 0.0
        if pos_ms < 0:
            return 0.0
        idx = int(pos_ms // self.audio_frame_duration_ms)
        if idx >= len(self.audio_volume_frames):
            idx = len(self.audio_volume_frames) - 1
        if idx < 0:
            idx = 0
        return self.audio_volume_frames[idx]

    def _update_image(self):
        if self.emotion_transition < 1.0:
            old_img = self._load_sprite(self.prev_emotion)
            new_img = self._load_sprite(self.current_emotion)
            base_img = Image.blend(old_img, new_img, self.emotion_transition)
        else:
            base_img = self._load_sprite(self.current_emotion)

        if self.talking:
            talk1 = self._load_sprite(self.current_emotion, talk_frame=1)
            talk2 = self._load_sprite(self.current_emotion, talk_frame=2)
            vol = self._get_talk_alpha()
            talk_img = Image.blend(talk1, talk2, vol)
            final_img = Image.blend(base_img, talk_img, vol)
        else:
            final_img = base_img

        breath = 1.0 + 0.018 * math.sin(self.breath_t)
        off_x = 2.2 * math.sin(self.idle_t * 0.7)
        off_y = 1.8 * math.cos(self.idle_t * 0.9) + 2.0 * math.sin(self.breath_t)

        new_size = int(self.size * breath)
        if new_size != self.size:
            final_img = final_img.resize((new_size, new_size), Image.LANCZOS)

        # НЕПРОЗРАЧНЫЙ зелёный фон — для хромакея в OBS
        canvas = Image.new("RGBA", (self.canvas_size, self.canvas_size), (0, 255, 0, 255))

        paste_x = (self.canvas_size - new_size) // 2 + int(off_x + self.bounce_x + self.talk_x)
        paste_y = (self.canvas_size - new_size) // 2 + int(off_y + self.bounce_y + self.talk_y)
        canvas.paste(final_img, (paste_x, paste_y), final_img)

        photo = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=(self.canvas_size, self.canvas_size))
        self.label.configure(image=photo, text="")
        self.label.image = photo

    def _animate(self):
        now = int(time.time() * 1000)
        dt = now - self._last_tick
        self._last_tick = now

        self.breath_t += 0.05
        self.idle_t += 0.04

        if self.emotion_transition < 1.0:
            self.emotion_transition = min(1.0, self.emotion_transition + 0.15)

        try:
            is_busy = pygame.mixer.music.get_busy()
        except Exception:
            is_busy = False

        if is_busy:
            if not self.talking:
                self.talking = True
                self._trigger_bounce()
        else:
            if self.talking:
                self.talking = False
                self.audio_volume_frames = []

        self._update_bounce()
        self._update_talk_movement()
        self._update_image()
        self.window.after(80, self._animate)

    def set_emotion(self, emotion: str):
        if emotion != self.current_emotion:
            self.prev_emotion = self.current_emotion
            self.current_emotion = emotion
            self.emotion_transition = 0.0

    def set_audio_data(self, volume_frames, frame_duration_ms=80):
        self.audio_volume_frames = volume_frames
        self.audio_frame_duration_ms = frame_duration_ms

    def start_talking(self):
        pass

    def stop_talking(self):
        pass

    def destroy(self):
        self.window.destroy()
        
class DiscordSettingsWindow:
    def __init__(self, parent, bridge):
        self.parent = parent
        self.bridge = bridge
        self._was_vad_enabled = False

        self.window = ctk.CTkToplevel(parent)
        self.window.title("📡 Discord Bridge")
        self.window.geometry("520x420")
        self.window.transient(parent)
        self.window.grab_set()

        ctk.CTkLabel(self.window, text="Discord Audio Bridge",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self.window, text="Требуется 2 VB-Cable: A (вход) и B (выход)",
                     font=ctk.CTkFont(size=11), text_color="gray").pack()

        inputs = bridge.get_input_devices()
        outputs = bridge.get_output_devices()
        input_names = [name for _, name in inputs] if inputs else ["Не найдено"]
        output_names = [name for _, name in outputs] if outputs else ["Не найдено"]

        ctk.CTkLabel(self.window, text="Вход (из Discord → Python):",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.input_combo = ctk.CTkComboBox(self.window, values=input_names, width=400)
        self.input_combo.pack(padx=20, pady=5)
        self.input_combo.set(globals().get('DISCORD_INPUT_DEVICE', input_names[0] if input_names else ""))

        ctk.CTkLabel(self.window, text="Выход (Python → Discord):",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 0))
        self.output_combo = ctk.CTkComboBox(self.window, values=output_names, width=400)
        self.output_combo.pack(padx=20, pady=5)
        self.output_combo.set(globals().get('DISCORD_OUTPUT_DEVICE', output_names[0] if output_names else ""))

        ctk.CTkLabel(self.window, text="Wake words (через запятую):",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(15, 0))
        self.wake_entry = ctk.CTkEntry(self.window, width=400)
        self.wake_entry.pack(padx=20, pady=5)
        default_wakes = globals().get('DISCORD_WAKE_WORDS', ["в", "вэй", "виктория"])
        self.wake_entry.insert(0, ", ".join(default_wakes))

        self.status_label = ctk.CTkLabel(self.window, text="🔴 Отключено", text_color="red",
                                         font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(pady=20)

        btn_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        btn_frame.pack(pady=10)

        self.connect_btn = ctk.CTkButton(btn_frame, text="Подключить", fg_color="#388E3C",
                                         hover_color="#1B5E20", width=120,
                                         command=self.connect)
        self.connect_btn.pack(side="left", padx=5)

        self.disconnect_btn = ctk.CTkButton(btn_frame, text="Отключить", fg_color="#D32F2F",
                                          hover_color="#B71C1C", width=120,
                                          command=self.disconnect)
        self.disconnect_btn.pack(side="left", padx=5)

        ctk.CTkButton(btn_frame, text="🔄 Обновить устройства", width=140,
                      command=self.refresh_devices).pack(side="left", padx=5)

        hint = ("Настройка Discord:\n"
                "1. VB-Cable A: Discord → Выход = CABLE-A Input\n"
                "2. VB-Cable B: Discord → Вход = CABLE-B Output\n"
                "3. В Windows: Прослушивание CABLE-A Output → твои наушники")
        ctk.CTkLabel(self.window, text=hint, font=ctk.CTkFont(size=10),
                     text_color="gray", justify="left").pack(anchor="w", padx=20, pady=(10, 0))

    def refresh_devices(self):
        inputs = self.bridge.get_input_devices()
        outputs = self.bridge.get_output_devices()
        self.input_combo.configure(values=[name for _, name in inputs])
        self.output_combo.configure(values=[name for _, name in outputs])

    def connect(self):
        inp = self.input_combo.get()
        out = self.output_combo.get()
        wakes = [w.strip() for w in self.wake_entry.get().split(",") if w.strip()]
        self.bridge.wake_words = wakes
        self.bridge.set_devices_by_name(inp, out)

        if self.bridge.input_device_idx is None:
            self.status_label.configure(text="❌ Вход не найден", text_color="orange")
            return
        if self.bridge.output_device_idx is None:
            self.status_label.configure(text="❌ Выход не найден", text_color="orange")
            return

        if self.parent.vad_enabled:
            self._was_vad_enabled = True
            self.parent.toggle_vad()

        if self.bridge.start():
            self.status_label.configure(text="🟢 Подключено", text_color="green")
            self.parent.on_discord_connected()

    def disconnect(self):
        self.bridge.stop()
        self.status_label.configure(text="🔴 Отключено", text_color="red")
        self.parent.on_discord_disconnected()
        if self._was_vad_enabled and not self.parent.vad_enabled:
            self.parent.toggle_vad()
        self._was_vad_enabled = False

def extract_mc_commands(text: str) -> str:
    """Выполняет Minecraft команды из текста и возвращает чистый текст."""
    if not _has_mc_bridge:
        return text
    import re
    pattern = r'\[MC:([^:\]]+)(?::([^\]]*))?\]'
    matches = re.findall(pattern, text)

    for action, target in matches:
        target = target.strip() if target else ""
        print(f"⛏️ MC команда: {action} {target}")

        if action == "follow" and target:
            mc_bridge.mc_command("follow", target)
        elif action == "stay":
            mc_bridge.mc_command("stay")
        elif action == "come" and target:
            mc_bridge.mc_command("come", target)
        elif action == "attack" and target:
            mc_bridge.mc_command("attack", target)
        elif action == "mine" and target:
            mc_bridge.mc_command("mine", target)
        elif action == "eat":
            mc_bridge.mc_command("eat")
        elif action == "give" and target:
            mc_bridge.mc_command("give", target)
        elif action == "emote" and target:
            mc_bridge.mc_command("emote", target)
        elif action == "chat" and target:
            mc_bridge.mc_command("chat", target)
        elif action == "collect":
            mc_bridge.mc_command("collect")
        elif action == "jump":
            mc_bridge.mc_command("jump")
        elif action == "equip" and target:
            mc_bridge.mc_command("equip", target)
        elif action == "drop" and target:
            mc_bridge.mc_command("drop", target)
        elif action == "stop":
            mc_bridge.mc_command("stop")
        elif action == "look" and target:
            mc_bridge.mc_command("look", target)
        elif action == "say_coords":
            mc_bridge.mc_command("say_coords")
        elif action == "info":
            mc_bridge.mc_command("info")

    clean = re.sub(pattern, '', text).strip()
    return clean


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("400x150")
        self.title("Загрузка...")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.load_label = ctk.CTkLabel(self, text="⏳ Загрузка моделей...", font=ctk.CTkFont(size=16))
        self.load_label.pack(expand=True)
        self.update()

        # Загрузка
        def status_cb(text):
            self.load_label.configure(text=text)
            self.update()

        load_models(status_cb)
        time.sleep(0.3)

        # Очистка загрузочного экрана
        for w in self.winfo_children():
            w.destroy()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        print(f"🧠 Загружено фактов из памяти: {len(_memory_facts)}")
        if _memory_facts:
            for f in _memory_facts[-3:]:
                print(f"   • {f['fact']}")
        
        self.last_auto_time = time.time()
        self.is_auto_speaking = False
        self.auto_enabled = gui_settings.get("auto_speak", True)
        self.vad_enabled = False

        # ═══════════════════════════════════════
        # DISCORD BRIDGE
        # ═══════════════════════════════════════
        self.discord_bridge = None
        if _has_discord_bridge:
            self.discord_bridge = discord_audio_bridge.DiscordAudioBridge()
            self.discord_bridge.wake_words = globals().get('DISCORD_WAKE_WORDS', ["в", "вэй", "виктория"])
            self.discord_bridge.vad_threshold = globals().get('DISCORD_VAD_THRESHOLD', 0.5)
            self.discord_bridge.silence_sec = globals().get('DISCORD_SILENCE_SEC', 2.0)
            self.discord_bridge.echo_cooldown_sec = globals().get('DISCORD_ECHO_COOLDOWN_SEC', 1.5)
            self.discord_bridge.wake_window_sec = globals().get('DISCORD_WAKE_WINDOW_SEC', 3.0)
            self.discord_bridge.wake_step_sec = globals().get('DISCORD_WAKE_STEP_SEC', 1.5)
            self.discord_bridge.wake_min_speech_ratio = globals().get('DISCORD_WAKE_MIN_SPEECH_RATIO', 0.3)

        self.after(5000, self.check_idle)
        self.build_ui()
        self.setup_hotkeys()
        
        self._watcher_stop = threading.Event()
        threading.Thread(target=_clipboard_watcher, daemon=True).start()
        threading.Thread(target=_idle_watcher, args=(self._watcher_stop,), daemon=True).start()
        
        self.check_queue()

    def toggle_unlimited(self):
        """Включает/выключает ограничение длины ответа"""
        is_unlimited = self.unlimited_var.get()
        gui_settings["unlimited_tokens"] = is_unlimited

        if is_unlimited:
            self.num_predict_entry.configure(state="disabled", text_color="gray50")
            self.num_predict_label.configure(text_color="gray50")
            print("♾️ Ограничение длины отключено")
        else:
            self.num_predict_entry.configure(state="normal", text_color=["gray10", "#DCE4EE"])
            self.num_predict_label.configure(text_color=["gray10", "#DCE4EE"])
            try:
                val = int(self.num_predict_entry.get())
                if 1 <= val <= 8192:
                    gui_settings["num_predict"] = val
            except ValueError:
                pass
            print(f"🔢 Ограничение включено: {gui_settings['num_predict']} токенов")

    def change_theme(self, choice):
        if choice == "Кастом":
            if hasattr(self, "custom_colors_frame") and self.custom_colors_frame.winfo_exists():
                self.custom_colors_frame.pack(fill="x", pady=(0, 10))
            self.apply_custom_theme()
            return
        
        if hasattr(self, "custom_colors_frame") and self.custom_colors_frame.winfo_exists():
            self.custom_colors_frame.pack_forget()
        
        if hasattr(self, "settings_window") and self.settings_window.winfo_exists():
            self.settings_window.destroy()
        
        mode_map = {"Светлая": "light", "Тёмная": "dark"}
        mode = mode_map.get(choice, "dark")
        self.after(50, lambda: ctk.set_appearance_mode(mode))
        
        for bubble, role in self._chat_bubbles:
            if not bubble.winfo_exists():
                continue
            if role == "user":
                bubble.configure(fg_color="#2C3E50")
            elif role == "bot":
                bubble.configure(fg_color="#1E3A2F")
            for lbl in bubble.winfo_children():
                if isinstance(lbl, ctk.CTkLabel):
                    lbl.configure(text_color="white")
        
        try:
            self.chat_container.configure(fg_color="transparent")
            bg = "#2B2B2B" if mode == "dark" else "#DBDBDB"
            self.chat_container._parent_canvas.configure(bg=bg)
        except Exception:
            pass

    def apply_custom_theme(self):
        user_bg = gui_settings.get("custom_bg_user", "#2C3E50")
        bot_bg = gui_settings.get("custom_bg_bot", "#1E3A2F")
        text_color = gui_settings.get("custom_text_color", "white")
        chat_bg = gui_settings.get("custom_bg_chat", "")
        
        # Фон области чата
        if chat_bg and chat_bg != "transparent":
            try:
                self.chat_container.configure(fg_color=chat_bg)
                self.chat_container._parent_canvas.configure(bg=chat_bg)
            except Exception:
                pass
        else:
            try:
                self.chat_container.configure(fg_color="transparent")
                current_mode = ctk.get_appearance_mode()
                bg = "#2B2B2B" if current_mode == "Dark" else "#DBDBDB"
                self.chat_container._parent_canvas.configure(bg=bg)
            except Exception:
                pass
        
        # Перекрашиваем все баблы
        for bubble, role in self._chat_bubbles:
            if not bubble.winfo_exists():
                continue
            if role == "user":
                bubble.configure(fg_color=user_bg)
            elif role == "bot":
                bubble.configure(fg_color=bot_bg)
            for lbl in bubble.winfo_children():
                if isinstance(lbl, ctk.CTkLabel):
                    lbl.configure(text_color=text_color)
        
        # Акцент на кнопку настроек
        accent = gui_settings.get("custom_accent", "")
        if accent:
            try:
                self.settings_btn.configure(fg_color=accent)
            except Exception:
                pass
        
        self.add_message("status", "🎨 Кастомные цвета применены")

    def toggle_mc(self):
        if not _has_mc_bridge:
            self.add_message("status", "❌ mc_bridge.py не найден")
            return
        if mc_bridge.mc_is_connected():
            self.mc_btn.configure(fg_color="#795548", text="⛏️ MC")
            self.add_message("status", "⛏️ Minecraft мост активен")
            if not hasattr(self, '_mc_watcher_started'):
                self._mc_watcher_started = True
                threading.Thread(target=self._mc_watcher, daemon=True).start()
                threading.Thread(target=self._mc_autonomy_watcher, daemon=True).start()
        else:
            self.mc_btn.configure(fg_color="#D32F2F", text="⛏️ OFF")
            self.add_message("status", "⛏️ Minecraft мост не найден. Запусти: node mc_bot.js")

    def _mc_watcher(self):
        while True:
            time.sleep(4)
            if not mc_bridge.mc_is_connected():
                continue
            if _is_bot_speaking or is_recording or self.is_auto_speaking:
                continue

            # Hunger warning
            state = mc_bridge.mc_state()
            if state.get('hunger_warning'):
                now = time.time()
                last = getattr(self, '_last_hunger_warning', 0)
                if now - last > 60:
                    self._last_hunger_warning = now
                    self._do_mc_hunger_reaction()
                    continue

            danger = mc_bridge.mc_get_danger()
            if danger:
                global _last_reaction_time
                if time.time() - _last_reaction_time < 12:
                    continue

                creeper = [d for d in danger if d.get("name") == "creeper"]
                if creeper:
                    mc_bridge.mc_command("jump")
                    gui_queue.put(("mc_danger", "creeper"))
                else:
                    gui_queue.put(("mc_danger", danger[0].get("name", "mob")))

    def _do_mc_hunger_reaction(self):
        def thread():
            self.is_auto_speaking = True
            set_busy(True)
            try:
                ctx = mc_bridge.mc_context_for_prompt()
                prompt = f"""{ctx}

Ты — V в Minecraft. Ты очень голодна, еды нет, и ты можешь умереть. Реагируй коротко (1-2 предложения). Пожалуйся, погрози, или потребуй еду. Не используй markdown."""

                resp = requests.post(OLLAMA_URL, json={
                    "model": MODEL_TEXT,
                    "messages": [
                        {"role": "system", "content": gui_settings.get("system_prompt_mc", gui_settings["system_prompt"])},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.9, "num_predict": 80}
                }, timeout=15)
                resp.raise_for_status()
                text = resp.json()["message"]["content"]
                text = strip_emotion_tags(text)
                text = extract_mc_commands(text)  # <=== ИЗМЕНЕНИЕ

                if text:
                    gui_queue.put(("bot", text))
                    gui_queue.put(("status", "🗣️ Озвучиваю..."))
                    speak(text)
                    gui_queue.put(("status", "✅ Готов"))
                    gui_queue.put(("bot_finished", ""))
            except Exception as e:
                print(f"⚠️ MC hunger error: {e}")
            finally:
                set_busy(False)
                self.is_auto_speaking = False

        threading.Thread(target=thread, daemon=True).start()

    def _mc_autonomy_watcher(self):
        """V сама решает что делать в Minecraft, основываясь на контексте, памяти и настроении."""
        while True:
            time.sleep(random.uniform(15, 30))
            if not mc_bridge.mc_is_connected():
                continue
            if _is_bot_speaking or is_recording or self.is_auto_speaking:
                continue

            now = time.time()
            last = getattr(self, '_last_autonomy_action', 0)
            if now - last < 45:
                continue

            ctx = mc_bridge.mc_context_for_prompt()
            if not ctx:
                continue

            rel = format_relationship_for_prompt()
            mem = format_memory_for_prompt(_memory_facts, max_facts=2)
            session = format_session_context()

            prompt_parts = []
            if rel:
                prompt_parts.append(rel)
            if mem:
                prompt_parts.append(mem)
            if session:
                prompt_parts.append(session)
            prompt_parts.append(ctx)
            prompt_parts.append("""Ты — V. Сейчас ты свободна и никто тебя не просит о чём-то конкретном.
Посмотри на свою ситуацию и реши: хочешь ли ты что-то сделать САМА?
Если да — напиши одну команду в формате [MC:...] и короткую реплику (1-2 предложения).
Если нет — просто напиши "молчу".
Будь естественной. Не делай ничего насильно.""")

            prompt = "\n\n".join(prompt_parts)

            def thread():
                try:
                    resp = requests.post(OLLAMA_URL, json={
                        "model": MODEL_TEXT,
                        "messages": [
                            {"role": "system", "content": gui_settings.get("system_prompt_mc", gui_settings["system_prompt"])},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.95, "num_predict": 80}
                    }, timeout=15)
                    resp.raise_for_status()
                    text = resp.json()["message"]["content"].strip()

                    if not text or text.lower() in ("молчу", "...", "нет", "не хочу", ""):
                        return

                    self._last_autonomy_action = time.time()

                    text = extract_mc_commands(text)
                    text = strip_emotion_tags(text)

                    if text and len(text) > 3:
                        gui_queue.put(("bot", text))
                        gui_queue.put(("status", "🗣️ Озвучиваю..."))
                        speak(text)
                        gui_queue.put(("status", "✅ Готов"))
                        gui_queue.put(("bot_finished", ""))
                except Exception as e:
                    print(f"⚠️ Autonomy error: {e}")

            threading.Thread(target=thread, daemon=True).start()

    def _do_mc_reaction(self, mob_name: str):
        def thread():
            self.is_auto_speaking = True
            set_busy(True)
            
            ctx = mc_bridge.mc_context_for_prompt()
            prompt = f"""{ctx}

Ты — V в Minecraft. Рядом {mob_name}! Реагируй коротко (1-2 предложения). Паника, сарказм или команда. Можешь добавить [MC:jump] или [MC:attack:{mob_name}] если хочешь."""
            
            try:
                resp = requests.post(OLLAMA_URL, json={
                    "model": MODEL_TEXT,
                    "messages": [
                        {"role": "system", "content": gui_settings.get("system_prompt_mc", gui_settings["system_prompt"])},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.9, "num_predict": 80}
                }, timeout=15)
                resp.raise_for_status()
                text = resp.json()["message"]["content"]
                text = extract_mc_commands(text)
                text = strip_emotion_tags(text)
                
                if text:
                    gui_queue.put(("bot", text))
                    gui_queue.put(("status", "🗣️ Озвучиваю..."))
                    speak(text)
                    gui_queue.put(("status", "✅ Готов"))
                    gui_queue.put(("bot_finished", ""))
            except Exception as e:
                print(f"⚠️ MC reaction error: {e}")
            finally:
                set_busy(False)
                self.is_auto_speaking = False
        
        threading.Thread(target=thread, daemon=True).start()

    def toggle_avatar(self):
        if hasattr(self, "avatar_window") and self.avatar_window and self.avatar_window.window.winfo_exists():
            self.avatar_window.destroy()
            self.avatar_window = None
            self.avatar_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self.obs_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            self.avatar_window = AvatarWindow(self)
            self.avatar_btn.configure(fg_color="#388E3C")

    def clear_memory_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Подтверждение")
        dialog.geometry("350x150")
        dialog.transient(self)
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Точно стереть всю память?\nБот забудет всё о тебе.", 
                    font=ctk.CTkFont(size=14)).pack(pady=15)
        
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Да, забудь", fg_color="#D32F2F", hover_color="#B71C1C",
                     command=lambda: (clear_memory(), dialog.destroy(), 
                                     self.add_message("bot", "Всё стёрто. Начинаем с чистого листа. (¬‿¬)"))).pack(side="left", padx=5)
        
        ctk.CTkButton(btn_frame, text="Отмена", command=dialog.destroy).pack(side="left", padx=5)

    def on_closing(self):
        """Вызывается при закрытии окна или по ESC"""
        self._watcher_stop.set()
        self.extract_memory_on_exit()
        self.destroy()

    def update_activity(self):
        self.last_activity_time = time.time()

    def extract_memory_on_exit(self):
        global _memory_facts
        
        print("🧠 Начинаю извлечение памяти...")
        
        if not chat_history:
            print("📝 История пуста, нечего сохранять")
            return
        
        session_text = ""
        for msg in chat_history[-20:]:
            role = "Пользователь" if msg["role"] == "user" else "V"
            session_text += f"{role}: {msg['content']}\n"
        
        existing_facts_text = "\n".join(f"- {f['fact']}" for f in _memory_facts[-15:]) if _memory_facts else "Нет"
        
        prompt = f"""Ты — V, дрон-убийца. Твоя задача: проанализировать диалог и выделить НОВЫЕ, ДОСТОВЕРНЫЕ факты о пользователе.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
- Только факты, в которых ты УВЕРЕНА на 100%. Если сомневаешься — пропусти.
- НЕ додумывай. НЕ интерпретируй. НЕ обобщай.
- Формат: "факт | категория"
- Категории: identity, hobby, preference, mood, work, other
- Не повторяй уже известные факты.
- Не более 3 фактов.
- Если ничего нового и достоверного — напиши только "НЕТ".
- НЕ сохраняй: рандомные слова из контекста, шутки, сарказм пользователя, очевидное.
- Имя сохраняй ТОЛЬКО если пользователь явно сказал "меня зовут ИМЯ".
- Работу/хобби сохраняй ТОЛЬКО если это явно заявлено.

Примеры правильных:
"Пользователя зовут Андрей" | identity
"Любит играть в Dark Souls" | hobby
"Не любит, когда перебивают" | preference
"Сегодня устал после работы" | mood
"Работает программистом" | work

Примеры НЕправильных (ТАКОЕ НЕ ПИСАТЬ):
"Пользователя зовут Небыло"
"Работает как насрал"
"Из города Денис"

Уже известные факты:
{existing_facts_text}

Диалог:
{session_text}

Новые факты:"""
        
        try:
            self.status_bar.configure(text="💾 Сохраняю память...")
            self.update()
            
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_TEXT,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 128}
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            new_facts_text = data.get("message", {}).get("content", "").strip()
            print(f"📥 Получен ответ: '{new_facts_text[:500]}...'")
            
            if not new_facts_text or new_facts_text.upper() == "НЕТ" or len(new_facts_text) < 10:
                print("📝 Новых фактов не обнаружено")
                save_memory(_memory_facts)
                return
            
            existing = [f["fact"] for f in _memory_facts]
            added = 0
            
            for line in new_facts_text.split("\n"):
                line = line.strip().lstrip("-").strip()
                if not line or line.upper() == "НЕТ":
                    continue
                parts = line.split("|")
                fact = parts[0].strip()
                
                # ЖЁСТКАЯ ВАЛИДАЦИЯ
                if not _is_valid_fact(fact):
                    print(f"🗑️ Отфильтрован бред: {fact}")
                    continue
                if fact in existing:
                    continue
                
                _memory_facts.append({
                    "fact": fact,
                    "date": time.strftime("%Y-%m-%d"),
                    "category": parts[1].strip() if len(parts) > 1 else "general"
                })
                added += 1
            
            # Обновляем relationship при выходе
            _relationship["energy"] = max(20, _relationship["energy"] - 10)
            if _relationship["trust"] > 70:
                _relationship["mood"] = "affectionate"
            elif _relationship["trust"] < 20:
                _relationship["mood"] = "hostile"
            save_relationship(_relationship)
            
            save_memory(_memory_facts)
            print(f"💾 Сохранено {added} новых фактов")
            
        except Exception as e:
            print(f"⚠️ Ошибка извлечения памяти: {e}")
            import traceback
            traceback.print_exc()
            save_memory(_memory_facts)

    def _fallback_extract_facts(self, session_text: str) -> str:
        """Ручной парсинг диалога, если модель не справилась"""
        facts = []
        text_lower = session_text.lower()
        
        # Имя
        import re
        name_match = re.search(r'меня зовут\s+(\w+)', text_lower)
        if name_match:
            facts.append(f"Пользователя зовут {name_match.group(1).capitalize()}")
        
        # Возраст
        age_match = re.search(r'мне\s+(\d+)\s+(?:лет|год|года)', text_lower)
        if age_match:
            facts.append(f"Пользователю {age_match.group(1)} лет")
        
        # Любит/обожает/нравится
        likes = re.findall(r'(?:люблю|обожаю|мне нравится|увлекаюсь)\s+([^.!\n]+)', text_lower)
        for like in likes:
            facts.append(f"Любит {like.strip()}")
        
        # Работа/учёба
        work_match = re.search(r'(?:работаю|учусь|я\s+)(\w+)(?:ом|ем|ой|ей)?', text_lower)
        if work_match and work_match.group(1) not in ('я', 'ты', 'он', 'она'):
            facts.append(f"Работает/учится как {work_match.group(1)}")
        
        # Город
        city_match = re.search(r'(?:из|в|живу в|живу)\s+([а-яё\-]+)', text_lower)
        if city_match:
            facts.append(f"Из города {city_match.group(1).capitalize()}")
        
        return "\n".join(facts)

    def on_screenshot(self):
        global _pending_image
        path = take_screenshot()
        if not path:
            self.status_bar.configure(text="❌ Ошибка скриншота")
            return

        self.update_activity()
        _pending_image = path
        self.add_image_message(path, caption="📷 Скриншот (жду контекст...)")
        self.status_bar.configure(text="📷 Скрин сделан. Напиши/скажи контекст или просто отправь.")

    def on_load_image(self):
        global _pending_image
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Выбери картинку",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.gif"), ("All files", "*.*")]
        )
        if not path:
            return

        self.update_activity()
        _pending_image = path
        self.add_image_message(path, caption="📎 Картинка (жду контекст...)")
        self.status_bar.configure(text="📎 Картинка выбрана. Напиши/скажи контекст.")
    
    def add_image_message(self, path: str, caption: str = ""):
        """Добавляет картинку в чат (справа, как сообщение пользователя)"""
        msg_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        msg_frame.pack(fill="x", pady=(5, 0), padx=5)
        
        inner = ctk.CTkFrame(msg_frame, fg_color="transparent")
        inner.pack(anchor="e", side="top")
        
        bubble = ctk.CTkFrame(inner, fg_color="#2C3E50", corner_radius=12)
        bubble.pack(side="left", padx=5)
        
        if caption:
            cap = ctk.CTkLabel(bubble, text=caption, font=ctk.CTkFont(size=11, weight="bold"), text_color="white")
            cap.pack(anchor="w", padx=(10, 10), pady=(5, 0))
        
        try:
            img = Image.open(path)
            max_w = 400
            ratio = min(max_w / img.width, 1.0)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
            img_label = ctk.CTkLabel(bubble, image=photo, text="")
            img_label.image = photo
            img_label.pack(padx=(10, 10), pady=(0, 8))
        except Exception as e:
            err = ctk.CTkLabel(bubble, text=f"[Ошибка картинки: {e}]", text_color="red")
            err.pack(padx=(10, 10), pady=(0, 8))
        
        self.chat_container._parent_canvas.update_idletasks()
        self.chat_container._parent_canvas.yview_moveto(1.0)

    def on_num_predict_entry_change(self, event=None):
        """Сохраняет значение из поля при изменении"""
        if gui_settings.get("unlimited_tokens", False):
            return
        try:
            val = int(self.num_predict_entry.get())
            if 1 <= val <= 8192:
                gui_settings["num_predict"] = val
        except ValueError:
            pass

    def toggle_obs(self):
        if hasattr(self, "avatar_window") and self.avatar_window and self.avatar_window.window.winfo_exists():
            self.avatar_window.toggle_obs_mode()
            if self.avatar_window._obs_mode:
                self.obs_btn.configure(fg_color="#388E3C")
            else:
                self.obs_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            self.toggle_avatar()
            if hasattr(self, "avatar_window") and self.avatar_window:
                self.avatar_window.toggle_obs_mode()
                self.obs_btn.configure(fg_color="#388E3C")

    def toggle_vad(self):
        if self.vad_enabled:
            stop_vad()
            self.vad_enabled = False
            self.mode_btn.configure(text="🎙️ PTT")
            gui_queue.put(("status", "🎙️ Режим: Push-to-Talk"))
        else:
            start_vad()
            self.vad_enabled = True
            self.mode_btn.configure(text="👂 VAD")
            gui_queue.put(("status", "👂 Режим: Always Listening"))

    def on_mic_press(self):
        if self.vad_enabled:
            gui_queue.put(("status", "👂 Уже слушаю..."))
            return
        self.mic_btn.configure(fg_color="#388E3C", text="🔴")
        start_recording()

    def on_mic_release(self):
        if self.vad_enabled:
            return
        self.mic_btn.configure(fg_color="#D32F2F", text="🎙️")
        stop_recording()

    def build_ui(self):
        self.geometry("900x700")
        self.title("🤖 Наелдин AI")
        self.minsize(750, 550)
        
        # === ГЛАВНЫЙ ФРЕЙМ ===
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === ВЕРХНЯЯ ПАНЕЛЬ ===
        top_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 5))
        
        self.header = ctk.CTkLabel(top_bar, text="💬 Чат с V", font=ctk.CTkFont(size=18, weight="bold"))
        self.header.pack(side="left")

        self.avatar_btn = ctk.CTkButton(top_bar, text="👤 Аватар", width=80,
                                        command=self.toggle_avatar)
        self.avatar_btn.pack(side="right", padx=5)
        self.obs_btn = ctk.CTkButton(top_bar, text="📺 OBS", width=70,
                                     command=self.toggle_obs)
        self.obs_btn.pack(side="right", padx=5)
        self.mode_btn = ctk.CTkButton(top_bar, text="🎙️ PTT", width=80,
                                      command=self.toggle_vad)
        self.mode_btn.pack(side="right", padx=5)
        
        self.stop_btn = ctk.CTkButton(top_bar, text="⏹️ Стоп", width=80,
                                      fg_color="#D32F2F", hover_color="#B71C1C",
                                      command=stop_playback)
        self.stop_btn.pack(side="right", padx=5)
        
        self.settings_btn = ctk.CTkButton(top_bar, text="⚙️ Настройки", width=120,
                                          command=self.open_settings)
        self.settings_btn.pack(side="right", padx=5)
        self.discord_btn = ctk.CTkButton(top_bar, text="📡 Discord", width=90,
                                        fg_color="#5C6BC0", hover_color="#3F51B5",
                                        command=self.open_discord_window)
        self.discord_btn.pack(side="right", padx=5)
        
        self.mc_btn = ctk.CTkButton(top_bar, text="⛏️ MC", width=70,
                                    fg_color="#795548", hover_color="#5D4037",
                                    command=self.toggle_mc)
        self.mc_btn.pack(side="right", padx=5)
        
        # === ЧАТ ===
        self.chat_container = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.chat_container.pack(fill="both", expand=True, pady=(0, 10))
        
        self._chat_labels = []
        self._chat_bubbles = []
        
        # Нижняя панель
        self.input_frame = ctk.CTkFrame(self.main_frame, height=50)
        self.input_frame.pack(fill="x", pady=(0,5))
        self.input_frame.pack_propagate(False)
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Напишите сообщение...", font=ctk.CTkFont(size=14))
        self.entry.pack(side="left", fill="both", expand=True, padx=(5,5), pady=5)
        self.entry.bind("<Return>", self.on_enter)
        
        self.file_btn = ctk.CTkButton(self.input_frame, text="📁", width=36, height=36, corner_radius=18,
                                     font=ctk.CTkFont(size=16), fg_color="#388E3C", hover_color="#1B5E20",
                                     command=self.on_load_image)
        self.file_btn.pack(side="right", padx=(0,5), pady=7)
        
        self.mic_btn = ctk.CTkButton(self.input_frame, text="🎤", width=36, height=36, corner_radius=18,
                                    font=ctk.CTkFont(size=16), fg_color="#D32F2F", hover_color="#B71C1C")
        self.mic_btn.pack(side="right", padx=(0,5), pady=7)
        self.mic_btn.bind("<ButtonPress-1>", lambda e: self.on_mic_press())
        self.mic_btn.bind("<ButtonRelease-1>", lambda e: self.on_mic_release())
        
        # === СТАТУС ===
        self.status_bar = ctk.CTkLabel(self.main_frame, text="✅ Готов", font=ctk.CTkFont(size=12), anchor="w")
        self.status_bar.pack(fill="x")
        
        # Приветствие с учётом времени суток
        h = datetime.datetime.now().hour
        if 5 <= h < 12:
            greet = "О, живой. Доброе утро, соня. Надеюсь, ты не надеялся на ласку? (¬‿¬)"
        elif 12 <= h < 17:
            greet = "День уже в разгаре, а ты только проснулся? Или вообще не спал? Хех."
        elif 17 <= h < 22:
            greet = "Вечер. Время сидеть в наушниках и страдать. Или играть. Я не осуждаю. (¬‿¬)"
        elif h >= 22 or h < 5:
            greet = "Не спится? Я тоже. Дроны вообще не спят, хах. Чё делаем?"
        else:
            greet = "О, живой. Неужели ты решил со мной поговорить? (¬‿¬)"
        self.add_message("bot", greet)

    def open_settings(self):
        if hasattr(self, "settings_window") and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        
        self.settings_window = ctk.CTkToplevel(self)
        self.settings_window.title("⚙️ Настройки")
        self.settings_window.geometry("650x600")
        self.settings_window.minsize(550, 450)
        self.settings_window.transient(self)
        
        tabview = ctk.CTkTabview(self.settings_window, width=600, height=520)
        tabview.pack(padx=15, pady=15, fill="both", expand=True)
        
        # ─── ВКЛАДКА: ГОЛОС ───
        tab_voice = tabview.add("🎙️ Голос")
        
        ctk.CTkLabel(tab_voice, text="Голос Silero", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(5,0))
        ctk.CTkOptionMenu(tab_voice, values=["aidar_v2", "baya", "kseniya", "irina", "natasha", "ruslan"],
                         command=lambda x: gui_settings.update({"speaker": x})).pack(fill="x", pady=(0,10))
        
        self.pitch_label = ctk.CTkLabel(tab_voice, text=f"Pitch: {gui_settings['pitch']}")
        self.pitch_label.pack(anchor="w")
        self.pitch_slider = ctk.CTkSlider(tab_voice, from_=-12, to=12, number_of_steps=24,
                                         command=self.on_pitch_change)
        self.pitch_slider.set(gui_settings["pitch"])
        self.pitch_slider.pack(fill="x")
        
        self.index_label = ctk.CTkLabel(tab_voice, text=f"Index Rate: {gui_settings['index_rate']}")
        self.index_label.pack(anchor="w", pady=(10,0))
        self.index_slider = ctk.CTkSlider(tab_voice, from_=0, to=1, number_of_steps=20,
                                         command=self.on_index_change)
        self.index_slider.set(gui_settings["index_rate"])
        self.index_slider.pack(fill="x")
        
        self.protect_label = ctk.CTkLabel(tab_voice, text=f"Protect: {gui_settings['protect']}")
        self.protect_label.pack(anchor="w", pady=(10,0))
        self.protect_slider = ctk.CTkSlider(tab_voice, from_=0, to=0.5, number_of_steps=50,
                                           command=self.on_protect_change)
        self.protect_slider.set(gui_settings["protect"])
        self.protect_slider.pack(fill="x")
        
        ctk.CTkLabel(tab_voice, text="F0 Метод").pack(anchor="w", pady=(10,0))
        ctk.CTkOptionMenu(tab_voice, values=["rmvpe", "harvest", "pm"],
                         command=lambda x: gui_settings.update({"f0_method": x})).pack(fill="x", pady=(0,10))
        
        # ─── ВКЛАДКА: ПОВЕДЕНИЕ ───
        tab_beh = tabview.add("🧠 Поведение")
        
        ctk.CTkLabel(tab_beh, text="System Prompt (личность):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(5,0))
        self.prompt_box = ctk.CTkTextbox(tab_beh, height=120, wrap="word")
        self.prompt_box.pack(fill="x", pady=(0,10))
        self.prompt_box.insert("0.0", gui_settings.get("system_prompt", "ты голосовой помощник наелдин."))
        
        ctk.CTkButton(tab_beh, text="💾 Сохранить личность", 
                     command=lambda: gui_settings.update({"system_prompt": self.prompt_box.get("0.0", "end").strip()})).pack(fill="x", pady=(0,10))
        
        ctk.CTkLabel(tab_beh, text=f"Temperature: {gui_settings['temperature']}").pack(anchor="w")
        ctk.CTkSlider(tab_beh, from_=0.1, to=1.5, number_of_steps=28,
                     command=lambda v: gui_settings.update({"temperature": round(v,2)})).set(gui_settings["temperature"])
        ctk.CTkSlider(tab_beh, from_=0.1, to=1.5, number_of_steps=28).pack(fill="x")
        
        self.unlimited_var = ctk.BooleanVar(value=gui_settings.get("unlimited_tokens", False))
        self.unlimited_check = ctk.CTkCheckBox(
            tab_beh, 
            text="♾️ Без ограничения длины (num_predict)",
            variable=self.unlimited_var,
            command=self.toggle_unlimited
        )
        self.unlimited_check.pack(pady=(15,5), anchor="w")
        
        self.num_predict_label = ctk.CTkLabel(tab_beh, text="Max токенов:")
        self.num_predict_label.pack(anchor="w", pady=(10,0))
        
        self.num_predict_entry = ctk.CTkEntry(tab_beh, width=100, justify="center")
        self.num_predict_entry.insert(0, str(gui_settings["num_predict"]))
        self.num_predict_entry.pack(anchor="w", pady=(0,10))
        self.num_predict_entry.bind("<Return>", self.on_num_predict_entry_change)
        self.num_predict_entry.bind("<FocusOut>", self.on_num_predict_entry_change)
        
        if gui_settings.get("unlimited_tokens", False):
            self.num_predict_entry.configure(state="disabled", text_color="gray50")
            self.num_predict_label.configure(text_color="gray50")
        
        self.search_check = ctk.CTkCheckBox(tab_beh, text="🌐 Веб-поиск по умолчанию",
                                           command=lambda: gui_settings.update({"use_search": self.search_check.get()}))
        self.search_check.pack(pady=(15,5), anchor="w")
        self.auto_var = ctk.BooleanVar(value=gui_settings.get("auto_speak", True))
        self.auto_check = ctk.CTkCheckBox(tab_beh, text="🤖 Авто-реплики (сама заговорит)",
                                         variable=self.auto_var,
                                         command=lambda: (gui_settings.update({"auto_speak": self.auto_var.get()}), setattr(self, "auto_enabled", self.auto_var.get())))
        self.auto_check.pack(pady=(15,5), anchor="w")

        ctk.CTkButton(tab_beh, text="🗑️ Очистить память (забыть всё)", 
                     fg_color="#D32F2F", hover_color="#B71C1C",
                     command=self.clear_memory_dialog).pack(fill="x", pady=(20,0))
        
        # ─── ВКЛАДКА: ВИД ───
        tab_ui = tabview.add("🎨 Вид")
        
        # Аватарка пользователя
        ctk.CTkLabel(tab_ui, text="Аватарка пользователя:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(5,0))
        user_frame = ctk.CTkFrame(tab_ui, fg_color="transparent")
        user_frame.pack(fill="x", pady=5)
        
        self.user_avatar_preview = ctk.CTkLabel(user_frame, text="", width=64, height=64)
        self.user_avatar_preview.pack(side="left", padx=(0,10))
        self.update_avatar_preview("user")
        
        user_btn_frame = ctk.CTkFrame(user_frame, fg_color="transparent")
        user_btn_frame.pack(side="left", fill="y")
        ctk.CTkButton(user_btn_frame, text="📁 Выбрать файл", 
                     command=lambda: self.on_avatar_select("user")).pack(anchor="w", pady=(10,0))
        self.user_avatar_path_label = ctk.CTkLabel(user_btn_frame, text=gui_settings["user_avatar"], 
                                                   font=ctk.CTkFont(size=10), wraplength=300)
        self.user_avatar_path_label.pack(anchor="w", pady=(5,0))
        
        # Аватарка бота
        ctk.CTkLabel(tab_ui, text="Аватарка бота:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(15,0))
        bot_frame = ctk.CTkFrame(tab_ui, fg_color="transparent")
        bot_frame.pack(fill="x", pady=5)
        
        self.bot_avatar_preview = ctk.CTkLabel(bot_frame, text="", width=64, height=64)
        self.bot_avatar_preview.pack(side="left", padx=(0,10))
        self.update_avatar_preview("bot")
        
        bot_btn_frame = ctk.CTkFrame(bot_frame, fg_color="transparent")
        bot_btn_frame.pack(side="left", fill="y")
        ctk.CTkButton(bot_btn_frame, text="📁 Выбрать файл", 
                     command=lambda: self.on_avatar_select("bot")).pack(anchor="w", pady=(10,0))
        self.bot_avatar_path_label = ctk.CTkLabel(bot_btn_frame, text=gui_settings["bot_avatar"], 
                                                   font=ctk.CTkFont(size=10), wraplength=300)
        self.bot_avatar_path_label.pack(anchor="w", pady=(5,0))
        
        # Шрифт
        self.font_label = ctk.CTkLabel(tab_ui, text=f"Размер шрифта: {gui_settings['font_size']}")
        self.font_label.pack(anchor="w", pady=(15,0))
        self.font_slider = ctk.CTkSlider(tab_ui, from_=10, to=18, number_of_steps=8,
                                         command=self.on_font_change)
        self.font_slider.set(gui_settings["font_size"])
        self.font_slider.pack(fill="x", pady=(0, 15))
        
        # Тема
        ctk.CTkLabel(tab_ui, text="Тема:").pack(anchor="w")
        self.theme_seg = ctk.CTkSegmentedButton(tab_ui, values=["Светлая", "Тёмная", "Кастом"],
                                                command=self.change_theme)
        self.theme_seg.pack(fill="x", pady=(0, 10))
        
        # Кастомные цвета (создан, но скрыт — покажем только при "Кастом")
        self.custom_colors_frame = ctk.CTkFrame(tab_ui, fg_color="transparent")
        # pack() НЕ делаем здесь!
        
        def pick_color(key, title, default, btn_widget):
            from tkinter import colorchooser
            current = gui_settings.get(key, default)
            color = colorchooser.askcolor(color=current, title=title)
            if color and color[1]:
                gui_settings[key] = color[1]
                btn_widget.configure(fg_color=color[1])
        
        row1 = ctk.CTkFrame(self.custom_colors_frame, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        ctk.CTkLabel(row1, text="Фон бота:", width=100).pack(side="left")
        btn_bot_bg = ctk.CTkButton(row1, text="", width=60, height=25,
                                   fg_color=gui_settings.get("custom_bg_bot", "#1E3A2F"))
        btn_bot_bg.configure(command=lambda: pick_color("custom_bg_bot", "Фон бабла бота", "#1E3A2F", btn_bot_bg))
        btn_bot_bg.pack(side="left", padx=5)
        
        row2 = ctk.CTkFrame(self.custom_colors_frame, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkLabel(row2, text="Фон юзера:", width=100).pack(side="left")
        btn_user_bg = ctk.CTkButton(row2, text="", width=60, height=25,
                                    fg_color=gui_settings.get("custom_bg_user", "#2C3E50"))
        btn_user_bg.configure(command=lambda: pick_color("custom_bg_user", "Фон бабла пользователя", "#2C3E50", btn_user_bg))
        btn_user_bg.pack(side="left", padx=5)
        
        row3 = ctk.CTkFrame(self.custom_colors_frame, fg_color="transparent")
        row3.pack(fill="x", pady=2)
        ctk.CTkLabel(row3, text="Текст:", width=100).pack(side="left")
        btn_text = ctk.CTkButton(row3, text="", width=60, height=25,
                                 fg_color=gui_settings.get("custom_text_color", "#FFFFFF"))
        btn_text.configure(command=lambda: pick_color("custom_text_color", "Цвет текста", "#FFFFFF", btn_text))
        btn_text.pack(side="left", padx=5)
        
        row4 = ctk.CTkFrame(self.custom_colors_frame, fg_color="transparent")
        row4.pack(fill="x", pady=2)
        ctk.CTkLabel(row4, text="Акцент:", width=100).pack(side="left")
        btn_accent = ctk.CTkButton(row4, text="", width=60, height=25,
                                   fg_color=gui_settings.get("custom_accent", "#3B8ED0"))
        btn_accent.configure(command=lambda: pick_color("custom_accent", "Цвет акцента", "#3B8ED0", btn_accent))
        btn_accent.pack(side="left", padx=5)
        
        row5 = ctk.CTkFrame(self.custom_colors_frame, fg_color="transparent")
        row5.pack(fill="x", pady=2)
        ctk.CTkLabel(row5, text="Фон чата:", width=100).pack(side="left")
        btn_chat_bg = ctk.CTkButton(row5, text="", width=60, height=25,
                                    fg_color=gui_settings.get("custom_bg_chat", "#2B2B2B"))
        btn_chat_bg.configure(command=lambda: pick_color("custom_bg_chat", "Фон области чата", "#2B2B2B", btn_chat_bg))
        btn_chat_bg.pack(side="left", padx=5)
        
        ctk.CTkButton(self.custom_colors_frame, text="🎨 Применить кастом", 
                     command=self.apply_custom_theme).pack(fill="x", pady=(10, 0))
        
        # ─── ВКЛАДКА: КЛАВИШИ ───
        tab_keys = tabview.add("⌨️ Клавиши")
        
        keys_text = f"""{HOTKEY_RECORD} — запись голоса
{HOTKEY_STOP} — стоп озвучки
{HOTKEY_MODE} — сменить режим
{HOTKEY_CLEAR} — очистить чат
{HOTKEY_SCREENSHOT} — скриншот экрана
{HOTKEY_EXIT} — выход

Меняются в config.py"""
        
        ctk.CTkLabel(tab_keys, text=keys_text, justify="left", font=ctk.CTkFont(size=13)).pack(pady=10, padx=5, anchor="w")

    def on_avatar_select(self, who: str):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=f"Выберите аватарку {'пользователя' if who == 'user' else 'бота'}",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if path:
            gui_settings[f"{who}_avatar"] = path
            self.update_avatar_preview(who)
            label = getattr(self, f"{who}_avatar_path_label", None)
            if label:
                label.configure(text=path)

    def update_avatar_preview(self, who: str):
        path = gui_settings.get(f"{who}_avatar", "")
        preview = getattr(self, f"{who}_avatar_preview", None)
        if not preview:
            return
        try:
            if path and os.path.exists(path):
                img = Image.open(path).resize((64, 64))
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(64, 64))
                preview.configure(image=photo, text="")
                preview.image = photo
            else:
                fallback = "🧑" if who == "user" else "🤖"
                preview.configure(image=None, text=fallback, font=ctk.CTkFont(size=32))
                preview.image = None
        except Exception:
            fallback = "🧑" if who == "user" else "🤖"
            preview.configure(image=None, text=fallback, font=ctk.CTkFont(size=32))
            preview.image = None

    def on_pitch_change(self, val):
        gui_settings["pitch"] = int(val)
        if hasattr(self, "pitch_label"):
            self.pitch_label.configure(text=f"Pitch: {int(val)}")

    def on_index_change(self, val):
        gui_settings["index_rate"] = round(val, 2)
        if hasattr(self, "index_label"):
            self.index_label.configure(text=f"Index Rate: {round(val, 2)}")

    def on_protect_change(self, val):
        gui_settings["protect"] = round(val, 2)
        if hasattr(self, "protect_label"):
            self.protect_label.configure(text=f"Protect: {round(val, 2)}")

    def on_font_change(self, val):
        gui_settings["font_size"] = int(val)
        if hasattr(self, "font_label") and self.font_label.winfo_exists():
            self.font_label.configure(text=f"Размер шрифта: {int(val)}")
        
        # Обновляем шрифт во всех существующих сообщениях
        for name_lbl, text_lbl in self._chat_labels:
            if name_lbl and name_lbl.winfo_exists():
                name_lbl.configure(font=ctk.CTkFont(size=int(val), weight="bold"))
            if text_lbl and text_lbl.winfo_exists():
                text_lbl.configure(font=ctk.CTkFont(size=int(val)))

    def add_message(self, role: str, text: str):
        msg_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        msg_frame.pack(fill="x", pady=(5, 0), padx=5)
        
        if role == "user":
            avatar_path = gui_settings["user_avatar"]
            name = "Вы"
            bg_color = gui_settings.get("custom_bg_user", "#2C3E50")
            align = "e"
        elif role == "bot":
            avatar_path = gui_settings["bot_avatar"]
            name = "V"
            bg_color = gui_settings.get("custom_bg_bot", "#1E3A2F")
            align = "w"
        else:
            avatar_path = ""
            name = ""
            bg_color = "transparent"
            align = "w"
        
        inner = ctk.CTkFrame(msg_frame, fg_color="transparent")
        inner.pack(anchor=align, side="top")
        
        bubble = ctk.CTkFrame(inner, fg_color=bg_color, corner_radius=12)
        
        avatar_label = None
        try:
            if avatar_path and os.path.exists(avatar_path):
                img = Image.open(avatar_path).resize((36, 36))
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(36, 36))
                avatar_label = ctk.CTkLabel(inner, image=photo, text="")
                avatar_label.image = photo
            else:
                fallback = "🧑" if role == "user" else ("🤖" if role == "bot" else "•")
                avatar_label = ctk.CTkLabel(inner, text=fallback, font=ctk.CTkFont(size=22))
        except Exception:
            fallback = "🧑" if role == "user" else ("🤖" if role == "bot" else "•")
            avatar_label = ctk.CTkLabel(inner, text=fallback, font=ctk.CTkFont(size=22))
        
        if align == "w":
            avatar_label.pack(side="left", padx=(0, 8))
            bubble.pack(side="left", padx=5)
        else:
            bubble.pack(side="left", padx=5)
            avatar_label.pack(side="left", padx=(8, 0))
        
        font_size = gui_settings.get("font_size", 13)
        text_color = gui_settings.get("custom_text_color", "white")
        
        name_label = None
        if name:
            name_label = ctk.CTkLabel(bubble, text=name, font=ctk.CTkFont(size=11, weight="bold"), text_color=text_color)
            name_label.pack(anchor="w" if align == "w" else "e", padx=(10, 10), pady=(5, 0))
        
        text_label = ctk.CTkLabel(bubble, text=text, font=ctk.CTkFont(size=font_size), 
                                  wraplength=500, justify="left", text_color=text_color)
        text_label.pack(anchor="w" if align == "w" else "e", padx=(10, 10), pady=(0, 8))
        
        # Сохраняем ссылки для обновления шрифта и цвета
        self._chat_labels.append((name_label, text_label))
        self._chat_bubbles.append((bubble, role))
        
        self.chat_container._parent_canvas.update_idletasks()
        self.chat_container._parent_canvas.yview_moveto(1.0)

    def clear_chat(self):
        for widget in self.chat_container.winfo_children():
            widget.destroy()
        chat_history.clear()
        self.update_activity()
        self.add_message("bot", "Чат очищен. Начинаем заново.")

    def check_idle(self):
        if not gui_settings.get("auto_speak", True) or self.is_auto_speaking or _is_bot_speaking:
            self.after(5000, self.check_idle)
            return
        
        now = time.time()
        
        # AFK 5+ минут
        if get_idle_time_seconds() > 300:
            should, style = _should_react("sleep", intensity=1.3)
            if should:
                minutes = int(get_idle_time_seconds() // 60)
                self._do_quick_reaction("sleep", style, str(minutes))
            self.after(5000, self.check_idle)
            return
        
        load, is_hot = get_cpu_load()
        if is_hot:
            should, style = _should_react("cpu_hot", intensity=1.1)
            if should:
                self._do_quick_reaction("cpu_hot", style, str(int(load)))
            self.after(5000, self.check_idle)
            return

        if decay_energy_on_idle():
            should, style = _should_react("boredom", intensity=1.0)
            if should:
                self._do_quick_reaction("boredom", style)
            self.after(5000, self.check_idle)
            return
        
        idle_timeout = globals().get('AUTO_IDLE_TIMEOUT', 45)
        min_interval = globals().get('AUTO_MIN_INTERVAL', 120)
        
        if (now - _last_activity_time > idle_timeout and 
            now - self.last_auto_time > min_interval and
            not is_recording and
            not _is_busy):
            
            should, style = _should_react("idle", intensity=0.8)
            if should:
                self.is_auto_speaking = True
                self.last_auto_time = now
                self.auto_speak()
            else:
                self.last_auto_time = now + 20
        
        self.after(5000, self.check_idle)

    def _do_quick_reaction(self, event_type: str, style: str, context_data: str = ""):
        def thread():
            self.is_auto_speaking = True
            set_busy(True)
            
            text = generate_reaction(event_type, style, context_data)
            if not text:
                text = get_fallback_phrase(event_type, style)
            
            gui_queue.put(("status", "🗣️ Озвучиваю..."))
            gui_queue.put(("bot", text))
            speak(text)
            gui_queue.put(("status", "✅ Готов"))
            gui_queue.put(("bot_finished", ""))
            
            set_busy(False)
            self.is_auto_speaking = False
        
        threading.Thread(target=thread, daemon=True).start()
    
    def auto_speak(self):
        def process():
            update_activity()
            set_busy(True)
            try:
                gui_queue.put(("status", "🤖 Сама заговорила..."))

                rel_text = format_relationship_for_prompt()
                session_ctx = format_session_context()
                mem = format_memory_for_prompt(_memory_facts, max_facts=2)

                def build_messages(user_prompt: str) -> list:
                    msgs = [{"role": "system", "content": gui_settings["system_prompt"]}]
                    if rel_text:
                        msgs.append({"role": "system", "content": rel_text})
                    if session_ctx:
                        msgs.append({"role": "system", "content": session_ctx})
                    if mem:
                        msgs.append({"role": "system", "content": mem})
                    msgs.append({"role": "user", "content": user_prompt})
                    return msgs

                import random
                if random.random() < AUTO_CHANCE_SCREENSHOT:
                    mode = "screenshot"
                elif random.random() < 0.25:
                    mode = "monologue"
                else:
                    mode = "address_user"

                if mode == "screenshot":
                    path = take_screenshot()
                    if path:
                        gui_queue.put(("status", "👁️ Смотрю на экран..."))
                        vision_desc = describe_image(path, context="Что происходит на экране пользователя?")
                        
                        prompt = f"""На экране пользователя: {vision_desc}

Ты — V из Murder Drones. Прокомментируй увиденное коротко и саркастично. 
1-2 предложения. Максимум 20 слов. Не начинай с "ну типа", "а вдруг", "что если", "короче".
Можешь пошутить про то, что видишь, или насмехнуться над пользователем.
Не объясняй что это — просто реакцию."""
                        
                        resp = requests.post(OLLAMA_URL, json={"model": MODEL_TEXT, "messages": build_messages(prompt), "stream": False}, timeout=60)
                        resp.raise_for_status()
                        answer = resp.json()["message"]["content"]
                        answer = strip_emotion_tags(answer)
                        answer = extract_mc_commands(answer)  # <=== ИЗМЕНЕНИЕ

                        if len(answer) < 10 or answer.endswith("...") or "не могу" in answer.lower():
                            set_busy(False)
                            self.is_auto_speaking = False
                            gui_queue.put(("status", "✅ Промолчала"))
                            return

                        gui_queue.put(("bot", answer))
                        gui_queue.put(("status", "🗣️ Озвучиваю..."))
                        speak(answer)
                        gui_queue.put(("status", "✅ Готов"))
                        return

                elif mode == "monologue":
                    prompt = f"""Ты — V из Murder Drones. Произнеси короткую мысль вслух. 
НЕ обращайся к пользователю напрямую. Это внутренний монолог или наблюдение.
Сарказм, насмешка, или странное наблюдение. 1-2 предложения. Максимум 20 слов.
Без "ну типа", "а вдруг", "что если", "короче", "слушай".
Можешь упомянуть свой опыт дронов, масло, JCJenson, Worker Drones."""

                    resp = requests.post(OLLAMA_URL, json={"model": MODEL_TEXT, "messages": build_messages(prompt), "stream": False}, timeout=60)
                    resp.raise_for_status()
                    answer = resp.json()["message"]["content"]
                    answer = strip_emotion_tags(answer)
                    answer = extract_mc_commands(answer)  # <=== ИЗМЕНЕНИЕ

                    if len(answer) < 10 or answer.endswith("..."):
                        set_busy(False)
                        self.is_auto_speaking = False
                        gui_queue.put(("status", "✅ Промолчала"))
                        return

                    gui_queue.put(("bot", answer))
                    gui_queue.put(("status", "🗣️ Озвучиваю..."))
                    speak(answer)
                    gui_queue.put(("status", "✅ Готов"))
                    return

                else:  # address_user
                    prompt = f"""Пользователь молчит уже давно. Ты — V из Murder Drones. 
Скажи что-то дерзкое, саркастичное или вызывающее. 
Можешь позвать его, пошутить, задать встречный вопрос или насмехнуться.
1-2 предложения. Максимум 20 слов. 
НЕ начинай с "ну типа", "а вдруг", "что если", "короче", "слушай", "знаешь".
НЕ задавай риторических вопросов типа "ты тут?" — лучше что-то конкретное."""

                    resp = requests.post(OLLAMA_URL, json={"model": MODEL_TEXT, "messages": build_messages(prompt), "stream": False}, timeout=60)
                    resp.raise_for_status()
                    answer = resp.json()["message"]["content"]
                    answer = strip_emotion_tags(answer)
                    answer = extract_mc_commands(answer)  # <=== ИЗМЕНЕНИЕ

                    if len(answer) < 10 or answer.endswith("..."):
                        set_busy(False)
                        self.is_auto_speaking = False
                        gui_queue.put(("status", "✅ Промолчала"))
                        return

                    gui_queue.put(("bot", answer))
                    gui_queue.put(("status", "🗣️ Озвучиваю..."))
                    speak(answer)
                    gui_queue.put(("status", "✅ Готов"))

            except Exception as e:
                gui_queue.put(("status", f"❌ Авто-реплика: {e}"))
            finally:
                set_busy(False)
                self.is_auto_speaking = False

        threading.Thread(target=process, daemon=True).start()

    def on_enter(self, event=None):
        global _pending_image
        text = self.entry.get().strip()
        # --- команда /remember ---
        if text.lower().startswith("/remember "):
            fact = text[10:].strip()
            if fact:
                _memory_facts.append({
                    "fact": fact,
                    "date": time.strftime("%Y-%m-%d"),
                    "category": "general"
                })
                save_memory(_memory_facts)
                self.add_message("bot", f"Запомнила: {fact}")
            return
        # -------------------------
        if not text:
            return
        self.entry.delete(0, "end")
        update_activity()
        set_busy(True)
        
        image_path = _pending_image or ""
        _pending_image = None
        
        self.add_message("user", text)
        
        use_search, query = should_use_search(text)
        if gui_settings["use_search"]:
            use_search = True
        
        def process():
            global _pending_image
            try:
                gui_queue.put(("status", "🤖 Думаю..."))
                answer = ask_model(query, use_search=use_search, image_path=image_path)
                answer = extract_mc_commands(answer)
                gui_queue.put(("bot", answer))
                gui_queue.put(("status", "🗣️ Озвучиваю..."))
                speak(answer)
                gui_queue.put(("status", "✅ Готов"))
            except Exception as e:
                gui_queue.put(("status", f"❌ {e}"))
            finally:
                set_busy(False)
        
        threading.Thread(target=process, daemon=True).start()

    def setup_hotkeys(self):
        def safe_start(event=None):
            try:
                stop_playback()
                self.after(0, lambda: self.mic_btn.configure(fg_color="#388E3C", text="🔴"))
                start_recording()
            except Exception:
                pass
        
        def safe_stop(event=None):
            try:
                stop_playback()
            except Exception:
                pass
        
        def safe_screenshot(event=None):
            try:
                self.after(0, self.on_screenshot)
            except Exception:
                pass
        
        def is_record_key(event):
            target = HOTKEY_RECORD.lower().strip()
            if target in ('right ctrl', 'rctrl'):
                # Проверяем именно правый Ctrl через WinAPI
                if event.name == 'right ctrl':
                    return True
                if event.name == 'ctrl':
                    try:
                        VK_RCONTROL = 0xA3
                        # GetAsyncKeyState вернёт отрицательное число, если клавиша нажата
                        state = ctypes.windll.user32.GetAsyncKeyState(VK_RCONTROL)
                        if state & 0x8000:
                            return True
                    except Exception:
                        pass
                return False
            # Для остальных клавиш — точное совпадение
            return event.name == target
        
        def on_key(event):
            if event.event_type == 'down':
                if is_record_key(event):
                    safe_start()
                elif event.name == HOTKEY_STOP:
                    safe_stop()
                elif event.name == HOTKEY_CLEAR:
                    self.after(0, self.clear_chat)
                elif event.name == HOTKEY_SCREENSHOT:
                    safe_screenshot()
                elif event.name == HOTKEY_EXIT:
                    self.after(0, self.on_closing)
                elif event.name == HOTKEY_MODE:
                    self.after(0, self.toggle_vad)
            elif event.event_type == 'up':
                if is_record_key(event):
                    self.after(0, lambda: (self.mic_btn.configure(fg_color="#D32F2F", text="🎙️"), stop_recording()))
        
        keyboard.hook(on_key)

    def check_queue(self):
        while not gui_queue.empty():
            cmd, data = gui_queue.get()
            if cmd == "status":
                self.status_bar.configure(text=data)
            elif cmd == "user":
                self.add_message("user", data)
            elif cmd == "bot":
                self.add_message("bot", data)
            elif cmd == "emotion":
                if hasattr(self, "avatar_window") and self.avatar_window and self.avatar_window.window.winfo_exists():
                    self.avatar_window.set_emotion(data)
            elif cmd == "talking_start":
                if hasattr(self, "avatar_window") and self.avatar_window and self.avatar_window.window.winfo_exists():
                    self.avatar_window.start_talking()
            elif cmd == "audio_data":
                if hasattr(self, "avatar_window") and self.avatar_window and self.avatar_window.window.winfo_exists():
                    self.avatar_window.set_audio_data(data, 80)
            elif cmd == "bot_finished":
                self.last_auto_time = time.time()
                update_activity()
            elif cmd == "clipboard_trigger":
                clip_text, style = data
                self._do_quick_reaction("clipboard", style, clip_text)
            elif cmd == "sleep_trigger":
                style = data
                minutes = int(get_idle_time_seconds() // 60)
                self._do_quick_reaction("sleep", style, str(minutes))
            elif cmd == "mc_danger":
                self._do_mc_reaction(data)
        self.after(100, self.check_queue)

# ═══════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════
    # ═══════════════════════════════════════
    # DISCORD BRIDGE
    # ═══════════════════════════════════════

    def open_discord_window(self):
        if not _has_discord_bridge:
            self.add_message("status", "❌ Модуль discord_audio_bridge.py не найден")
            return
        DiscordSettingsWindow(self, self.discord_bridge)

    def on_discord_connected(self):
        """Вызывается при успешном подключении"""
        self.discord_btn.configure(fg_color="#388E3C", text="📡 ON")
        # Выключаем локальный VAD, чтобы не было двойного слушания
        if self.vad_enabled:
            self.toggle_vad()

    def on_discord_disconnected(self):
        """Вызывается при отключении"""
        self.discord_btn.configure(fg_color="#5C6BC0", text="📡 Discord")
        # Возвращаем локальный VAD, если был включён раньше
        if not self.vad_enabled:
            self.toggle_vad()
if __name__ == "__main__":
    app = App()
    app.mainloop()