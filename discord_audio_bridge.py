"""
Discord Audio Bridge для V — Sliding Window Wake Word Edition
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import threading
import time
import os
import queue
import re
import base64
import shutil
import requests
from collections import deque
import sys

core = sys.modules['__main__']


class DiscordAudioBridge:
    def __init__(self):
        self.running = False
        self.thread = None
        self.input_device_idx = None
        self.output_device_idx = None

        self.wake_words = ["в", "вэй", "виктория", "эй в", "привет в"]
        self.wake_window_sec = 3.0
        self.wake_step_sec = 1.5
        self.wake_min_speech_ratio = 0.3

        self.vad_threshold = 0.5
        self.silence_sec = 2.0
        self.min_speech_sec = 0.8
        self.max_speech_sec = 15.0
        self.echo_cooldown_sec = 1.5

        self._state_lock = threading.Lock()
        self.state = "LISTENING"
        self.is_speaking = False
        self._echo_until = 0.0

        self.ring_buffer = []
        self.ring_max_chunks = int(16000 * self.wake_window_sec / 512)
        self.speech_buffer = []

        self._last_wake_check = 0.0
        self._wake_checking = False

    @staticmethod
    def get_input_devices():
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    @staticmethod
    def get_output_devices():
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]

    def set_devices_by_name(self, input_name, output_name):
        inputs = self.get_input_devices()
        outputs = self.get_output_devices()

        self.input_device_idx = None
        self.output_device_idx = None

        for idx, name in inputs:
            if input_name.lower() in name.lower():
                self.input_device_idx = idx
                break

        for idx, name in outputs:
            if output_name.lower() in name.lower():
                self.output_device_idx = idx
                break

        print(f"[DiscordBridge] Input idx={self.input_device_idx}, Output idx={self.output_device_idx}")

    def start(self):
        if self.running:
            return True
        if self.input_device_idx is None:
            core.gui_queue.put(("status", "❌ Discord: входное устройство не найдено"))
            return False
        if self.output_device_idx is None:
            core.gui_queue.put(("status", "❌ Discord: выходное устройство не найдено"))
            return False

        with self._state_lock:
            self.state = "LISTENING"
        self.ring_buffer = []
        self.speech_buffer = []
        self._wake_checking = False
        self._last_wake_check = 0.0

        self.running = True
        self.thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.thread.start()
        core.gui_queue.put(("status", "👂 Discord: слушаю канал (wake word)..."))
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        self.is_speaking = False
        core.gui_queue.put(("status", "📡 Discord отключён"))

    def _vad_worker(self):
        sr = 16000
        chunk_samples = 512
        silence_maxframes = int(self.silence_sec * sr / chunk_samples)

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            if not self.running:
                raise sd.CallbackAbort
            audio_q.put(indata.copy())

        try:
            stream = sd.InputStream(
                device=self.input_device_idx,
                samplerate=sr,
                channels=1,
                dtype='float32',
                blocksize=chunk_samples,
                callback=callback
            )
            stream.start()
            print(f"[DiscordBridge] ✅ VAD поток запущен на устройстве idx={self.input_device_idx}")
        except Exception as e:
            print(f"[DiscordBridge] ❌ Ошибка запуска аудиопотока: {e}")
            core.gui_queue.put(("status", f"❌ Discord audio: {e}"))
            self.running = False
            return

        silence_frames = 0

        try:
            while self.running:
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk_flat = chunk.flatten()
                if len(chunk_flat) < chunk_samples:
                    continue
                chunk_flat = chunk_flat[:chunk_samples]

                if time.time() < self._echo_until:
                    continue

                tensor = torch.from_numpy(chunk_flat).float()
                with torch.no_grad():
                    prob = core.vad_model(tensor, sr).item()
                is_speech = prob > self.vad_threshold

                with self._state_lock:
                    current_state = self.state

                if current_state == "LISTENING":
                    self.ring_buffer.append((chunk_flat, is_speech))
                    if len(self.ring_buffer) > self.ring_max_chunks:
                        self.ring_buffer.pop(0)

                    now = time.time()
                    if (now - self._last_wake_check > self.wake_step_sec and
                            not self._wake_checking and
                            len(self.ring_buffer) >= self.ring_max_chunks * 0.7):

                        speech_ratio = sum(1 for _, sp in self.ring_buffer if sp) / len(self.ring_buffer)
                        print(f"[DiscordBridge] Wake check timer fired. Speech ratio: {speech_ratio:.2f}")

                        if speech_ratio >= self.wake_min_speech_ratio:
                            self._last_wake_check = now
                            self._wake_checking = True
                            audio_copy = np.concatenate([c for c, _ in self.ring_buffer])
                            print(f"[DiscordBridge] Launching wake check thread...")
                            threading.Thread(
                                target=self._check_wake_word,
                                args=(audio_copy,),
                                daemon=True
                            ).start()
                        else:
                            print(f"[DiscordBridge] Too quiet, skipping wake check")
                            self._last_wake_check = now

                elif current_state == "CAPTURING":
                    self.speech_buffer.append(chunk_flat)
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1
                        if silence_frames >= silence_maxframes:
                            if len(self.speech_buffer) > 0:
                                audio = np.concatenate(self.speech_buffer, axis=0)
                                duration = len(audio) / sr
                                print(f"[DiscordBridge] Silence detected. Captured {duration:.1f}s")
                                if self.min_speech_sec <= duration <= self.max_speech_sec:
                                    threading.Thread(
                                        target=self._process_full,
                                        args=(audio,),
                                        daemon=True
                                    ).start()

                            with self._state_lock:
                                self.state = "LISTENING"
                            self.speech_buffer = []
                            silence_frames = 0

        except Exception as e:
            print(f"[DiscordBridge] ❌ VAD ошибка: {e}")
        finally:
            stream.stop()
            stream.close()
            print("[DiscordBridge] VAD поток завершён")

    def _check_wake_word(self, audio_np):
        try:
            print(f"[DiscordBridge] Wake check: audio len={len(audio_np)} samples")
            path = os.path.join(core.TEMP_DIR, f"dc_wake_{int(time.time() * 1000)}.wav")
            sf.write(path, audio_np, 16000)

            segments, _ = core.whisper_model.transcribe(
                path,
                language="ru",
                beam_size=1,
                best_of=1,
                vad_filter=True
            )
            text = " ".join([s.text for s in segments]).strip()
            print(f"[DiscordBridge] Wake check Whisper result: '{text}'")

            text_lower = text.lower()
            has_wake = any(w.lower() in text_lower for w in self.wake_words)
            if has_wake:
                print(f"[DiscordBridge] ✅ Wake word DETECTED!")
                with self._state_lock:
                    if self.state == "LISTENING":
                        self.state = "CAPTURING"
                        self.speech_buffer = [c for c, _ in self.ring_buffer]
                        self.ring_buffer = []
            else:
                print(f"[DiscordBridge] ❌ Wake word not found")
                with self._state_lock:
                    if self.state == "LISTENING":
                        self.ring_buffer = []

        except Exception as e:
            print(f"[DiscordBridge] Wake check error: {e}")
        finally:
            self._wake_checking = False
            try:
                os.remove(path)
            except:
                pass

    def _process_full(self, audio_np):
        if self.is_speaking:
            print("[DiscordBridge] Already speaking, skipping")
            return

        try:
            path = os.path.join(core.TEMP_DIR, f"dc_full_{int(time.time() * 1000)}.wav")
            sf.write(path, audio_np, 16000)

            text = core.transcribe_audio(path)
            if not text or len(text.strip()) < 2:
                print("[DiscordBridge] Full whisper: empty text")
                return

            clean_text = text
            for w in self.wake_words:
                pattern = rf'(^|\s){re.escape(w.lower())}(\s|$|[,;:.!?])'
                clean_text = re.sub(pattern, ' ', clean_text, flags=re.IGNORECASE)
            clean_text = clean_text.strip(' ,.;:!?')
            if not clean_text:
                clean_text = text

            if core._last_bot_text and clean_text.lower() in core._last_bot_text.lower():
                print("[DiscordBridge] Echo filtered")
                return

            print(f"[DiscordBridge] 🎤 {text}")
            print(f"[DiscordBridge] → {clean_text}")

            core.gui_queue.put(("user", f"[Discord] {text}"))
            core.gui_queue.put(("status", "🤖 Думаю..."))

            answer = core.ask_model(clean_text, use_search=False, image_path="", save_history=True)
            core.gui_queue.put(("bot", answer))
            core.gui_queue.put(("status", "🗣️ Озвучиваю..."))

            self._speak(answer)

        except Exception as e:
            core.gui_queue.put(("status", f"❌ Discord: {e}"))
            print(f"[DiscordBridge] Process error: {e}")

    def _speak(self, text):
        self.is_speaking = True
        self._echo_until = time.time() + self.echo_cooldown_sec

        try:
            text = core.strip_emotion_tags(text)
            text = core.preprocess_for_tts(text)
            if not text.strip():
                return

            s = core.gui_settings
            audio = core.model.apply_tts(
                text=text,
                speaker=s.get("speaker", "aidar"),
                sample_rate=core.sample_rate
            )

            base = os.path.join(core.TEMP_DIR, f"dc_tts_{int(time.time() * 1000)}.wav")
            sf.write(base, audio.cpu().numpy(), core.sample_rate)

            try:
                with open(base, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                payload = {
                    "data": [
                        0,
                        {"name": os.path.basename(base), "data": f"data:audio/wav;base64,{b64}"},
                        s.get("pitch", 0),
                        s.get("f0_method", "rmvpe"),
                        "",
                        s.get("index_rate", 0.85),
                        0,
                        s.get("volume_mix", 0.25),
                        s.get("protect", 0.33)
                    ]
                }
                resp = requests.post(f"{core.RVC_URL}/run/infer_convert", json=payload, timeout=60)
                resp.raise_for_status()
                item = resp.json()["data"][1]
                if isinstance(item, dict) and item.get("is_file"):
                    time.sleep(0.3)
                    out = os.path.join(core.TEMP_DIR, f"dc_rvc_{int(time.time() * 1000)}.wav")
                    shutil.copy(item["name"], out)
                    self._play(out)
                    return
            except Exception as e:
                print(f"[DiscordBridge] RVC skip: {e}")

            self._play(base)

        finally:
            self.is_speaking = False

    def _play(self, filepath):
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            volume_frames = core.analyze_audio_volume(filepath, frame_ms=80)
            core.gui_queue.put(("audio_data", volume_frames))
            core.gui_queue.put(("emotion", core.get_emotion_name(core._last_bot_text)))
            core.gui_queue.put(("talking_start", ""))

            print(f"[DiscordBridge] Playing {len(data)} samples to device {self.output_device_idx}")
            sd.play(data, sr, device=self.output_device_idx)
            sd.wait()
            print("[DiscordBridge] Playback finished")

        except Exception as e:
            print(f"[DiscordBridge] Play error: {e}")