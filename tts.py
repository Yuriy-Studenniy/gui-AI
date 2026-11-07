import os
import torch
import soundfile as sf

# Создаём папку рядом со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "assets", "vocalizations")
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_ru',
    device=device,
    trust_repo=True
)

phrases = {
    "laugh": "ха ха ха",
    "chuckle": "хи хи",
    "sigh": "эээх",
    "exhale": "фух",
    "hmm": "хммм",
    "think": "ааа",
    "gasp": "о-о-о",
    "oh": "ой",
    "tsk": "тьфу",
    "grunt": "хм-м",
}

for name, text in phrases.items():
    audio = model.apply_tts(text=text, speaker="aidar", sample_rate=48000)
    path = os.path.join(out_dir, f"{name}.wav")
    sf.write(path, audio.cpu().numpy(), 48000)
    print(f"✅ {path}")

print("\n🎉 Готово! Файлы лежат в:", out_dir)