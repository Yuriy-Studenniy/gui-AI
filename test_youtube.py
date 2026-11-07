import yt_dlp
import json

url = "https://youtu.be/8JBXHVqZDzg?si=DRkG1Faajx04yq2_"  # замени на свою ссылку

ydl_opts = {
    'skip_download': True,
    'quiet': True,
    'no_warnings': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    
    # Посмотрим, что лежит в автоматических субтитрах для русского языка
    ru_subs = info.get('automatic_captions', {}).get('ru', [])
    print("=== Структура ru_subs ===")
    for item in ru_subs:
        print(f"ext: {item.get('ext')}, url: {item.get('url', '')[:80]}...")