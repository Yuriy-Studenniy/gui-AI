import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import math
import time

# ─────────── CONFIG ───────────
# ─────────── CONFIG ───────────
# Сюда пиши пути к трём шрифтам по порядку
FONT_PATHS = [
    "fonts/zekton.otf",     # 0 — Zekton
    "fonts/oswald.ttf",     # 1 — Oswald Bold
    "fonts/unbounded.ttf",  # 2 — Unbounded
]
FONT_INDEX = 0            # 0 / 1 / 2 — какой сейчас тестируем
FONT_PATH = FONT_PATHS[FONT_INDEX]

FONT_SIZE = 46
WIN_W = 1400
WIN_H = 240
BG = "#00FF00"

STRIPE_COLOR = (255, 212, 0)
STRIPE_BASE = (72, 72, 72)
STRIPE_WIDTH = 7
STRIPE_STEP = 22

SHADOW_COLOR = (35, 35, 35)
SHADOW_OFF = 4
OUTLINE_WIDTH = 3

TEXT = "адольф гитлер ни в чем не виноват, все будет хорошо, бурмалда, пизда или нормалы, стоп а где колбаска гогого гогого"

# ─────────── FONT ───────────
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)


def make_pattern(w, h):
    img = Image.new("RGB", (w, h), STRIPE_BASE)
    d = ImageDraw.Draw(img)
    for x in range(-h, w + h, STRIPE_STEP):
        d.line([(x, h), (x + h, 0)], fill=STRIPE_COLOR, width=STRIPE_WIDTH)
    return img


def render_word(word, pattern, px, py):
    bbox = font.getbbox(word)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 10
    W = tw + pad * 2
    H = th + pad * 2

    # маска заливки
    inner = Image.new("L", (W, H), 0)
    ImageDraw.Draw(inner).text(
        (pad - bbox[0], pad - bbox[1]), word,
        font=font, fill=255
    )

    # маска контура
    outer = Image.new("L", (W, H), 0)
    ImageDraw.Draw(outer).text(
        (pad - bbox[0], pad - bbox[1]), word,
        font=font, fill=255,
        stroke_width=OUTLINE_WIDTH, stroke_fill=255
    )

    # тень
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.composite(
        Image.new("RGBA", (W, H), SHADOW_COLOR + (255,)),
        shadow, outer
    )

    # контур
    outline = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    outline = Image.composite(
        Image.new("RGBA", (W, H), (0, 0, 0, 255)),
        outline, outer
    )

    # вырезка паттерна
    x0 = int(px) - pad
    y0 = int(py) - pad
    fill = pattern.crop((x0, y0, x0 + W, y0 + H)).convert("RGBA")
    fill.putalpha(inner)

    return fill, outline, shadow, pad


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Subs test")
        self.root.configure(bg=BG)
        self.canvas = tk.Canvas(self.root, width=WIN_W, height=WIN_H,
                                bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack()

        self.pattern = make_pattern(WIN_W + 300, WIN_H + 300)

        self.words = TEXT.split()
        self.items = []
        self._imgs = []  # keep references

        sp = font.getlength(" ")
        widths = []
        for w in self.words:
            bb = font.getbbox(w)
            widths.append(bb[2] - bb[0])

        total = sum(widths) + sp * (len(self.words) - 1)
        cur_x = (WIN_W - total) / 2
        base_y = (WIN_H - FONT_SIZE) / 2

        for i, w in enumerate(self.words):
            fill, outline, shadow, pad = render_word(w, self.pattern, cur_x, base_y)

            tk_sh = ImageTk.PhotoImage(shadow)
            tk_ol = ImageTk.PhotoImage(outline)
            tk_fl = ImageTk.PhotoImage(fill)
            self._imgs += [tk_sh, tk_ol, tk_fl]

            x = cur_x - pad
            y = base_y - pad

            id_sh = self.canvas.create_image(x + SHADOW_OFF, y + SHADOW_OFF, image=tk_sh, anchor="nw")
            id_ol = self.canvas.create_image(x, y, image=tk_ol, anchor="nw")
            id_fl = self.canvas.create_image(x, y, image=tk_fl, anchor="nw")

            self.items.append({
                "sh": id_sh, "ol": id_ol, "fl": id_fl,
                "bx": x, "by": y,
                "phase": i * 0.55,
            })

            cur_x += widths[i] + sp

        self.t0 = time.time()
        self.animate()

    def animate(self):
        t = time.time() - self.t0
        n = len(self.items)

        # караоке: активное слово меняется раз в 0.45 сек
        active = int(t / 0.45) % n

        for i, it in enumerate(self.items):
            # базовая волна всего блока
            wave = math.sin(t * 2.0 + it["phase"]) * 3

            # доп. bounce для активного слова
            if i == active:
                wave += -6 + math.sin(t * 9.0) * 5

            x = it["bx"]
            y = it["by"] + wave

            self.canvas.coords(it["fl"], x, y)
            self.canvas.coords(it["ol"], x, y)
            self.canvas.coords(it["sh"], x + SHADOW_OFF, y + SHADOW_OFF)

        self.root.after(16, self.animate)


if __name__ == "__main__":
    App()
    tk.mainloop()