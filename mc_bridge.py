import requests
import time

MC_API = "http://localhost:3000"

def mc_command(action: str, target: str = "") -> dict:
    try:
        r = requests.post(f"{MC_API}/command", json={"action": action, "target": target}, timeout=30)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mc_state() -> dict:
    try:
        r = requests.get(f"{MC_API}/state", timeout=3)
        return r.json()
    except Exception:
        return {"connected": False}

def mc_is_connected() -> bool:
    return mc_state().get("connected", False)

def mc_get_danger() -> list:
    state = mc_state()
    if not state.get("connected"):
        return []
    mobs = state.get("mobs", [])
    danger = []
    for m in mobs:
        if m.get("distance", 999) < 8:
            danger.append(m)
    return danger

def mc_sleep() -> dict:
    return mc_command("sleep")

def mc_wake() -> dict:
    return mc_command("wake")

def mc_interact(block_name: str) -> dict:
    return mc_command("interact", block_name)

def mc_craft(item_name: str, count: int = 1) -> dict:
    return mc_command("craft", f"{item_name}:{count}")

def mc_blocks() -> dict:
    return mc_command("blocks")

def mc_context_for_prompt() -> str:
    state = mc_state()
    if not state.get("connected"):
        return ""

    parts = []
    parts.append(f"Твой HP: {state.get('health', '?')}, голод: {state.get('food', '?')}.")
    p = state.get('pos', {})
    parts.append(f"Ты на координатах X:{p.get('x','?')} Y:{p.get('y','?')} Z:{p.get('z','?')}.")

    mode = state.get('mode', 'idle')
    parts.append(f"Текущий режим: {mode}.")

    if state.get('is_mining'):
        parts.append("Сейчас добываю ресурсы.")

    if state.get('hunger_warning'):
        parts.append("Ты очень голодна, еды нет, ты можешь умереть.")

    mobs = state.get("mobs", [])
    if mobs:
        mob_str = ", ".join([f"{m['name']}({m['distance']}б)" for m in mobs[:4]])
        parts.append(f"Рядом: {mob_str}.")

    blocks = state.get("blocks", [])
    if blocks:
        block_str = ", ".join([f"{b['name']}({b['distance']}б)" for b in blocks[:8]])
        parts.append(f"Блоки рядом: {block_str}.")

    inv = state.get("inventory", [])
    if inv:
        parts.append(f"В инвентаре: {', '.join(inv[:5])}.")

    return "\n".join(parts)