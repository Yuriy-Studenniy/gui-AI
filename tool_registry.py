
"""
V-Agent Tool Registry

Единый реестр инструментов V.
Содержит:
- названия и описания инструментов;
- JSON Schema аргументов;
- предусловия вызова;
- условия проверки результата;
- валидацию аргументов.

Здесь намеренно нет игрового исполнения.
Реальные действия будут выполнять Python Tool Executor и C# Game Dispatcher.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ============================================================
    # НАБЛЮДЕНИЕ
    # ============================================================

    "get_game_state": {
        "category": "observation",
        "description": (
            "Получить актуальное состояние игрового мира: "
            "положение игрока и V, состояние автомобиля, "
            "роль V в автомобиле, маршрут и другие доступные данные."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [],
        "completion_check": "fresh_game_state_received",
        "timeout_seconds": 10
    },

    "get_nearby_objects": {
        "category": "observation",
        "description": (
            "Получить структурированный список ближайших физических объектов "
            "вокруг V. Каждый объект содержит id, name, unity_name, position, "
            "distance и pickupable. Используй id для последующего адресного "
            "взаимодействия с конкретным предметом. Не путай одинаковые детали."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "radius_m": {
                    "type": "number",
                    "description": "Радиус поиска в метрах.",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 15
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимальное количество объектов.",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 100
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [],
        "completion_check": "nearby_objects_received",
        "timeout_seconds": 10
    },

    "list_destinations": {
        "category": "observation",
        "description": (
            "Получить список известных точек назначения, "
            "доступных системе навигации."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [],
        "completion_check": "destination_list_received",
        "timeout_seconds": 10
    },

    # ============================================================
    # ПЕРЕДВИЖЕНИЕ V
    # ============================================================

    "follow_player": {
        "category": "movement",
        "description": (
            "Начать следовать за игроком."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is not in a vehicle"
        ],
        "completion_check": "following_is_true",
        "timeout_seconds": 5
    },

    "stop_following": {
        "category": "movement",
        "description": (
            "Прекратить следовать за игроком."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [],
        "completion_check": "following_is_false",
        "timeout_seconds": 5
    },

    "jump": {
        "category": "movement",
        "description": (
            "Выполнить прыжок."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is not in a vehicle"
        ],
        "completion_check": "jump_command_accepted",
        "timeout_seconds": 5
    },

    # ============================================================
    # ПРЕДМЕТЫ
    # ============================================================

    "approach_object": {
        "category": "movement",
        "description": (
            "Подойти к конкретному физическому объекту. "
            "Обязательно используй object_id, полученный через "
            "get_nearby_objects. Инструмент завершится, когда V "
            "подойдёт достаточно близко к выбранному объекту. "
            "Не используй для этого pickup_nearest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": (
                        "Уникальный ID конкретного объекта Unity, "
                        "полученный через get_nearby_objects."
                    ),
                    "minLength": 1,
                    "maxLength": 64
                }
            },
            "required": [
                "object_id"
            ],
            "additionalProperties": False
        },
        "preconditions": [],
        "completion_check": "object_approach_succeeded",
        "timeout_seconds": 50
    },

    "pickup_object": {
        "category": "items",
        "description": (
            "Поднять конкретный физический объект по object_id. "
            "Перед вызовом V должна выполнить approach_object "
            "для того же object_id. Подтверди, что в руках оказался "
            "именно выбранный предмет."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": (
                        "ID предмета, выбранного при сканировании."
                    ),
                    "minLength": 1,
                    "maxLength": 64
                }
            },
            "required": [
                "object_id"
            ],
            "additionalProperties": False
        },
        "preconditions": [
            "V is not holding an item"
        ],
        "completion_check": "held_object_id_matches",
        "timeout_seconds": 5
    },

    "pickup_nearest": {
        "category": "items",
        "description": (
            "Поднять ближайший подходящий физический предмет. "
            "V должна иметь свободные руки."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is not holding an item"
        ],
        "completion_check": "has_item_is_true",
        "timeout_seconds": 5
    },

    "drop_item": {
        "category": "items",
        "description": (
            "Положить предмет, который V держит в руках."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is holding an item"
        ],
        "completion_check": "has_item_is_false",
        "timeout_seconds": 5
    },

    "throw_item": {
        "category": "items",
        "description": (
            "Бросить предмет, который V держит в руках."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is holding an item"
        ],
        "completion_check": "item_released",
        "timeout_seconds": 5
    },

    # ============================================================
    # АВТОМОБИЛЬ
    # ============================================================

    "enter_vehicle": {
        "category": "vehicle",
        "description": (
            "Найти ближайший автомобиль и сесть в него. "
            "Самостоятельно выполнить необходимое взаимодействие "
            "с дверью и процесс посадки. Не вызывай отдельно "
            "open_nearest_door для обычной посадки."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seat_role": {
                    "type": "string",
                    "description": (
                        "Место V: водитель или пассажир."
                    ),
                    "enum": [
                        "driver",
                        "passenger"
                    ],
                    "default": "driver"
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is not in a vehicle",
            "A suitable vehicle is nearby"
        ],
        "completion_check": "in_vehicle_and_seat_role_matches",
        "timeout_seconds": 45
    },

    "exit_vehicle": {
        "category": "vehicle",
        "description": (
            "Выйти из автомобиля, в котором находится V."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "V is in a vehicle"
        ],
        "completion_check": "in_vehicle_is_false",
        "timeout_seconds": 15
    },

    "navigate_to": {
        "category": "vehicle",
        "description": (
            "Начать полноценную навигацию к указанному месту. "
            "Использовать существующую систему RouteNavigator. "
            "Не заменять навигацию прямым управлением газом и рулём."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": (
                        "Название точки назначения из списка доступных мест."
                    ),
                    "minLength": 1,
                    "maxLength": 100
                }
            },
            "required": [
                "destination"
            ],
            "additionalProperties": False
        },
        "preconditions": [
            "V is in the driver seat",
            "Destination exists",
            "Navigation is not already active"
        ],
        "completion_check": "destination_reached_and_route_completed",
        "timeout_seconds": 600
    },

    "cancel_navigation": {
        "category": "vehicle",
        "description": (
            "Безопасно отменить текущую навигацию V: контроллер плавно "
            "тормозит автомобиль, переводит коробку в нейтраль и только "
            "после остановки завершает отмену. Используй этот инструмент, "
            "если продолжать текущий маршрут невозможно или небезопасно: "
            "V или автомобиль перевернулись, машина застряла и не может "
            "продолжить движение, после столкновения заблокирован путь, "
            "автомобиль потерял управляемость либо возникла другая серьёзная "
            "нештатная ситуация. Также используй его, если игрок прямо "
            "попросил немедленно остановить текущую поездку или сменить "
            "маршрут. Перед отменой оцени актуальное состояние через "
            "get_game_state, если ситуация не очевидна. Не отменяй маршрут "
            "из-за обычного разговора, запроса посмотреть состояние, "
            "кратковременной остановки, низкой скорости или небольшого "
            "столкновения, после которого машина всё ещё может ехать. "
            "Если препятствие можно преодолеть без отмены маршрута, "
            "продолжай текущую навигацию. Не вызывай инструмент повторно, "
            "если отмена уже запрошена или навигация уже завершилась."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "Navigation is active or preparing"
        ],
        "completion_check": "navigation_cancelled",
        "timeout_seconds": 10
    },

    # ============================================================
    # ИГРОК
    # ============================================================

    "unseat_player": {
        "category": "player",
        "description": (
            "Высадить игрока с пассажирского места. "
            "Использовать только если игрок действительно находится "
            "на пассажирском месте, отслеживаемом модом."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "player_is_passenger == true"
        ],
        "completion_check": "player_is_passenger_is_false",
        "timeout_seconds": 10
    },

    # ============================================================
    # ДВЕРИ И ОКРУЖЕНИЕ
    # ============================================================

    "open_nearest_door": {
        "category": "interaction",
        "description": (
            "Целенаправленно открыть ближайшую доступную дверь."
            "Не использовать как часть обычной посадки."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "An accessible door is nearby"
        ],
        "completion_check": "door_command_started",
        "timeout_seconds": 10
    },

    "close_nearest_door": {
        "category": "interaction",
        "description": (
            "Целенаправленно закрыть ближайшую доступную дверь."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "preconditions": [
            "An accessible door is nearby"
        ],
        "completion_check": "door_command_started",
        "timeout_seconds": 10
    }
}


# ================================================================
# ПОЛУЧЕНИЕ ИНФОРМАЦИИ ИЗ РЕЕСТРА
# ================================================================

def get_tool_names() -> List[str]:
    """Вернуть названия всех зарегистрированных инструментов."""
    return list(TOOL_REGISTRY.keys())


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Получить описание одного инструмента."""
    tool = TOOL_REGISTRY.get(name)
    return deepcopy(tool) if tool is not None else None


def get_all_tools() -> Dict[str, Dict[str, Any]]:
    """Получить копию всего реестра."""
    return deepcopy(TOOL_REGISTRY)


def get_tool_count() -> int:
    """Количество зарегистрированных инструментов."""
    return len(TOOL_REGISTRY)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Получить определения в формате OpenAI-compatible tools.
    Можно будет использовать для API с поддержкой function calling.
    """
    result = []

    for name, tool in TOOL_REGISTRY.items():
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": deepcopy(tool["parameters"])
            }
        })

    return result


def get_tool_catalog_text() -> str:
    """
    Получить текстовый каталог для передачи модели в промпте.
    Полезен для текущей схемы NVIDIA NIM без native tool_calls.
    """
    lines = [
        "ДОСТУПНЫЕ ИНСТРУМЕНТЫ V:",
        "Вызывай только инструменты из этого списка."
    ]

    for name, tool in TOOL_REGISTRY.items():
        lines.append("")
        lines.append("Инструмент: " + name)
        lines.append("Категория: " + tool["category"])
        lines.append("Описание: " + tool["description"])
        lines.append(
            "Аргументы: " +
            json.dumps(
                tool["parameters"],
                ensure_ascii=False
            )
        )
        preconditions = tool.get("preconditions", [])

        lines.append(
            "Предусловия: " +
            (
                "; ".join(preconditions)
                if preconditions
                else "нет специальных условий"
            )
        )

        lines.append(
            "Проверка результата: " +
            str(tool.get("completion_check", "не указана"))
        )

    return "\n".join(lines)


# ================================================================
# ПРОВЕРКА АРГУМЕНТОВ
# ================================================================

def _validate_value(
    key: str,
    value: Any,
    schema: Dict[str, Any]
) -> Optional[str]:
    """Проверить одно значение согласно его JSON Schema."""

    expected_type = schema.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            return key + " должен быть строкой."

        value = value.strip()

        if len(value) < schema.get("minLength", 0):
            return key + " слишком короткий."

        if len(value) > schema.get("maxLength", 1000000):
            return key + " слишком длинный."

    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return key + " должен быть целым числом."

    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return key + " должен быть числом."

    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return key + " должен быть true или false."

    elif expected_type == "array":
        if not isinstance(value, list):
            return key + " должен быть массивом."

    elif expected_type == "object":
        if not isinstance(value, dict):
            return key + " должен быть объектом."

    if "enum" in schema and value not in schema["enum"]:
        return (
            key + " имеет недопустимое значение. "
            "Допустимо: " + ", ".join(map(str, schema["enum"]))
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return key + " меньше допустимого минимума."

        if "maximum" in schema and value > schema["maximum"]:
            return key + " превышает допустимый максимум."

    return None


def validate_tool_call(
    name: str,
    arguments: Any
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Проверить имя инструмента и его аргументы.

    Возвращает:
        (True, normalized_arguments, None)
        либо
        (False, {}, error_message)
    """

    tool = TOOL_REGISTRY.get(name)

    if tool is None:
        return False, {}, "Неизвестный инструмент: " + str(name)

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return False, {}, "Аргументы инструмента должны быть JSON-объектом."

    schema = tool["parameters"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Проверяем лишние аргументы.
    if schema.get("additionalProperties") is False:
        unknown = set(arguments.keys()) - set(properties.keys())

        if unknown:
            return (
                False,
                {},
                "Неизвестные аргументы: " + ", ".join(sorted(unknown))
            )

    normalized = dict(arguments)

    # Проверяем обязательные аргументы.
    for key in required:
        if key not in normalized:
            return False, {}, "Отсутствует обязательный аргумент: " + key

    # Подставляем значения по умолчанию и проверяем типы.
    for key, property_schema in properties.items():

        if key not in normalized:
            if "default" in property_schema:
                normalized[key] = property_schema["default"]
            continue

        error = _validate_value(
            key,
            normalized[key],
            property_schema
        )

        if error:
            return False, {}, error

        if isinstance(normalized[key], str):
            normalized[key] = normalized[key].strip()

    return True, normalized, None


# ================================================================
# БЫСТРАЯ САМОПРОВЕРКА
# ================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("V-AGENT TOOL REGISTRY")
    print("=" * 55)

    print("Количество инструментов:", get_tool_count())

    for tool_name in get_tool_names():
        print(" +", tool_name)

    print("\nПроверка аргументов:")

    tests = [
        ("enter_vehicle", {}),
        ("enter_vehicle", {"seat_role": "driver"}),
        ("enter_vehicle", {"seat_role": "passenger"}),
        ("enter_vehicle", {"seat_role": "pilot"}),
        ("navigate_to", {"destination": "zerno"}),
        ("navigate_to", {}),
        ("navigate_to", {"destination": "", "extra": 123}),
        ("get_game_state", {})
    ]

    for name, arguments in tests:
        valid, normalized, error = validate_tool_call(
            name,
            arguments
        )

        print(
            name,
            "OK" if valid else "ERROR",
            normalized if valid else error
        )

    print("\nСамопроверка завершена.")
