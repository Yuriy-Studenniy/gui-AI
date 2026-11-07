
"""
V-Agent Tool Executor

Подготавливает вызовы игровых инструментов.

Отвечает за:
- проверку существования инструмента;
- проверку аргументов;
- проверку игровых предусловий;
- создание уникального call_id;
- формирование пакета для C#;
- единый формат результатов выполнения.

Важно:
status="ready" означает, что команда проверена
и готова к передаче игровому исполнителю.
Это НЕ означает, что действие уже выполнено.
"""

import uuid
from typing import Any, Dict, Optional, Tuple

from .tool_registry import (
    get_tool,
    validate_tool_call
)


# Версия протокола между Python и C#.
PROTOCOL_VERSION = "v_game_tool_v1"

# Статусы, которые может возвращать игровой исполнитель.
EXECUTION_STATUSES = {
    "accepted",
    "running",
    "succeeded",
    "failed",
    "rejected",
    "timeout"
}


class ToolExecutor:

    def __init__(self):
        pass

    # ============================================================
    # ОСНОВНАЯ ПОДГОТОВКА ВЫЗОВА
    # ============================================================

    def prepare_call(
        self,
        tool_name: str,
        arguments: Any = None,
        game_state: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Проверить запрос модели и подготовить команду для C#.

        Возвращает:
            ready    — команда проверена и готова к отправке;
            rejected — действие запрещено текущим состоянием игры;
            invalid  — неизвестный инструмент или неправильные аргументы.
        """

        if call_id is None:
            call_id = "call_" + uuid.uuid4().hex

        if not isinstance(call_id, str) or not call_id.strip():
            return self._make_error(
                "",
                tool_name,
                "invalid",
                "invalid_call_id",
                "Некорректный идентификатор вызова."
            )

        if len(call_id) > 128:
            return self._make_error(
                call_id,
                tool_name,
                "invalid",
                "invalid_call_id",
                "Слишком длинный идентификатор вызова."
            )

        # 1. Проверяем, зарегистрирован ли инструмент.
        tool = get_tool(tool_name)

        if tool is None:
            return self._make_error(
                call_id,
                tool_name,
                "invalid",
                "unknown_tool",
                "Инструмент не зарегистрирован: " + str(tool_name)
            )

        # 2. Проверяем аргументы по JSON Schema.
        valid, normalized_args, error = validate_tool_call(
            tool_name,
            arguments
        )

        if not valid:
            return self._make_error(
                call_id,
                tool_name,
                "invalid",
                "invalid_arguments",
                error or "Неправильные аргументы."
            )

        # 3. Проверяем игровые предусловия.
        state = game_state if isinstance(game_state, dict) else {}

        allowed, reason_code, reason = self._check_preconditions(
            tool_name,
            normalized_args,
            state
        )

        if not allowed:
            return self._make_error(
                call_id,
                tool_name,
                "rejected",
                reason_code,
                reason
            )

        # 4. Формируем пакет для игрового исполнителя.
        dispatch = {
            "protocol": PROTOCOL_VERSION,
            "call_id": call_id,
            "tool": tool_name,
            "arguments": normalized_args
        }

        # 5. Возвращаем проверенную команду.
        return {
            "call_id": call_id,
            "tool": tool_name,
            "status": "ready",
            "arguments": normalized_args,
            "dispatch": dispatch,
            "verification": {
                "expected_check": tool.get("completion_check"),
                "timeout_seconds": tool.get("timeout_seconds", 30)
            },
            "reason": None
        }

    # ============================================================
    # ПРОВЕРКА ПРЕДУСЛОВИЙ
    # ============================================================

    def _check_preconditions(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Проверяет условия, которые можно подтвердить
        по текущему game_state.

        Условия, требующие непосредственного обращения к Unity
        (например, наличие двери рядом), проверит C#-диспетчер.
        """

        # --------------------------------------------------------
        # Передвижение
        # --------------------------------------------------------

        if tool_name in ("follow_player", "jump"):

            value = self._get_required_bool(state, "v_in_car")

            if value is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить, находится ли V в автомобиле."
                )

            if value:
                return (
                    False,
                    "v_in_vehicle",
                    "V уже находится в автомобиле. Действие запрещено."
                )

        # --------------------------------------------------------
        # Предметы
        # --------------------------------------------------------

        elif tool_name == "pickup_nearest":

            has_item = self._get_required_bool(state, "has_item")

            if has_item is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить, свободны ли руки V."
                )

            if has_item:
                return (
                    False,
                    "hands_not_free",
                    "V уже держит предмет."
                )

        elif tool_name in ("drop_item", "throw_item"):

            has_item = self._get_required_bool(state, "has_item")

            if has_item is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить, держит ли V предмет."
                )

            if not has_item:
                return (
                    False,
                    "no_item",
                    "V ничего не держит."
                )

        # --------------------------------------------------------
        # Посадка в автомобиль
        # --------------------------------------------------------

        elif tool_name == "enter_vehicle":

            in_car = self._get_required_bool(state, "v_in_car")

            if in_car is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить положение V."
                )

            if in_car:
                return (
                    False,
                    "already_in_vehicle",
                    "V уже находится в автомобиле."
                )

            # Дополнительная защита от повторной посадки.
            entering = state.get("is_entering_car")

            if entering is True:
                return (
                    False,
                    "entering_vehicle",
                    "V уже выполняет посадку в автомобиль."
                )

        elif tool_name == "exit_vehicle":

            in_car = self._get_required_bool(state, "v_in_car")

            if in_car is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить положение V."
                )

            if not in_car:
                return (
                    False,
                    "not_in_vehicle",
                    "V не находится в автомобиле."
                )

        # --------------------------------------------------------
        # Навигация
        # --------------------------------------------------------

        elif tool_name == "navigate_to":

            seat_role = state.get("seat_role")

            if seat_role is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить место V в автомобиле."
                )

            if seat_role != "driver":
                return (
                    False,
                    "not_driver",
                    "V не за рулём. Самостоятельно управлять навигацией нельзя."
                )

            route_active = self._get_required_bool(
                state,
                "route_active"
            )

            route_preparing = self._get_required_bool(
                state,
                "route_preparing"
            )

            if route_active is None or route_preparing is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить состояние навигации."
                )

            if route_active or route_preparing:
                return (
                    False,
                    "navigation_already_active",
                    "Навигация уже выполняется или подготавливается."
                )

            route_finishing = state.get("route_finishing")

            if route_finishing is True:
                return (
                    False,
                    "navigation_finishing",
                    "Предыдущий маршрут ещё завершается."
                )

        elif tool_name == "cancel_navigation":

            route_active = self._get_required_bool(
                state,
                "route_active"
            )

            route_preparing = self._get_required_bool(
                state,
                "route_preparing"
            )

            if route_active is None or route_preparing is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось определить состояние навигации."
                )

            if not route_active and not route_preparing:
                return (
                    False,
                    "navigation_not_active",
                    "Сейчас нет активной навигации для отмены."
                )

        # --------------------------------------------------------
        # Игрок
        # --------------------------------------------------------

        elif tool_name == "unseat_player":

            player_is_passenger = self._get_required_bool(
                state,
                "player_is_passenger"
            )

            if player_is_passenger is None:
                return (
                    False,
                    "state_unavailable",
                    "Не удалось подтвердить положение игрока."
                )

            if not player_is_passenger:
                return (
                    False,
                    "player_not_passenger",
                    "Игрок не находится на отслеживаемом пассажирском месте. "
                    "Высадка запрещена."
                )

        # --------------------------------------------------------
        # Остальные инструменты
        # --------------------------------------------------------

        # get_game_state, get_nearby_objects, list_destinations,
        # stop_following, open_nearest_door, close_nearest_door
        # не требуют здесь дополнительных проверок.
        #
        # Проверка наличия конкретных объектов и физических
        # условий будет выполняться на стороне C#.

        return True, "", None

    # ============================================================
    # ЧТЕНИЕ ПОЛЕЙ GAME STATE
    # ============================================================

    @staticmethod
    def _get_required_bool(
        state: Dict[str, Any],
        key: str
    ) -> Optional[bool]:
        """
        Строго читает boolean из game_state.

        None означает, что поле отсутствует или имеет
        неправильный тип. Это не считается False.
        """

        if key not in state:
            return None

        value = state[key]

        if not isinstance(value, bool):
            return None

        return value

    # ============================================================
    # ФОРМИРОВАНИЕ ОШИБОК
    # ============================================================

    @staticmethod
    def _make_error(
        call_id: str,
        tool_name: str,
        status: str,
        reason_code: str,
        message: str
    ) -> Dict[str, Any]:

        return {
            "call_id": call_id,
            "tool": tool_name,
            "status": status,
            "arguments": {},
            "dispatch": None,
            "verification": None,
            "reason": {
                "code": reason_code,
                "message": message
            }
        }


# ================================================================
# ЕДИНЫЙ ФОРМАТ РЕЗУЛЬТАТА ОТ ИГРОВОГО ИСПОЛНИТЕЛЯ
# ================================================================

def make_tool_result(
    call_id: str,
    tool_name: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    reason: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Формирует ответ C# на выполненный вызов инструмента.

    Эта функция будет использоваться при подключении
    игрового диспетчера.
    """

    if status not in EXECUTION_STATUSES:
        raise ValueError(
            "Недопустимый статус выполнения: " + str(status)
        )

    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Отсутствует call_id.")

    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("Отсутствует название инструмента.")

    return {
        "call_id": call_id,
        "tool": tool_name,
        "status": status,
        "result": result or {},
        "reason": reason
    }


# ================================================================
# САМОПРОВЕРКА
# ================================================================

if __name__ == "__main__":

    executor = ToolExecutor()

    # Базовое состояние для тестов.
    base_state = {
        "v_in_car": False,
        "v_following": False,
        "has_item": False,
        "route_active": False,
        "route_preparing": False,
        "route_finishing": False,
        "is_entering_car": False,
        "seat_role": "none",
        "player_is_passenger": False
    }

    tests = [

        (
            "Посадка за руль",
            "enter_vehicle",
            {},
            base_state
        ),

        (
            "Посадка пассажиром",
            "enter_vehicle",
            {"seat_role": "passenger"},
            base_state
        ),

        (
            "Повторная посадка",
            "enter_vehicle",
            {},
            {**base_state, "v_in_car": True}
        ),

        (
            "Поднять предмет со свободными руками",
            "pickup_nearest",
            {},
            base_state
        ),

        (
            "Поднять предмет, когда руки заняты",
            "pickup_nearest",
            {},
            {**base_state, "has_item": True}
        ),

        (
            "Положить предмет, когда он есть",
            "drop_item",
            {},
            {**base_state, "has_item": True}
        ),

        (
            "Положить предмет, когда рук ничего не держит",
            "drop_item",
            {},
            base_state
        ),

        (
            "Навигация за рулём",
            "navigate_to",
            {"destination": "zerno"},
            {
                **base_state,
                "v_in_car": True,
                "seat_role": "driver"
            }
        ),

        (
            "Навигация с пассажирского места",
            "navigate_to",
            {"destination": "zerno"},
            {
                **base_state,
                "v_in_car": True,
                "seat_role": "passenger"
            }
        ),

        (
            "Повторный запуск навигации",
            "navigate_to",
            {"destination": "zerno"},
            {
                **base_state,
                "v_in_car": True,
                "seat_role": "driver",
                "route_active": True
            }
        ),

        (
            "Отмена активной навигации",
            "cancel_navigation",
            {},
            {
                **base_state,
                "v_in_car": True,
                "seat_role": "driver",
                "route_active": True
            }
        ),

        (
            "Отмена неактивной навигации",
            "cancel_navigation",
            {},
            base_state
        ),

        (
            "Высадка пассажира",
            "unseat_player",
            {},
            {
                **base_state,
                "player_is_passenger": True
            }
        ),

        (
            "Высадка игрока, который не пассажир",
            "unseat_player",
            {},
            base_state
        ),

        (
            "Неизвестный инструмент",
            "fly_to_moon",
            {},
            base_state
        )
    ]

    print("=" * 65)
    print("V-AGENT TOOL EXECUTOR")
    print("=" * 65)

    passed = 0

    for title, name, arguments, state in tests:

        prepared = executor.prepare_call(
            name,
            arguments,
            state
        )

        status = prepared["status"]

        print("\n[" + status.upper() + "] " + title)
        print("Инструмент:", name)

        if status == "ready":
            print("Call ID:", prepared["call_id"])
            print("Аргументы:", prepared["arguments"])
            print("Пакет:", prepared["dispatch"])
        else:
            print("Причина:", prepared["reason"])

        passed += 1

    print("\n" + "=" * 65)
    print("Проверено сценариев:", passed)
    print("Самопроверка завершена.")
    print("=" * 65)
