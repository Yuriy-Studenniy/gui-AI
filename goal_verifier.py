"""
Goal Verifier

Проверяет, достигнута ли игровая цель
по фактическому состоянию игры.

Не принимает решения за LLM.
Не выполняет инструменты.
Только проверяет условия достижения цели.
"""

from typing import Any, Dict, Optional


class GoalVerifier:

    # --------------------------------------------------------
    # ДОПУСТИМЫЕ ТИПЫ ЦЕЛЕЙ
    # --------------------------------------------------------

    SUPPORTED_GOALS = {
        "enter_vehicle",
        "exit_vehicle",
        "follow_player",
        "stop_following",
        "navigate_to",
        "pickup_item",
        "pickup_object",
        "approach_object",
        "return_to_player",
        "bring_item",
        "drop_item",
        "unseat_player"
    }

    # --------------------------------------------------------
    # ПРОВЕРКА И НОРМАЛИЗАЦИЯ ОПИСАНИЯ ЦЕЛИ
    # --------------------------------------------------------

    @classmethod
    def normalize_goal_spec(
        cls,
        goal_spec: Any
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(goal_spec, dict):
            return None

        goal_type = goal_spec.get("type")

        if not isinstance(goal_type, str):
            return None

        goal_type = goal_type.strip().lower()

        if goal_type not in cls.SUPPORTED_GOALS:
            return None

        normalized = {
            "type": goal_type
        }

        # Цель: сесть в автомобиль.
        if goal_type == "enter_vehicle":
            seat_role = goal_spec.get("seat_role", "any")

            if seat_role not in ("driver", "passenger", "any"):
                return None

            normalized["seat_role"] = seat_role

        # Цель: добраться до пункта назначения.
        elif goal_type == "navigate_to":
            destination = goal_spec.get("destination")

            if not isinstance(destination, str):
                return None

            destination = destination.strip()

            if not destination:
                return None

            normalized["destination"] = destination

        # Цели, связанные с конкретным физическим объектом.
        elif goal_type in (
            "approach_object",
            "pickup_object",
            "return_to_player",
            "bring_item"
        ):
            object_id = goal_spec.get("object_id")

            if not isinstance(object_id, str):
                return None

            object_id = object_id.strip()

            if not object_id or len(object_id) > 64:
                return None

            normalized["object_id"] = object_id

            if goal_type in ("return_to_player", "bring_item"):
                max_distance = goal_spec.get(
                    "max_distance_m",
                    2.5
                )

                if (
                    isinstance(max_distance, bool)
                    or not isinstance(max_distance, (int, float))
                    or max_distance < 0.5
                    or max_distance > 10
                ):
                    return None

                normalized["max_distance_m"] = float(max_distance)

        return normalized

    # --------------------------------------------------------
    # ОСНОВНАЯ ПРОВЕРКА ЦЕЛИ
    # --------------------------------------------------------

    def evaluate(
        self,
        goal_spec: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:

        normalized = self.normalize_goal_spec(goal_spec)

        if normalized is None:
            return self._result(
                "invalid",
                "Некорректное или неподдерживаемое описание цели."
            )

        if not isinstance(game_state, dict):
            return self._result(
                "unknown",
                "Состояние игры недоступно."
            )

        goal_type = normalized["type"]

        # ----------------------------------------------------
        # ПОСАДКА В МАШИНУ
        # ----------------------------------------------------

        if goal_type == "enter_vehicle":
            if "v_in_car" not in game_state:
                return self._result(
                    "unknown",
                    "Нет данных о нахождении V в машине."
                )

            if game_state["v_in_car"] is not True:
                return self._result(
                    "not_achieved",
                    "V пока не находится в машине.",
                    observed=game_state["v_in_car"]
                )

            expected_role = normalized["seat_role"]

            if expected_role != "any":
                if "seat_role" not in game_state:
                    return self._result(
                        "unknown",
                        "Неизвестно, на каком месте находится V."
                    )

                actual_role = game_state["seat_role"]

                if actual_role != expected_role:
                    return self._result(
                        "not_achieved",
                        "V находится не на нужном месте.",
                        expected=expected_role,
                        observed=actual_role
                    )

            return self._result(
                "achieved",
                "V успешно находится в машине.",
                observed={
                    "v_in_car": True,
                    "seat_role": game_state.get("seat_role")
                }
            )

        # ----------------------------------------------------
        # ВЫХОД ИЗ МАШИНЫ
        # ----------------------------------------------------

        if goal_type == "exit_vehicle":
            required = (
                "v_in_car",
                "is_entering_car",
                "seat_role"
            )

            missing = [
                key for key in required
                if key not in game_state
            ]

            if missing:
                return self._result(
                    "unknown",
                    "Недостаточно данных для проверки выхода.",
                    missing=missing
                )

            achieved = (
                game_state["v_in_car"] is False
                and game_state["is_entering_car"] is False
                and game_state["seat_role"] == "none"
            )

            if achieved:
                return self._result(
                    "achieved",
                    "V вышла из машины."
                )

            return self._result(
                "not_achieved",
                "V ещё не завершила выход из машины."
            )

        # ----------------------------------------------------
        # СЛЕДОВАНИЕ ЗА ИГРОКОМ
        # ----------------------------------------------------

        if goal_type == "follow_player":
            return self._check_boolean(
                game_state,
                "v_following",
                True,
                "V следует за игроком.",
                "V пока не следует за игроком."
            )

        # ----------------------------------------------------
        # ОСТАНОВКА СЛЕДОВАНИЯ
        # ----------------------------------------------------

        if goal_type == "stop_following":
            return self._check_boolean(
                game_state,
                "v_following",
                False,
                "V прекратила следование.",
                "V всё ещё следует за игроком."
            )

        # ----------------------------------------------------
        # НАВИГАЦИЯ
        # ----------------------------------------------------

        if goal_type == "navigate_to":
            if "navigation_outcome" not in game_state:
                return self._result(
                    "unknown",
                    "Нет информации о результате навигации."
                )

            outcome = game_state["navigation_outcome"]

            if outcome in ("failed", "cancelled"):
                return self._result(
                    "not_achieved",
                    "Навигация завершилась без достижения цели.",
                    observed=outcome
                )

            if outcome != "succeeded":
                return self._result(
                    "not_achieved",
                    "Навигация ещё не завершена.",
                    observed=outcome
                )

            expected_destination = normalized["destination"]

            actual_destination = game_state.get(
                "navigation_destination"
            )

            if not isinstance(actual_destination, str):
                return self._result(
                    "unknown",
                    "Неизвестен фактический пункт назначения."
                )

            if (
                actual_destination.strip().casefold()
                != expected_destination.casefold()
            ):
                return self._result(
                    "not_achieved",
                    "Завершена навигация не к нужному пункту.",
                    expected=expected_destination,
                    observed=actual_destination
                )

            return self._result(
                "achieved",
                "Навигация к нужному пункту завершена.",
                observed=actual_destination
            )

        # ----------------------------------------------------
        # ПОДХОД К КОНКРЕТНОМУ ОБЪЕКТУ
        # ----------------------------------------------------

        if goal_type == "approach_object":
            required = (
                "object_approach_id",
                "object_approach_outcome"
            )

            if any(key not in game_state for key in required):
                return self._result(
                    "unknown",
                    "Нет информации о результате подхода к объекту."
                )

            expected_id = normalized["object_id"]
            actual_id = game_state["object_approach_id"]
            outcome = game_state["object_approach_outcome"]

            if actual_id != expected_id:
                return self._result(
                    "not_achieved",
                    "V подошла не к выбранному объекту.",
                    expected=expected_id,
                    observed=actual_id
                )

            if outcome == "succeeded":
                return self._result(
                    "achieved",
                    "V подошла к выбранному объекту.",
                    object_id=actual_id
                )

            return self._result(
                "not_achieved",
                "V ещё не завершила подход к объекту.",
                outcome=outcome
            )

        # ----------------------------------------------------
        # ПОДБОР КОНКРЕТНОГО ОБЪЕКТА
        # ----------------------------------------------------

        if goal_type == "pickup_object":
            expected_id = normalized["object_id"]

            if "has_item" not in game_state:
                return self._result(
                    "unknown",
                    "Неизвестно, держит ли V предмет."
                )

            if "held_object_id" not in game_state:
                return self._result(
                    "unknown",
                    "Нет информации об ID предмета в руках V."
                )

            if game_state["has_item"] is not True:
                return self._result(
                    "not_achieved",
                    "V пока не держит предмет."
                )

            actual_id = game_state["held_object_id"]

            if actual_id != expected_id:
                return self._result(
                    "not_achieved",
                    "V держит не тот предмет.",
                    expected=expected_id,
                    observed=actual_id
                )

            return self._result(
                "achieved",
                "V держит именно выбранный предмет.",
                object_id=actual_id
            )

        # ----------------------------------------------------
        # ВОЗВРАЩЕНИЕ К ИГРОКУ С ПРЕДМЕТОМ
        # ----------------------------------------------------

        if goal_type == "return_to_player":
            required = (
                "has_item",
                "held_object_id",
                "player_distance_m"
            )

            missing = [
                key for key in required
                if key not in game_state
            ]

            if missing:
                return self._result(
                    "unknown",
                    "Недостаточно данных для проверки возвращения.",
                    missing=missing
                )

            expected_id = normalized["object_id"]

            if (
                game_state["has_item"] is not True
                or game_state["held_object_id"] != expected_id
            ):
                return self._result(
                    "not_achieved",
                    "V должна продолжать держать выбранный предмет.",
                    expected=expected_id,
                    observed=game_state["held_object_id"]
                )

            distance = game_state["player_distance_m"]

            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
            ):
                return self._result(
                    "unknown",
                    "Некорректная дистанция до игрока."
                )

            maximum = normalized["max_distance_m"]

            if distance > maximum:
                return self._result(
                    "not_achieved",
                    "V ещё не подошла достаточно близко к игроку.",
                    distance_m=distance,
                    maximum_m=maximum
                )

            return self._result(
                "achieved",
                "V вернулась к игроку с нужным предметом.",
                object_id=expected_id,
                distance_m=distance
            )

        # ----------------------------------------------------
        # ДОСТАВКА ПРЕДМЕТА ИГРОКУ
        # ----------------------------------------------------

        if goal_type == "bring_item":
            object_id = goal_spec.get("object_id", "")
            max_distance_m = float(
                goal_spec.get("max_distance_m", 2.5)
            )

            required_keys = (
                "has_item",
                "held_object_id",
                "player_distance_m",
                "last_released_object_id",
                "last_released_object_distance_m"
            )

            missing_keys = [
                key for key in required_keys
                if key not in game_state
            ]

            if missing_keys:
                return self._result(
                    "unknown",
                    "Недостаточно телеметрии для проверки доставки: "
                    + ", ".join(missing_keys)
                )

            has_item = game_state.get("has_item")
            held_id = game_state.get("held_object_id")
            player_distance = game_state.get("player_distance_m")

            released_id = game_state.get("last_released_object_id")
            released_distance = game_state.get(
                "last_released_object_distance_m"
            )

            # Вариант 1: V держит нужный предмет рядом с игроком.
            held_near_player = (
                has_item is True
                and held_id == object_id
                and isinstance(player_distance, (int, float))
                and not isinstance(player_distance, bool)
                and 0 <= player_distance <= max_distance_m
            )

            # Вариант 2: V отпустила или бросила нужный предмет
            # рядом с игроком.
            released_near_player = (
                has_item is False
                and released_id == object_id
                and isinstance(released_distance, (int, float))
                and not isinstance(released_distance, bool)
                and 0 <= released_distance <= max_distance_m
            )

            if held_near_player:
                return self._result(
                    "achieved",
                    "V доставила выбранный предмет и держит его рядом с игроком."
                )

            if released_near_player:
                return self._result(
                    "achieved",
                    "Выбранный предмет находится рядом с игроком."
                )

            return self._result(
                "not_achieved",
                "Выбранный предмет ещё не доставлен к игроку."
            )

        # ----------------------------------------------------
        # ПОДБОР ПРЕДМЕТА
        # ----------------------------------------------------

        if goal_type == "pickup_item":
            return self._check_boolean(
                game_state,
                "has_item",
                True,
                "V держит предмет.",
                "V пока не держит предмет."
            )

        # ----------------------------------------------------
        # ВЫКЛАДЫВАНИЕ ПРЕДМЕТА
        # ----------------------------------------------------

        if goal_type == "drop_item":
            return self._check_boolean(
                game_state,
                "has_item",
                False,
                "V больше не держит предмет.",
                "V всё ещё держит предмет."
            )

        # ----------------------------------------------------
        # ВЫСАДКА ИГРОКА С ПАССАЖИРСКОГО МЕСТА
        # ----------------------------------------------------

        if goal_type == "unseat_player":
            return self._check_boolean(
                game_state,
                "player_is_passenger",
                False,
                "Игрок больше не находится на пассажирском месте.",
                "Игрок всё ещё находится на пассажирском месте."
            )

        return self._result(
            "invalid",
            "Для этой цели проверка не реализована."
        )

    # --------------------------------------------------------
    # ПРОВЕРКА БУЛЕВЫХ СОСТОЯНИЙ
    # --------------------------------------------------------

    @staticmethod
    def _check_boolean(
        game_state: Dict[str, Any],
        field: str,
        expected: bool,
        success_message: str,
        failure_message: str
    ) -> Dict[str, Any]:

        if field not in game_state:
            return GoalVerifier._result(
                "unknown",
                "Нет данных для проверки поля: " + field
            )

        actual = game_state[field]

        if not isinstance(actual, bool):
            return GoalVerifier._result(
                "unknown",
                "Поле " + field + " имеет некорректный тип.",
                observed=actual
            )

        if actual is expected:
            return GoalVerifier._result(
                "achieved",
                success_message,
                observed=actual
            )

        return GoalVerifier._result(
            "not_achieved",
            failure_message,
            expected=expected,
            observed=actual
        )

    # --------------------------------------------------------
    # ЕДИНЫЙ ФОРМАТ РЕЗУЛЬТАТА
    # --------------------------------------------------------

    @staticmethod
    def _result(
        status: str,
        reason: str,
        **details
    ) -> Dict[str, Any]:

        result = {
            "status": status,
            "reason": reason
        }

        if details:
            result["details"] = details

        return result

if __name__ == "__main__":

    verifier = GoalVerifier()

    object_id = "object_-16904"

    tests = [
        (
            "approach_object",
            {
                "object_approach_id": object_id,
                "object_approach_outcome": "succeeded"
            }
        ),
        (
            "pickup_object",
            {
                "has_item": True,
                "held_object_id": object_id
            }
        ),
        (
            "return_to_player",
            {
                "has_item": True,
                "held_object_id": object_id,
                "player_distance_m": 2.0
            }
        ),
        (
            "bring_item",
            {
                "has_item": False,
                "held_object_id": "",
                "player_distance_m": 10.0,
                "last_released_object_id": "object_-16904",
                "last_released_object_distance_m": 1.1
            }
        )
    ]

    for goal_type, state in tests:
        goal = {
            "type": goal_type,
            "object_id": object_id
        }

        result = verifier.evaluate(goal, state)

        print(
            goal_type,
            "=>",
            result["status"],
            result["reason"]
        )