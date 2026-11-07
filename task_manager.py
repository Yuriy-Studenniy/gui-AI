"""Task Manager для динамического V-Agent.

Хранит исходную просьбу игрока, структурированную проверяемую цель
(если она есть) и журнал фактических результатов инструментов.
Не вызывает LLM и не исполняет игровые действия.

Старый API статических планов сохранён для совместимости, но динамический
AgentLoop не обязан создавать или выполнять фиксированный список этапов.
"""

from typing import Any, Dict, List, Optional

from .goal_verifier import GoalVerifier


class TaskManager:

    MAX_PLAN_STEPS = 12
    MAX_ACTION_HISTORY = 30

    def __init__(
        self,
        goal_verifier: Optional[GoalVerifier] = None
    ):
        self._verifier = (
            goal_verifier
            if goal_verifier is not None
            else GoalVerifier()
        )

        # Динамическая память задачи — основной API.
        self._original_goal = ""
        self._goal_spec = None
        self._goal_spec_decided = False
        self._last_goal_verification = None
        self._actions: List[Dict[str, Any]] = []
        self._task_status = "empty"
        self._task_reason = ""

        # Legacy: поддержка прежних пользователей set_plan/verify_current.
        self._steps: List[Dict[str, Any]] = []
        self._current_index = 0
        self._status = "empty"

    # ========================================================
    # ДИНАМИЧЕСКАЯ ЗАДАЧА
    # ========================================================

    def start_task(self, original_goal: str = "") -> Dict[str, Any]:
        """Начинает новую задачу и очищает память предыдущей."""
        self.reset()
        self._original_goal = (
            original_goal.strip()
            if isinstance(original_goal, str)
            else str(original_goal or "")
        )
        self._task_status = "active"
        return self.get_progress()

    def set_original_goal(self, original_goal: str) -> None:
        self._original_goal = (
            original_goal.strip()
            if isinstance(original_goal, str)
            else str(original_goal or "")
        )
        if self._task_status == "empty":
            self._task_status = "active"

    def get_original_goal(self) -> str:
        return self._original_goal

    def normalize_goal_spec(self, goal_spec: Any) -> Optional[Dict[str, Any]]:
        """Нормализует структурированную цель через GoalVerifier."""
        return self._verifier.normalize_goal_spec(goal_spec)

    def set_goal_spec(self, goal_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Сохраняет опциональную формальную цель, если GoalVerifier её знает."""
        if goal_spec is None:
            self._goal_spec = None
            return self._result("ready", "Структурированная цель не задана.")

        normalized = self._verifier.normalize_goal_spec(goal_spec)
        if normalized is None:
            return self._result("invalid", "Некорректная структурированная цель.")

        if self._goal_spec is not None and normalized != self._goal_spec:
            return self._result(
                "conflict",
                "Нельзя незаметно изменить уже установленную цель.",
                current_goal_spec=self._goal_spec
            )

        self._goal_spec = normalized
        self._last_goal_verification = None
        return self._result("ready", "Структурированная цель сохранена.")

    def get_goal_spec(self) -> Optional[Dict[str, Any]]:
        return dict(self._goal_spec) if self._goal_spec is not None else None

    def verify_structured_goal(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Проверяет формальную цель, если она задана."""
        if self._goal_spec is None:
            return self._result(
                "not_configured",
                "Для этой составной задачи нет отдельной формальной цели."
            )

        verification = self._verifier.evaluate(
            self._goal_spec,
            game_state if isinstance(game_state, dict) else {}
        )
        self._last_goal_verification = verification
        return verification

    def mark_goal_spec_decided(self, decided: bool = True) -> None:
        """Фиксирует выбор режима: структурированная цель или составная задача."""
        self._goal_spec_decided = bool(decided)

    def is_goal_spec_decided(self) -> bool:
        return self._goal_spec_decided

    def record_action(
        self,
        tool: str,
        arguments: Optional[Dict[str, Any]] = None,
        status: str = "unknown",
        result_summary: Any = None,
        reason: Any = None
    ) -> Dict[str, Any]:
        """Записывает терминальный результат одного инструмента."""
        if not isinstance(tool, str) or not tool.strip():
            return self._result("invalid", "Не указано имя инструмента.")

        if isinstance(result_summary, str):
            summary = result_summary
        else:
            summary = str(result_summary) if result_summary is not None else ""

        if len(summary) > 1200:
            summary = summary[:1200] + "...[обрезано]"

        record = {
            "tool": tool.strip(),
            "arguments": dict(arguments) if isinstance(arguments, dict) else {},
            "status": str(status or "unknown"),
            "result_summary": summary,
            "reason": reason
        }

        self._actions.append(record)
        if len(self._actions) > self.MAX_ACTION_HISTORY:
            self._actions = self._actions[-self.MAX_ACTION_HISTORY:]

        return self._result(
            "recorded",
            "Результат действия добавлен в журнал.",
            action_count=len(self._actions)
        )

    def get_progress(self, limit: int = 12) -> Dict[str, Any]:
        """Возвращает контекст задачи для LLM и диагностики."""
        try:
            limit = max(0, int(limit))
        except (TypeError, ValueError):
            limit = 12

        recent = self._actions[-limit:] if limit else []
        counts = {
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "timeout": 0,
            "cancelled_by_agent": 0,
            "cancel_requested": 0,
            "other": 0
        }
        for action in self._actions:
            action_status = action.get("status", "other")
            if action_status in counts:
                counts[action_status] += 1
            elif action_status not in ("accepted", "running"):
                counts["other"] += 1

        return {
            "original_goal": self._original_goal,
            "goal_spec": self.get_goal_spec(),
            "goal_spec_decided": self._goal_spec_decided,
            "goal_verification": self._last_goal_verification,
            "task_status": self._task_status,
            "task_reason": self._task_reason,
            "action_count": len(self._actions),
            "action_counts": counts,
            "recent_actions": [dict(item) for item in recent],
            "last_action": dict(self._actions[-1]) if self._actions else None
        }

    def mark_status(self, status: str, reason: str = "") -> None:
        allowed = ("active", "completed", "abandoned", "waiting_for_player")
        if status not in allowed:
            raise ValueError("Недопустимый статус задачи: " + str(status))
        self._task_status = status
        self._task_reason = str(reason or "")

    # ========================================================
    # LEGACY: СТАТИЧЕСКИЕ ПЛАНЫ (ДЛЯ СОВМЕСТИМОСТИ)
    # ========================================================

    def set_plan(self, plan: Any) -> Dict[str, Any]:
        if not isinstance(plan, list):
            return self._result("invalid", "План должен быть списком этапов.")
        if not plan:
            return self._result("invalid", "План не может быть пустым.")
        if len(plan) > self.MAX_PLAN_STEPS:
            return self._result("invalid", "План превышает максимально допустимое количество этапов.")

        prepared_steps = []
        used_ids = set()
        for index, step in enumerate(plan):
            if not isinstance(step, dict):
                return self._result("invalid", "Этап " + str(index + 1) + " должен быть объектом.")

            step_id = step.get("id", "step_" + str(index + 1))
            if not isinstance(step_id, str) or not step_id.strip():
                return self._result("invalid", "Некорректный идентификатор этапа.")
            step_id = step_id.strip()
            if step_id in used_ids:
                return self._result("invalid", "Повторяется идентификатор этапа: " + step_id)
            used_ids.add(step_id)

            description = step.get("description", "")
            if not isinstance(description, str):
                return self._result("invalid", "Описание этапа должно быть строкой.")

            goal_spec = self._verifier.normalize_goal_spec(step.get("goal_spec"))
            if goal_spec is None:
                return self._result("invalid", "Некорректная цель этапа: " + step_id)

            prepared_steps.append({
                "id": step_id,
                "description": description,
                "goal_spec": goal_spec,
                "status": "pending",
                "verification": None
            })

        self._steps = prepared_steps
        self._current_index = 0
        self._status = "active"
        self._task_status = "active"
        return self._result("ready", "План создан.", plan=self.get_state())

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        if self._status != "active" or self._current_index >= len(self._steps):
            return None
        step = self._steps[self._current_index]
        return {
            "index": self._current_index,
            "total": len(self._steps),
            "id": step["id"],
            "description": step["description"],
            "goal_spec": dict(step["goal_spec"]),
            "status": step["status"]
        }

    def verify_current(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if self._status == "completed":
            return self._result("plan_completed", "Все этапы плана уже завершены.", plan=self.get_state())
        if self._status != "active":
            return self._result("no_plan", "Нет активного плана.")

        step = self._steps[self._current_index]
        verification = self._verifier.evaluate(step["goal_spec"], game_state)
        step["verification"] = verification

        if verification["status"] == "achieved":
            step["status"] = "completed"
            completed_step = step["id"]
            self._current_index += 1
            if self._current_index >= len(self._steps):
                self._status = "completed"
                self._task_status = "completed"
                return self._result(
                    "plan_completed", "Все этапы успешно выполнены.",
                    completed_step=completed_step,
                    verification=verification,
                    plan=self.get_state()
                )
            return self._result(
                "step_completed", "Текущий этап успешно выполнен.",
                completed_step=completed_step,
                next_step=self.get_current_step(),
                verification=verification,
                plan=self.get_state()
            )

        if verification["status"] == "not_achieved":
            return self._result(
                "not_achieved", verification["reason"],
                current_step=self.get_current_step(),
                verification=verification
            )
        return self._result(
            "unknown", verification["reason"],
            current_step=self.get_current_step(),
            verification=verification
        )

    def abandon(self, reason: str) -> Dict[str, Any]:
        if self._status == "empty" and self._task_status == "empty":
            return self._result("no_plan", "Нет активной задачи.")
        if self._steps:
            self._status = "abandoned"
        self.mark_status("abandoned", reason)
        return self._result("abandoned", reason, state=self.get_state())

    def get_state(self) -> Dict[str, Any]:
        """Полное состояние: динамическая задача и legacy-план, если он есть."""
        state_status = (
            self._status if self._steps else self._task_status
        )
        return {
            "status": state_status,
            "original_goal": self._original_goal,
            "goal_spec": self.get_goal_spec(),
            "goal_spec_decided": self._goal_spec_decided,
            "goal_verification": self._last_goal_verification,
            "action_count": len(self._actions),
            "recent_actions": [dict(item) for item in self._actions[-12:]],
            "current_index": self._current_index,
            "total_steps": len(self._steps),
            "steps": [
                {
                    "id": step["id"],
                    "description": step["description"],
                    "goal_spec": dict(step["goal_spec"]),
                    "status": step["status"],
                    "verification": step["verification"]
                }
                for step in self._steps
            ]
        }

    def reset(self) -> None:
        self._original_goal = ""
        self._goal_spec = None
        self._goal_spec_decided = False
        self._last_goal_verification = None
        self._actions = []
        self._task_status = "empty"
        self._task_reason = ""
        self._steps = []
        self._current_index = 0
        self._status = "empty"

    @staticmethod
    def _result(status: str, reason: str, **details) -> Dict[str, Any]:
        result = {"status": status, "reason": reason}
        if details:
            result["details"] = details
        return result
