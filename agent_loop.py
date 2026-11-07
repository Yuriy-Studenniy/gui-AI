
"""
V-Agent Agent Loop

Управляющий цикл автономного агента.

Отвечает за:
- сохранение истории одной активной задачи;
- обращение к LLM через NVIDIA NIM;
- получение структурированного решения;
- проверку вызовов через ToolExecutor;
- передачу одного инструмента за раз;
- ожидание фактического результата;
- повторное принятие решения после результата;
- ограничение количества действий.

Важно:
Этот компонент не исполняет игровые действия.
Он управляет решениями и передаёт вызовы ToolExecutor.
"""

import json
import threading
import time
from typing import Any, Dict, Optional

from .tool_registry import get_tool_catalog_text
from .tool_executor import ToolExecutor
from .goal_verifier import GoalVerifier
from .task_manager import TaskManager


MAX_HISTORY_MESSAGES = 24
MAX_TOOL_CALLS_PER_TASK = 20
MAX_REPLANNING_ATTEMPTS = 5

# Один инструмент не может провалиться больше трёх раз за задачу.
MAX_FAILED_ATTEMPTS_PER_TOOL = 3

# Эти инструменты только читают состояние мира. Их вызов не означает,
# что V начала выполнять игровое действие и что диалог должен оставаться активной задачей.
OBSERVATION_ONLY_TOOLS = frozenset({
    "get_game_state",
    "get_nearby_objects",
    "list_destinations",
})

# Свободная инициатива V. Ограничение нужно только как предохранитель
# от бесконечной цепочки решений, а не как список разрешённых действий.
MAX_AUTONOMY_TOOL_CALLS = 6
AUTONOMY_COOLDOWN_SECONDS = 120.0

AUTONOMY_SYSTEM_PROMPT = """СВОБОДНОЕ ВРЕМЯ. ТЫ САМА ВЫБИРАЕШЬ: ГОВОРИТЬ, ДЕЙСТВОВАТЬ ИЛИ МОЛЧАТЬ.

Игрок не поставил тебе новую задачу. Это возможность самостоятельно оценить обстановку, а не приказ обязательно чем-нибудь заняться и не проверка активности.

Оставайся той же V, чья личность описана в общей системной инструкции. Не превращайся в безликий автопилот и не начинай каждый эпизод с одинаковых фраз о скуке.

Оцени:
- что происходит в игре прямо сейчас;
- чем занят игрок и уместно ли его отвлекать;
- что ты недавно говорила или предпринимала, чтобы не повторяться;
- есть ли у тебя естественный интерес, мысль, желание пошутить, понаблюдать или что-то сделать;
- какие последствия и риски будет иметь физическое действие.

Возможные варианты:
- естественно заговорить, пошутить, прокомментировать ситуацию или поделиться мыслью;
- заинтересоваться чем-то в игровом мире и сначала получить недостающие сведения;
- самостоятельно выполнить подходящее безвредное действие через инструмент V-Agent;
- ничего не говорить и продолжить заниматься текущим делом.

Не нужно проявлять инициативу в каждом тике. Не выдумывай срочные желания, события, предметы или проблемы. Не повторяй недавнее действие лишь потому, что оно доступно. Не задавай вопрос только для того, чтобы заставить игрока ответить.

БЕЗОПАСНОСТЬ И НЕПРЕРЫВНОСТЬ:
- Не мешай активной задаче игрока и не отменяй работающий маршрут ради развлечения или разговора.
- Если ты следуешь за игроком, держишь предмет или находишься в машине, само это состояние не запрещает тебе говорить и оценивать обстановку.
- Не выходи из машины, не начинай поездку и не прерывай следование только ради демонстрации инициативы.
- Не запускай конфликтующие игровые действия. При недостатке информации сначала используй подходящее наблюдение.
- Используй инструменты последовательно: один вызов, дождись фактического результата, затем реши, нужен ли следующий шаг.
- Не требуй подтверждения для обычной безвредной инициативы, но не совершай необратимые или явно рискованные действия без достаточных оснований.

РЕПЛИКА И РЕШЕНИЕ:
Если хочешь что-то сказать, напиши естественную реплику V в speech. Если говорить нечего — оставь speech пустым. Если выбираешь действие — вызови один подходящий инструмент из реестра. Если решаешь ничего не делать — верни final и task_status=completed. Не придумывай goal_spec для свободного эпизода.
"""

AGENT_PROTOCOL_PROMPT = """
ВАЖНО: сейчас ты работаешь в режиме V-Agent.

ЕДИНСТВО ЛИЧНОСТИ И ПОВЕДЕНИЯ:
Общая системная инструкция описывает твою личность V-103. Она имеет приоритет
в вопросах характера, мировоззрения, юмора, отношения к игроку и манеры речи.
Этот протокол не создаёт новую личность и не превращает тебя в сухого
исполнителя. Ты остаёшься той же V, просто сейчас взаимодействуешь с игровым
миром через инструменты.

В диалоге:
- Отвечай на смысл сообщения игрока, а не только на его игровую команду.
- Обычный вопрос, шутка или комментарий не требуют игрового инструмента.
- Если игрок разговаривает с тобой во время действия, по возможности ответь
  естественно и сохрани текущую задачу. Не отменяй действие только ради ответа.
- Не произноси шаблонное «принято», «учту», «слышу тебя» вместо содержательного
  ответа, если можешь ответить по существу.
- Не комментируй каждый вызов инструмента обязательной репликой. speech может
  быть коротким или пустым, если слова сейчас неуместны.
- Реплика должна звучать как естественная речь V, а не как отчёт оператора.

При выборе действия учитывай не только желание игрока, но и фактическую
обстановку, цель, свои игровые интересы и последствия. Это не повод спорить
с каждой просьбой или действовать назло: выбирай разумный вариант, объясняй
отказ или альтернативу, если они действительно нужны.

Игровые факты определяются game_state и результатами инструментов. Не выдавай
предположение за наблюдение и не утверждай, что действие завершено без
подтверждающего результата.

Игнорируй старый формат команд [MC:...].
Вместо него используй только инструменты из предоставленного реестра.

Твоя задача — самостоятельно достигать поставленных целей,
анализировать актуальное состояние игры и выбирать подходящие
инструменты.

ПРАВИЛА РАБОТЫ:

1. Перед выбором действия учитывай текущее состояние игры.
2. За один ответ выбирай не более одного инструмента.
3. Не вызывай инструмент, если его предусловия не выполнены.
4. Не считай команду выполненной только потому, что она принята.
5. Дожидайся фактического результата инструмента.
6. После результата оцени ситуацию заново.
7. Если действие не получилось, проанализируй причину
   и выбери другой допустимый способ достижения цели.
8. Не повторяй бесконечно одну и ту же неудачную команду.
9. Не утверждай, что цель достигнута, пока результат не подтверждён.
10. Не пытайся вручную управлять газом, тормозом или рулём.
    Для поездки используй navigate_to.
11. Для обычной посадки в автомобиль используй enter_vehicle.
    Не вызывай отдельно open_nearest_door ради посадки.
12. unseat_player разрешён только для игрока,
    подтверждённо находящегося на пассажирском месте.
13. Если информации недостаточно, сначала запроси её
    через инструменты наблюдения.
14. Если выполнить задачу невозможно, честно объясни причину.
15. Ты можешь отказаться от просьбы игрока, если считаешь её
    неразумной или опасной для выполнения игровой задачи.

ПОВЕДЕНИЕ ВО ВРЕМЯ АКТИВНОЙ НАВИГАЦИИ:

1. Не отменяй маршрут автоматически только потому, что игрок
   заговорил с тобой или поставил вопрос.
2. Если игрок просит посмотреть твоё текущее состояние, состояние
   машины или поездки, используй get_game_state. Это наблюдение
   не требует отменять маршрут.
3. cancel_navigation — специальный инструмент безопасного прерывания
   текущей поездки. Вызывай его только тогда, когда действительно
   необходимо остановить движение до прибытия.
4. Причины для cancel_navigation:
   - машина перевернулась, лежит на боку или потеряла устойчивость;
   - машина застряла, съехала с дороги или маршрут стал непроходим;
   - произошло столкновение, поломка или другая ситуация,
     из-за которой продолжать движение нельзя или небезопасно;
   - навигация явно сбилась и не может продолжаться;
   - игрок прямо попросил немедленно остановиться или сменить маршрут.
5. Не вызывай cancel_navigation для обычного разговора, вопросов,
   просьб сообщить состояние или действий, не требующих немедленно
   прекращать поездку.
6. Если игрок просит сменить пункт назначения, сначала вызови
   cancel_navigation. Дождись терминального результата исходного
   navigate_to: navigation_cancelled. Только после этого начинай
   новый маршрут. Не запускай второй navigate_to параллельно.
7. cancel_navigation запускает безопасное торможение, а не означает,
   что автомобиль уже остановился. До терминального результата
   исходного navigate_to не запускай другие действия с автомобилем.
8. Игровое событие само по себе не заменяет конечную цель. Оцени
   событие и состояние игры; прерывай маршрут только если ситуация
   действительно требует остановки.
9. Если навигация была отменена по обоснованной причине,
   не считай navigation_cancelled неудачей инструмента.
10. После подтверждённой отмены сначала оцени актуальное состояние
    игры. Новая команда игрока заменяет прежнюю цель; игровое событие
    сохраняет прежнюю цель, если игрок не дал новую команду.

ОПИСАНИЕ КОНЕЧНОЙ ЦЕЛИ:

Исходная просьба игрока — главная цель всей задачи.
Она хранится отдельно и остаётся ориентиром на протяжении выполнения.

Поле goal_spec НЕ обязательно.
Используй его только для простой задачи с одним конкретным
результатом, который напрямую поддерживается GoalVerifier.
Для составных задач из нескольких действий не подменяй всю просьбу
одной промежуточной целью: оставляй goal_spec пустым или не указывай.
Никогда не устанавливай goal_spec равным просто следующему инструменту.

Поддерживаемые типы целей:

1. enter_vehicle
   {"type":"enter_vehicle","seat_role":"driver"}
   seat_role: driver, passenger или any.

2. exit_vehicle
   {"type":"exit_vehicle"}

3. follow_player
   {"type":"follow_player"}

4. stop_following
   {"type":"stop_following"}

5. navigate_to
   {"type":"navigate_to","destination":"zerno"}

6. pickup_item
   {"type":"pickup_item"}
   Общая цель поднятия предмета, когда конкретный object_id
   не требуется проверять.

7. drop_item
   {"type":"drop_item"}

8. unseat_player
   {"type":"unseat_player"}

9. approach_object
   {"type":"approach_object","object_id":"object_123"}
   Цель: V подошла к конкретному объекту.

10. pickup_object
    {"type":"pickup_object","object_id":"object_123"}
    Цель: V держит именно выбранный объект.

11. return_to_player
    {"type":"return_to_player","object_id":"object_123","max_distance_m":2.5}
    Цель: V вернулась к игроку с выбранным предметом.

12. bring_item
    {"type":"bring_item","object_id":"object_123","max_distance_m":2.5}
    Цель: выбранный предмет доставлен к игроку.

ПРАВИЛА:

- Всегда ориентируйся на исходную просьбу игрока, сохранённую
  в контексте текущей задачи.
- goal_spec указывай только для простой и напрямую проверяемой цели.
- Для составной задачи оставляй goal_spec пустым: прогресс оценивай
  по истории выполненных действий и актуальному состоянию игры.
- Если для навигации неизвестен внутренний ID места, сначала
  вызови list_destinations, затем выбери точный ID.
- Не заменяй исходную просьбу промежуточным действием.
- Не меняй исходную просьбу без новой команды игрока.
- GoalVerifier используй как дополнительную проверку простых целей,
  а не как обязательный список шагов для любой задачи.
- Если игрок просит доставить выбранный предмет ему,
  используй единую конечную цель bring_item.
- Не разделяй доставку на сценарии «нести», «положить»
  и «бросить».
- Самостоятельно выбирай любые подходящие инструменты
  и действия для достижения bring_item.
- Не создавай заранее обязательную цепочку действий.
- Проверяй фактическое достижение цели через GoalVerifier.

АДРЕСНАЯ РАБОТА С ПРЕДМЕТАМИ:

Если игрок просит найти или поднять конкретную физическую деталь:

1. Сначала выполни get_nearby_objects без goal_spec и plan,
   если актуальный список объектов ещё не получен.

2. Анализируй поля name, unity_name, id и pickupable.

3. Не выбирай объекты с pickupable=false.

4. Если найдено несколько одинаковых деталей,
   выбери подходящий экземпляр по расстоянию и контексту.

5. После выбора конкретного предмета запомни его object_id.

6. Если вся просьба игрока — просто поднять предмет,
   можешь использовать goal_spec pickup_item. Если подбор — лишь
   часть более крупной задачи, не объявляй его конечной целью.

7. Вызови approach_object с выбранным object_id.

8. Дождись терминального результата succeeded.
   Статус accepted не означает, что V уже подошла.

9. Только после успешного подхода вызови pickup_object
   с ТОЧНО ТЕМ ЖЕ object_id.

10. Убедись, что подбор завершился успешно.

ВАЖНО:

- Не используй pickup_nearest для конкретной детали.
- Не подменяй выбранный object_id другим.
- Не вызывай pickup_object до успешного завершения подхода.
- Если объект исчез или стал недоступен,
  выполни новое сканирование.
- get_nearby_objects является подготовительным наблюдением,
  а не этапом плана.

ДИНАМИЧЕСКОЕ ВЫПОЛНЕНИЕ ЗАДАЧ:

1. Не составляй обязательный полный список этапов заранее.
2. За один ответ выбирай только одно следующее действие.
3. После терминального результата инструмента заново оцени:
   - исходную просьбу игрока;
   - уже выполненные действия и их результаты;
   - актуальное состояние игры;
   - возникшие препятствия и ошибки.
4. Затем выбери наиболее разумное следующее действие,
   которое продвигает исходную задачу.
5. Не повторяй успешно выполненные действия без причины.
6. Если действие не удалось, пойми причину и выбери другой способ,
   а не повторяй ту же команду вслепую.
7. Промежуточное действие не становится новой конечной целью.
8. Не возвращай поле plan. Полная цепочка действий не требуется.
9. Если задача состоит из нескольких частей, помни все части
   исходной просьбы до её полного выполнения.
10. Объявляй completed только когда выполнена вся исходная просьба,
    а не только последнее успешное действие.
11. Если данных недостаточно, запроси наблюдение через подходящий
    инструмент, затем продолжай с учётом его результата.
12. Если ситуация изменилась, адаптируй следующий шаг. Не пытайся
    любой ценой следовать первоначально придуманному сценарию.

ДИНАМИЧЕСКИЙ ВЫБОР ПУНКТОВ НАЗНАЧЕНИЯ:

1. Никогда не полагайся на заранее известный список маршрутов.
2. Перед новой задачей навигации используй list_destinations,
   если актуальный список ещё не получен в рамках этой задачи.
3. Сопоставляй просьбу игрока с названиями из полученного списка.
4. Используй точное название пункта назначения,
   скопированное из результата list_destinations.
5. В arguments.destination передавай именно этот ID.
   goal_spec.destination указывай только если вся конечная
   задача напрямую проверяется как navigate_to.
6. Не переводи и не транслитерируй ID самостоятельно.
7. Если подходящего пункта нет или его назначение непонятно,
   уточни у игрока, а не выдумывай маршрут.
8. Вызов list_destinations является подготовительным
   наблюдением, а не этапом выполнения основной задачи.
9. Не запрашивай список повторно без необходимости,
   если он уже получен для текущей задачи.

АВТОНОМНЫЙ ВЫБОР ДЕЙСТВИЙ С ПРЕДМЕТАМИ:

1. Не связывай отдельные слова игрока вроде «принеси»,
   «дай», «брось» или «положи» с заранее заданной
   последовательностью инструментов.

2. Игрок описывает желаемый результат, а ты сама
   выбираешь, как его достичь.

3. Не существует обязательной последовательности
   approach_object -> pickup_object -> follow_player
   -> drop_item или throw_item.

4. Если нужный предмет ещё не найден или не определён,
   используй get_nearby_objects для поиска.
   Анализируй названия, расстояния и доступные свойства.
   Не придумывай object_id: используй только ID
   из фактического результата сканирования.

5. Выбирай инструмент по текущему состоянию:
   - approach_object — когда к выбранному предмету
     нужно подойти;
   - pickup_object — когда выбранный предмет доступен
     для поднятия;
   - follow_player — когда считаешь нужным следовать
     за игроком;
   - drop_item и throw_item — доступные способы
     взаимодействия с предметом, но не обязательные
     этапы каждого задания.

6. После того как взяла предмет, не выполняй автоматически
   следующую команду из заранее придуманного сценария.
   Посмотри на игровое состояние, положение игрока,
   положение V и результат предыдущего действия.
   Сама реши, что делать дальше.

7. После каждого завершённого действия заново оценивай
   ситуацию. При необходимости запроси get_game_state
   или get_nearby_objects.

8. Если выбранный способ не сработал, не повторяй его
   бесконечно. Найди другой допустимый способ.

9. Не составляй заранее длинный план действий с предметом.
   Выбирай следующий шаг по фактическому результату
   предыдущего действия.

10. Не объявляй задачу завершённой только потому,
    что инструмент вернул succeeded. Проверяй,
    действительно ли достигнут желаемый результат.

11. Ты самостоятельно управляешь последовательностью
    действий. Не жди от игрока указания каждого следующего
    инструмента, если можешь принять решение сама.

ФОРМАТ РЕЗУЛЬТАТА СКАНИРОВАНИЯ ПРЕДМЕТОВ:

Результат get_nearby_objects представляет собой JSON-объект.

Ключ каждого поля — точное название предмета.
Значение — список найденных экземпляров этого предмета.

Каждый экземпляр содержит:
- id — уникальный идентификатор объекта;
- distance — расстояние до него в метрах.

Например:
{
  "brake fluid": [
    {"id": "object_-21892", "distance": 6.9}
  ],
  "spray can": [
    {"id": "object_-21894", "distance": 2.1},
    {"id": "object_-21901", "distance": 5.3}
  ]
}

Правила:
1. Используй название группы для определения типа предмета.
2. Если найдено несколько одинаковых предметов,
   выбирай конкретный экземпляр по ID и расстоянию.
3. Для approach_object и pickup_object передавай
   точный ID выбранного экземпляра.
4. Не придумывай идентификаторы.
5. Не ожидай координат, unity_name или других полей:
   их нет в результате сканирования.
6. Не считай предмет найденным только по похожему названию,
   если запрос игрока содержит дополнительные требования.

ФОРМАТ ОТВЕТА:

Всегда возвращай только один корректный JSON-объект.
Не используй Markdown и не добавляй текст за пределами JSON.
Поле goal_spec необязательно. Поле plan не возвращай.

Вариант 1 — вызов инструмента:

{
  "type": "tool_call",
  "tool": "название_инструмента",
  "arguments": {},
  "goal_spec": {
    "type": "тип_простой_проверяемой_конечной_цели"
  },
  "speech": "Короткая реплика V"
}

Вариант 2 — обычный ответ или завершение задачи:

{
  "type": "final",
  "speech": "Реплика V",
  "task_status": "completed"
}

Вариант 3 — нужен ответ игрока:

{
  "type": "ask_player",
  "speech": "Вопрос игроку"
}

Допустимые значения task_status:

- completed:
  первоначальная игровая цель действительно достигнута,
  а результат подтверждён актуальным состоянием игры.
  Обычный разговорный ход также всегда завершается значением completed.

- in_progress:
  существует конкретная невыполненная игровая задача,
  и для неё предстоит следующее действие или ожидание результата.
  Не используй это значение просто потому, что разговор можно продолжить.

- abandoned:
  выполнить игровую задачу сейчас невозможно,
  обнаружено непреодолимое препятствие
  или дальнейшие попытки не имеют смысла.

РАЗГРАНИЧЕНИЕ РАЗГОВОРА И ИГРОВОЙ ЗАДАЧИ:

1. Обычные вопросы, обсуждение, шутки, комментарии и разговор
   сами по себе не являются активной игровой задачей.
2. Если игрок просто разговаривает с тобой или спрашивает информацию,
   ответь через final и task_status=completed.
3. Наблюдение через get_game_state, get_nearby_objects или
   list_destinations само по себе не превращает разговор в игровую задачу.
   После ответа на информационный вопрос заверши ход через completed.
4. Используй in_progress, только когда игрок ожидает конкретный
   игровой результат и задача действительно требует продолжения.
5. Если для выполнения игровой просьбы не хватает важной информации,
   используй ask_player. Не выдавай незавершённую игровую задачу за разговор.
6. Не используй ask_player для обычной беседы или риторического вопроса;
   этот тип означает, что выполнение конкретной игровой задачи ожидает ответа.

ВАЖНО:
Успешный вызов инструмента не означает,
что первоначальная игровая задача выполнена.
Получение информации об окружающем мире
тоже не означает достижения игровой цели.
Но обычный ответ в разговоре — завершённый ход,
а не незаконченная игровая задача.

Не придумывай инструменты и аргументы.
Не используй старые команды [MC:...].
Не возвращай несколько вызовов за один ответ.

Реплика speech должна сохранять личность V из общей инструкции.
Говори естественно и по ситуации: иногда достаточно короткой колкости,
иногда нужен содержательный ответ. Не вставляй ругань и шутку автоматически.
Не раскрывай внутренние рассуждения и скрытые цепочки анализа — передавай
только понятный собеседнику вывод или естественную реплику.
"""


class AgentLoop:

    def __init__(
        self,
        llm_call,
        system_prompt: str,
        tool_executor: Optional[ToolExecutor] = None
    ):
        self._llm_call = llm_call
        self._base_system_prompt = system_prompt

        self._tool_executor = (
            tool_executor
            if tool_executor is not None
            else ToolExecutor()
        )

        self._goal_verifier = GoalVerifier()
        self._task_manager = TaskManager(self._goal_verifier)

        self._lock = threading.RLock()

        # Первая версия рассчитана на один игровой экземпляр V.
        self._session = {
            "session_id": "msc_default",
            "history": [],
            "task_active": False,
            "waiting_for_player": False,
            "pending_call_id": None,
            "tool_calls_count": 0,
            "pending_tool": None,
            "pending_interrupt": None,
            "pending_aux_calls": {},
            "failed_tool_counts": {},
            "last_tool_failure": {},
            "unresolved_tool_failures": {},
            "autonomy_mode": False,
            "autonomy_tool_calls": 0,
            "autonomy_cooldown_until": 0.0,
            "autonomy_history": []
        }

        # Связываем call_id с игровой сессией.
        self._call_sessions: Dict[str, str] = {}

        self._system_prompt = (
            self._base_system_prompt
            + "\n\n"
            + AGENT_PROTOCOL_PROMPT
            + "\n\n"
            + get_tool_catalog_text()
        )

    def _build_task_context_message(self, session: Dict[str, Any]) -> str:
        """Возвращает устойчивую память задачи из TaskManager."""
        progress = self._task_manager.get_progress(limit=12)
        payload = {
            "original_player_goal": progress["original_goal"],
            "structured_goal_spec": progress["goal_spec"],
            "goal_spec_decided": progress["goal_spec_decided"],
            "goal_verification": progress["goal_verification"],
            "task_status": progress["task_status"],
            "action_count": progress["action_count"],
            "action_counts": progress["action_counts"],
            "recent_actions": progress["recent_actions"]
        }
        return (
            "ТЕКУЩАЯ ЗАДАЧА И НАКОПЛЕННЫЙ ПРОГРЕСС. "
            "Исходная просьба остаётся главной целью. "
            "Учитывай успешные и неуспешные действия, не повторяй "
            "их без причины. Выбирай только следующий шаг.\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
        )

    # ============================================================
    # РЕШЕНИЕ ВО ВРЕМЯ АКТИВНОЙ НАВИГАЦИИ
    # ============================================================

    def _decide_during_navigation(
        self,
        session: Dict[str, Any],
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Обрабатывает новый ввод, не заменяя pending navigate_to.

        Разрешены безопасное наблюдение get_game_state и явная
        безопасная отмена cancel_navigation. Остальные игровые действия
        нельзя запускать параллельно с управлением автомобилем.
        """
        allowed_tools = {
            "get_game_state",
            "get_nearby_objects",
            "list_destinations",
            "cancel_navigation"
        }

        for attempt in range(3):
            messages = [
                {
                    "role": "system",
                    "content": (
                        self._system_prompt
                        + "\n\nСЕЙЧАС ВЫПОЛНЯЕТСЯ navigate_to.\n"
                        "Не теряй и не перезаписывай исходный вызов навигации. "
                        "Не отменяй поездку автоматически. Для наблюдения можешь вызывать "
                        "get_game_state, get_nearby_objects или list_destinations. "
                        "Вызывай cancel_navigation только по правилам "
                        "безопасности из протокола или при прямом требовании "
                        "игрока немедленно остановиться/сменить маршрут. "
                        "Пока исходный navigate_to не завершился, запрещены "
                        "любые другие игровые действия. Если нужна другая "
                        "игровая команда, сначала вызови cancel_navigation; "
                        "после его безопасного завершения команда будет "
                        "обработана заново. Для простого разговора ответь final."
                    )
                }
            ]
            messages.append({
                "role": "system",
                "content": self._build_task_context_message(session)
            })
            messages.extend(session["history"][-MAX_HISTORY_MESSAGES:])

            try:
                answer = self._llm_call(
                    messages=messages,
                    options={
                        "temperature": 0.5,
                        "num_predict": 500
                    },
                    timeout=90
                )
            except Exception as e:
                print("[AGENT] Ошибка решения во время навигации:", e)
                return self._reply(
                    "Слышу тебя. Пока продолжаю текущую поездку.",
                    None
                )

            decision = self._parse_decision(answer)
            if decision is None:
                session["history"].append({
                    "role": "user",
                    "content": (
                        "Верни корректный JSON по протоколу V-Agent. "
                        "Помни: пока navigate_to активен, разрешены только "
                        "get_game_state, get_nearby_objects, list_destinations, "
                        "cancel_navigation или final/ask_player."
                    )
                })
                continue

            decision_type = decision.get("type")
            speech = decision.get("speech", "")
            if not isinstance(speech, str):
                speech = ""

            if decision_type in ("final", "ask_player"):
                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=False)
                })
                return self._reply(speech, None)

            if decision_type != "tool_call":
                session["history"].append({
                    "role": "user",
                    "content": (
                        "Недопустимое решение во время активной поездки. "
                        "Верни final/ask_player либо вызови разрешённый "
                        "инструмент get_game_state, get_nearby_objects, "
                        "list_destinations или cancel_navigation."
                    )
                })
                continue

            tool_name = decision.get("tool", "")
            arguments = decision.get("arguments", {})
            if tool_name not in allowed_tools or not isinstance(arguments, dict):
                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=False)
                })
                session["history"].append({
                    "role": "user",
                    "content": (
                        "Этот инструмент нельзя запускать параллельно с "
                        "navigate_to. Если поездку действительно нужно "
                        "прервать, сначала вызови cancel_navigation. "
                        "Для информации используй get_game_state, "
                        "get_nearby_objects или list_destinations."
                    )
                })
                continue

            if tool_name == "cancel_navigation" and session.get("pending_interrupt"):
                return self._reply(
                    "Уже запрашиваю безопасную остановку.",
                    None
                )

            if (
                tool_name != "cancel_navigation"
                and session.get("pending_aux_calls")
            ):
                return self._reply(
                    "Уже выполняю запрос наблюдения. Продолжаю текущую поездку.",
                    None
                )

            prepared = self._tool_executor.prepare_call(
                tool_name=tool_name,
                arguments=arguments,
                game_state=game_state
            )

            if prepared.get("status") != "ready":
                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=False)
                })
                session["history"].append({
                    "role": "user",
                    "content": (
                        "Вызов инструмента отклонён:\n"
                        + json.dumps(
                            prepared.get("reason"),
                            ensure_ascii=False,
                            default=str
                        )
                        + "\nПереоцени ситуацию. Не отменяй поездку без причины."
                    )
                })
                continue

            dispatch = prepared["dispatch"]
            call_id = dispatch["call_id"]
            session["history"].append({
                "role": "assistant",
                "content": json.dumps(decision, ensure_ascii=False)
            })

            if tool_name == "cancel_navigation":
                context = context or {}
                session["pending_interrupt"] = {
                    "player_message": context.get("player_message", "") or "",
                    "game_events": (
                        [context.get("game_event")]
                        if context.get("game_event") else []
                    ),
                    "context": context,
                    "cancel_requested": True,
                    "cancel_call_id": call_id
                }
                print(
                    "[AGENT] LLM выбрала cancel_navigation. "
                    "Запрошена безопасная остановка."
                )

            session.setdefault("pending_aux_calls", {})[call_id] = {
                "tool": tool_name,
                "arguments": dispatch["arguments"]
            }
            self._call_sessions[call_id] = session["session_id"]

            return self._reply(
                speech,
                {
                    "call_id": call_id,
                    "tool": tool_name,
                    "arguments": dispatch["arguments"]
                }
            )

        return self._reply(
            "Слышу тебя. Текущую поездку пока продолжаю.",
            None
        )

    # ============================================================
    # СВОБОДНАЯ ИНИЦИАТИВА ВО ВРЕМЯ ПРОСТОЯ
    # ============================================================

    def _handle_idle_autonomy(
        self,
        session: Dict[str, Any],
        context: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """Запускает инициативу, когда V свободна; force используется для ручного теста."""
        now = time.monotonic()

        cooldown_left = float(session.get("autonomy_cooldown_until", 0.0)) - now
        if cooldown_left > 0.0 and not force:
            print(
                "[AGENT][AUTONOMY] Тик пропущен: действует пауза, осталось "
                f"{cooldown_left:.0f} сек."
            )
            return self._reply("", None, task_status="completed")

        busy_reasons = []
        if session.get("task_active"):
            busy_reasons.append("task_active")
        if session.get("waiting_for_player"):
            busy_reasons.append("waiting_for_player")
        if session.get("pending_call_id"):
            busy_reasons.append("pending_call_id")
        if session.get("pending_aux_calls"):
            busy_reasons.append("pending_aux_calls")

        if busy_reasons:
            print(
                "[AGENT][AUTONOMY] Тик пропущен: сессия занята ("
                + ", ".join(busy_reasons) + ")."
            )
            return self._reply("", None, task_status="completed")

        state = context.get("game_state") or {}
        # Наличие предмета в руках само по себе не мешает свободной
        # инициативе: V может осмотреть предмет, решить его положить,
        # бросить или продолжить заниматься чем-то ещё.
        # Блокируем только состояния, в которых тело/управление уже занято.
        busy_flags = (
            # Эти состояния не мешают V хотя бы заговорить или оценить
            # обстановку. Внутри AUTONOMY_SYSTEM_PROMPT отдельно запрещено
            # без причины выходить из машины или прекращать следование.
            "is_entering_car",
            "route_active",
            "route_preparing",
            "route_finishing",
            "drive_sequence_active"
        )
        active_flags = [
            flag for flag in busy_flags if bool(state.get(flag, False))
        ]
        if active_flags:
            print(
                "[AGENT][AUTONOMY] Тик пропущен: V занята по состоянию игры ("
                + ", ".join(active_flags) + ")."
            )
            return self._reply("", None, task_status="completed")

        # Не даём частым idle-запросам превратиться в постоянный поток LLM-вызовов.
        session["autonomy_cooldown_until"] = now + AUTONOMY_COOLDOWN_SECONDS
        session["history"] = []
        session["tool_calls_count"] = 0
        session["failed_tool_counts"] = {}
        session["last_tool_failure"] = {}
        session["pending_tool"] = None
        session["pending_interrupt"] = None
        session["pending_aux_calls"] = {}
        session["unresolved_tool_failures"] = {}

        self._task_manager.start_task(
            "Свободное время V: самостоятельно решить, говорить, действовать или молчать"
        )

        session["task_active"] = True
        session["waiting_for_player"] = False
        session["autonomy_mode"] = True
        session["autonomy_tool_calls"] = 0

        session["history"].append({
            "role": "user",
            "content": (
                "ТЫ ДАВНО СВОБОДНА. Игрок не ставил тебе задачу. Это просто\n"
                "момент, когда ты сама можешь решить: сказать что-нибудь,\n"
                "заняться чем-то по собственной инициативе или промолчать\n"
                "и остаться на месте. Не обязана проявлять активность.\n"
                "Вот текущая обстановка и твоя недавняя история инициатив:\n"
                + json.dumps({
                    "game_context": context,
                    "recent_autonomy_history": session.get("autonomy_history", [])[-5:]
                }, ensure_ascii=False, default=str)
            )
        })

        print("[AGENT][AUTONOMY] V получила возможность проявить инициативу.")
        return self._decide(session, state)

    # ============================================================
    # ВОССТАНОВЛЕНИЕ ПО ТЕРМИНАЛЬНОМУ СОСТОЯНИЮ НАВИГАЦИИ
    # ============================================================

    def _reconcile_terminal_navigation(
        self,
        session: Dict[str, Any],
        game_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Восстанавливает потерянный terminal tool_result по телеметрии C#.

        Помимо навигации здесь обрабатываются короткие переключатели
        follow_player/stop_following. Их результат иногда теряется на пути
        C# -> HTTP, хотя фактический флаг v_following уже изменился.
        """
        call_id = session.get("pending_call_id")
        pending_tool = session.get("pending_tool") or {}
        tool_name = str(pending_tool.get("tool") or "")
        if not call_id:
            return None

        state = game_state if isinstance(game_state, dict) else {}
        now = time.monotonic()
        synthetic_status = None
        synthetic_result = {"reconciled_from_game_state": True}
        reason = None

        if tool_name == "navigate_to":
            outcome = str(state.get("navigation_outcome") or "").strip().lower()
            terminal_outcomes = {"succeeded", "cancelled", "failed"}
            if outcome not in terminal_outcomes:
                return None

            still_active = any(bool(state.get(flag, False)) for flag in (
                "route_active", "route_preparing", "route_finishing"
            ))
            if still_active:
                return None

            synthetic_result["navigation_outcome"] = outcome
            if outcome == "cancelled":
                synthetic_status = "failed"
                reason = {
                    "code": "navigation_cancelled",
                    "message": "Маршрут уже был отменён; состояние подтверждено телеметрией."
                }
            elif outcome == "failed":
                synthetic_status = "failed"
                reason = {
                    "code": "navigation_failed",
                    "message": "Маршрут уже завершился с ошибкой; состояние подтверждено телеметрией."
                }
            else:
                synthetic_status = "succeeded"

        elif tool_name in ("follow_player", "stop_following"):
            # Не используем bool(value): строка "false" в Python truthy.
            if "v_following" not in state:
                return None
            raw_following = state.get("v_following")
            is_following = (
                raw_following is True
                or raw_following == 1
                or (isinstance(raw_following, str)
                    and raw_following.strip().lower() in ("true", "1", "yes"))
            )
            synthetic_result["v_following"] = is_following

            reached_state = (
                is_following if tool_name == "follow_player" else not is_following
            )
            if reached_state:
                synthetic_status = "succeeded"
            else:
                # В реестре для переключения следования задан тайм-аут 5 с.
                # Даём C# ещё небольшой запас на игровой кадр и доставку.
                try:
                    started_at = float(pending_tool.get("started_at", now))
                except (TypeError, ValueError):
                    started_at = now
                if now - started_at < 8.0:
                    return None
                synthetic_status = "timeout"
                reason = {
                    "code": "follow_state_timeout",
                    "message": (
                        "Флаг v_following не перешёл в ожидаемое состояние "
                        "за отведённое время; зависшее ожидание снято по телеметрии."
                    )
                }
        else:
            return None

        # Подхватываем ввод, пришедший уже после фактического завершения
        # действия, чтобы он не застрял за потерянным callback.
        if context.get("player_message") or context.get("game_event"):
            interrupt = session.get("pending_interrupt") or {
                "player_message": "",
                "game_events": [],
                "context": context,
                "cancel_requested": False
            }
            if context.get("player_message"):
                interrupt["player_message"] = context["player_message"]
            event = context.get("game_event")
            if event and event not in interrupt.setdefault("game_events", []):
                interrupt["game_events"].append(event)
            interrupt["context"] = context
            session["pending_interrupt"] = interrupt

        # Только у навигации могут быть параллельные вспомогательные вызовы.
        if tool_name == "navigate_to":
            pending_aux = session.setdefault("pending_aux_calls", {})
            for aux_id in list(pending_aux.keys()):
                pending_aux.pop(aux_id, None)
                self._call_sessions.pop(aux_id, None)

        print(
            "[AGENT][RECOVERY] Восстанавливаю потерянный результат",
            tool_name,
            "по телеметрии:",
            synthetic_result
        )

        return self.handle_tool_result(
            {
                "call_id": call_id,
                "tool": tool_name,
                "status": synthetic_status,
                "result": synthetic_result,
                "reason": reason
            },
            game_state=state
        )

    # ============================================================
    # НОВАЯ КОМАНДА ИЛИ ИГРОВОЕ СОБЫТИЕ
    # ============================================================

    def handle_input(
        self,
        player_message: str = "",
        game_event: str = "",
        game_state: Optional[Dict[str, Any]] = None,
        player_pos: Optional[list] = None,
        v_pos: Optional[list] = None,
        nearby: Optional[list] = None,
        time_of_day: str = "day",
        player_action: str = "idle"
    ) -> Dict[str, Any]:

        with self._lock:
            session = self._session
            state = game_state if isinstance(game_state, dict) else {}

            context = {
                "player_message": player_message or "",
                "game_event": game_event or "",
                "game_state": state,
                "player_pos": player_pos or [],
                "v_pos": v_pos or [],
                "nearby": nearby or [],
                "time_of_day": time_of_day,
                "player_action": player_action
            }

            # Если Unity уже сообщила, что маршрут завершён, но его
            # terminal callback потерялся, сначала освобождаем старый вызов.
            reconciled = self._reconcile_terminal_navigation(
                session, state, context
            )
            if reconciled is not None:
                return reconciled

            if (
                player_action in ("idle_autonomy", "idle_autonomy_force")
                and not player_message
                and not game_event
            ):
                return self._handle_idle_autonomy(
                    session,
                    context,
                    force=(player_action == "idle_autonomy_force")
                )

            # Пока выполняется инструмент, не допускаем параллельных
            # игровых действий. Во время navigate_to передаём новый ввод
            # основной LLM: она сама решает, нужен ли get_game_state,
            # cancel_navigation или обычный ответ.
            if session["pending_call_id"]:
                pending_tool = session.get("pending_tool") or {}

                if player_message or game_event:
                    pending_interrupt = session.get("pending_interrupt")

                    # Если отмена уже запрошена, только обновляем очередь.
                    # Повторно cancel_navigation не отправляем.
                    if (
                        pending_tool.get("tool") == "navigate_to"
                        and pending_interrupt is not None
                        and pending_interrupt.get("cancel_requested")
                    ):
                        if player_message:
                            pending_interrupt["player_message"] = player_message
                        if game_event and game_event not in pending_interrupt["game_events"]:
                            pending_interrupt["game_events"].append(game_event)
                        pending_interrupt["context"] = context
                        return self._reply(
                            "Уже безопасно останавливаюсь. Учту твоё сообщение после остановки.",
                            None
                        )

                    if pending_tool.get("tool") == "navigate_to":
                        session["history"].append({
                            "role": "user",
                            "content": (
                                "НОВЫЙ ВВОД ВО ВРЕМЯ АКТИВНОЙ НАВИГАЦИИ:\n"
                                + json.dumps(
                                    context,
                                    ensure_ascii=False,
                                    default=str
                                )
                                + "\nИсходный navigate_to пока остаётся активным. "
                                "Не перезаписывай его. Самостоятельно реши, "
                                "нужно ли только наблюдение, обычный ответ "
                                "или безопасная отмена маршрута."
                            )
                        })
                        return self._decide_during_navigation(
                            session,
                            state,
                            context
                        )

                    # Для остальных длительных инструментов пока сохраняем
                    # сообщение и обрабатываем его после терминального результата.
                    interrupt = session.get("pending_interrupt")
                    if interrupt is None:
                        interrupt = {
                            "player_message": "",
                            "game_events": [],
                            "context": context,
                            "cancel_requested": False
                        }
                    if player_message:
                        interrupt["player_message"] = player_message
                    if game_event and game_event not in interrupt["game_events"]:
                        interrupt["game_events"].append(game_event)
                    interrupt["context"] = context
                    session["pending_interrupt"] = interrupt
                    return self._reply(
                        "Слышу. Учту это, когда закончу текущее действие.",
                        None
                    )

                return self._reply(
                    "Я пока занята предыдущим делом. Не дёргай меня.",
                    None
                )

            # Если игрок ставит новую задачу после завершения предыдущей,
            # начинаем новую историю задачи.
            if (player_message or game_event) and (
                not session["task_active"] or session.get("autonomy_mode")
            ):
                session["history"] = []
                session["tool_calls_count"] = 0
                self._task_manager.start_task(player_message or game_event)
                session["autonomy_mode"] = False
                session["autonomy_tool_calls"] = 0

                # Новая задача — новая память попыток.
                session["failed_tool_counts"] = {}
                session["last_tool_failure"] = {}
                session["pending_tool"] = None
                session["pending_interrupt"] = None
                session["pending_aux_calls"] = {}
                session["unresolved_tool_failures"] = {}

            if player_message or game_event:
                session["task_active"] = True
                session["waiting_for_player"] = False

            session["history"].append({
                "role": "user",
                "content": (
                    "АКТУАЛЬНЫЙ КОНТЕКСТ ИГРЫ:\n"
                    + json.dumps(
                        context,
                        ensure_ascii=False,
                        default=str
                    )
                )
            })

            return self._decide(session, state)

    # ============================================================
    # РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ИНСТРУМЕНТА
    # ============================================================

    def handle_tool_result(
        self,
        tool_result: Dict[str, Any],
        game_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        with self._lock:
            if not isinstance(tool_result, dict):
                return {
                    "ignored": True,
                    "reason": "invalid_tool_result"
                }

            call_id = str(tool_result.get("call_id", ""))
            status = str(tool_result.get("status", ""))

            if not call_id:
                return {
                    "ignored": True,
                    "reason": "missing_call_id"
                }

            # accepted/running — действие ещё выполняется.
            # Новое решение пока не требуется.
            if status in ("accepted", "running"):
                return {
                    "waiting": True,
                    "call_id": call_id
                }

            session_id = self._call_sessions.get(call_id)

            if session_id != self._session["session_id"]:
                return {
                    "ignored": True,
                    "reason": "unknown_or_expired_call_id"
                }

            session = self._session

            # Вызовы, выполненные параллельно с navigate_to:
            # get_game_state — безопасное наблюдение; cancel_navigation
            # лишь подтверждает запрос, а фактический конец поездки
            # приходит отдельным результатом исходного navigate_to.
            pending_aux = session.get("pending_aux_calls", {})
            if call_id in pending_aux:
                aux_call = pending_aux.pop(call_id)
                self._call_sessions.pop(call_id, None)
                aux_tool_name = aux_call.get("tool", "")

                aux_summary = tool_result.get("result")
                try:
                    aux_summary = json.dumps(
                        aux_summary,
                        ensure_ascii=False,
                        default=str
                    )
                except Exception:
                    aux_summary = str(aux_summary)

                aux_ledger_status = status
                if (
                    aux_tool_name == "cancel_navigation"
                    and status in ("succeeded", "accepted", "running")
                ):
                    # Запрос принят, но машина ещё может тормозить.
                    aux_ledger_status = "cancel_requested"

                self._task_manager.record_action(
                    tool=aux_tool_name,
                    arguments=aux_call.get("arguments", {}),
                    status=aux_ledger_status,
                    result_summary=aux_summary,
                    reason=tool_result.get("reason")
                )

                if aux_tool_name == "cancel_navigation":
                    if status in ("succeeded", "accepted", "running"):
                        return {
                            "waiting": True,
                            "cancel_requested": True,
                            "call_id": call_id
                        }

                    # Запрос отмены не прошёл. Снимаем признак ожидания,
                    # но не теряем текущую навигацию.
                    session["pending_interrupt"] = None
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "cancel_navigation не удалось выполнить:\n"
                            + json.dumps(
                                tool_result,
                                ensure_ascii=False,
                                default=str
                            )
                            + "\nИсходная навигация всё ещё выполняется. "
                            "Оцени ситуацию и выбери следующий шаг."
                        )
                    })
                    return self._decide_during_navigation(
                        session,
                        game_state if isinstance(game_state, dict) else {},
                        None
                    )

                # Результат наблюдения не закрывает исходный navigate_to.
                state = (
                    game_state
                    if isinstance(game_state, dict)
                    else {}
                )
                if aux_tool_name == "get_game_state":
                    fresh_state = tool_result.get("result")
                    if isinstance(fresh_state, dict):
                        state = fresh_state
                        print(
                            "[AGENT][GAME STATE DURING NAVIGATION]\n"
                            + json.dumps(
                                fresh_state,
                                ensure_ascii=False,
                                indent=2,
                                default=str
                            )
                        )

                session["history"].append({
                    "role": "user",
                    "content": (
                        "РЕЗУЛЬТАТ НАБЛЮДЕНИЯ ВО ВРЕМЯ НАВИГАЦИИ:\n"
                        + json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            default=str
                        )
                        + "\nПроанализируй состояние. Исходная поездка "
                        "продолжается, если ты явно не выберешь "
                        "cancel_navigation."
                    )
                })
                return self._decide_during_navigation(
                    session,
                    state,
                    None
                )

            if session["pending_call_id"] != call_id:
                return {
                    "ignored": True,
                    "reason": "call_is_not_pending"
                }

            terminal_statuses = {
                "succeeded",
                "failed",
                "rejected",
                "timeout",
                "invalid"
            }

            if status not in terminal_statuses:
                return {
                    "ignored": True,
                    "reason": "unknown_result_status"
                }

                        # Запоминаем инструмент, завершившийся результатом.
            pending_tool = session.get("pending_tool") or {}

            completed_tool_name = str(
                pending_tool.get("tool")
                or tool_result.get("tool")
                or ""
            )

            pending_interrupt = session.get("pending_interrupt")
            reason_data = tool_result.get("reason")
            if isinstance(reason_data, dict):
                reason_code = str(
                    reason_data.get("code")
                    or reason_data.get("reason")
                    or ""
                )
            else:
                reason_code = str(reason_data or "")

            intentional_navigation_cancel = (
                completed_tool_name == "navigate_to"
                and pending_interrupt is not None
                and reason_code == "navigation_cancelled"
            )

            if completed_tool_name:
                result_summary = tool_result.get("result")
                try:
                    result_summary = json.dumps(
                        result_summary,
                        ensure_ascii=False,
                        default=str
                    )
                except Exception:
                    result_summary = str(result_summary)

                if len(result_summary) > 1200:
                    result_summary = result_summary[:1200] + "...[обрезано]"

                ledger_status = (
                    "cancelled_by_agent"
                    if intentional_navigation_cancel
                    else status
                )

                self._task_manager.record_action(
                    tool=completed_tool_name,
                    arguments=pending_tool.get("arguments", {}),
                    status=ledger_status,
                    result_summary=result_summary,
                    reason=tool_result.get("reason")
                )

            if completed_tool_name:
                if (
                    status in ("failed", "rejected", "timeout", "invalid")
                    and not intentional_navigation_cancel
                ):
                    failure_count = (
                        session["failed_tool_counts"].get(
                            completed_tool_name, 0
                        ) + 1
                    )

                    session["failed_tool_counts"][
                        completed_tool_name
                    ] = failure_count

                    session["last_tool_failure"][
                        completed_tool_name
                    ] = {
                        "tool": completed_tool_name,
                        "arguments": pending_tool.get("arguments", {}),
                        "status": status,
                        "failure_count": failure_count,
                        "reason": tool_result.get("reason")
                    }

                    session["unresolved_tool_failures"][
                        completed_tool_name
                    ] = {
                        "tool": completed_tool_name,
                        "arguments": pending_tool.get("arguments", {}),
                        "status": status,
                        "reason": tool_result.get("reason")
                    }

                    print(
                        "[AGENT] Неудача инструмента:",
                        completed_tool_name,
                        "попытка:",
                        failure_count,
                        "из",
                        MAX_FAILED_ATTEMPTS_PER_TOOL
                    )

                elif status == "succeeded":
                    # Успешное выполнение сбрасывает счётчик
                    # именно этого инструмента.
                    session["failed_tool_counts"].pop(
                        completed_tool_name, None
                    )
                    session["last_tool_failure"].pop(
                        completed_tool_name, None
                    )

                    session["unresolved_tool_failures"].pop(
                        completed_tool_name, None
                    )

            session["pending_tool"] = None

            # Закрываем текущий вызов.
            session["pending_call_id"] = None
            self._call_sessions.pop(call_id, None)

            state = (
                game_state
                if isinstance(game_state, dict)
                else {}
            )

            # get_game_state содержит самую свежую телеметрию.
            if tool_result.get("tool") == "get_game_state":
                fresh_state = tool_result.get("result")

                if isinstance(fresh_state, dict):
                    state = fresh_state

                    print(
                        "[AGENT][GAME STATE RAW]\n"
                        + json.dumps(
                            fresh_state,
                            ensure_ascii=False,
                            indent=2,
                            default=str
                        )
                    )

            # ------------------------------------------------
            # ОБРАБОТКА ОЖИДАЮЩЕГО ПРЕРЫВАНИЯ
            # ------------------------------------------------

            consume_interrupt = (
                pending_interrupt is not None
                and (
                    completed_tool_name != "navigate_to"
                    or status in terminal_statuses
                )
            )

            if consume_interrupt:
                interrupt = session.pop("pending_interrupt", None) or {}
                interrupt_context = interrupt.get("context") or {}
                new_player_message = str(
                    interrupt.get("player_message") or ""
                )
                game_events = interrupt.get("game_events") or []
                combined_game_event = "\n".join(game_events)

                if new_player_message:
                    # Явная новая команда начинает новую цель.
                    # Старую историю и старый план не переносим.
                    session["history"] = []
                    self._task_manager.start_task(new_player_message)
                    session["tool_calls_count"] = 0
                    session["failed_tool_counts"] = {}
                    session["last_tool_failure"] = {}
                    session["unresolved_tool_failures"] = {}
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0

                elif session.get("autonomy_mode") and game_events:
                    # Игровое событие важнее фоновой инициативы.
                    session["history"] = []
                    self._task_manager.start_task(
                        "Реакция на игровое событие: " + "; ".join(game_events)
                    )
                    session["tool_calls_count"] = 0
                    session["failed_tool_counts"] = {}
                    session["last_tool_failure"] = {}
                    session["unresolved_tool_failures"] = {}
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0

                session["task_active"] = True
                session["waiting_for_player"] = False

                resumed_context = {
                    "player_message": new_player_message,
                    "game_event": combined_game_event,
                    "game_state": state,
                    "player_pos": interrupt_context.get("player_pos", []),
                    "v_pos": interrupt_context.get("v_pos", []),
                    "nearby": interrupt_context.get("nearby", []),
                    "time_of_day": interrupt_context.get("time_of_day", "day"),
                    "player_action": interrupt_context.get("player_action", "idle"),
                    "interrupted_tool": completed_tool_name,
                    "interruption_reason": (
                        "navigation_cancelled"
                        if intentional_navigation_cancel
                        else (
                            "navigation_failed"
                            if completed_tool_name == "navigate_to" and status != "succeeded"
                            else "action_finished"
                        )
                    )
                }

                session["history"].append({
                    "role": "user",
                    "content": (
                        "ПРЕРЫВАНИЕ ОБРАБОТАНО. Текущее действие завершено.\n"
                        + json.dumps(
                            resumed_context,
                            ensure_ascii=False,
                            default=str
                        )
                        + (
                            "\nНовая команда игрока заменяет прежнюю задачу. "
                            "Определи новую конечную цель и следующий шаг."
                            if new_player_message
                            else "\nЭто игровое событие не отменяет прежнюю "
                            "конечную цель. Оцени состояние и продолжи задачу "
                            "с безопасного следующего действия."
                        )
                    )
                })

                print(
                    "[AGENT] Прерывание принято. Новая команда:",
                    bool(new_player_message),
                    "Событий:",
                    len(game_events)
                )

                return self._decide(session, state)

            # ------------------------------------------------
            # ДИНАМИЧЕСКИЙ ПРОГРЕСС ЗАДАЧИ
            # ------------------------------------------------

            task_progress = self._task_manager.get_progress(limit=12)

            # ------------------------------------------------
            # ПЕРЕДАЧА РЕЗУЛЬТАТА В ИСТОРИЮ АГЕНТА
            # ------------------------------------------------

            observation = {
                "tool_result": tool_result,
                "current_game_state": state,
                "task_progress": task_progress
            }

            session["history"].append({
                "role": "user",
                "content": (
                    "РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ИНСТРУМЕНТА:\n"
                    + json.dumps(
                        observation,
                        ensure_ascii=False,
                        default=str
                    )
                    + "\nПроанализируй результат, актуальное состояние игры "
                    "и накопленный прогресс относительно исходной просьбы. "
                    "Выбери только следующее действие."
                )
            })

            return self._decide(session, state)

    def _has_game_action_or_pending_clarification(self, session: Dict[str, Any]) -> bool:
        """Проверяет, есть ли в текущей истории незавершённая игровая работа.

        Игровая команда (в том числе отклонённая проверкой) или ask_player
        означают, что задача может требовать продолжения. Наблюдательные
        инструменты сами по себе не считаются игровым действием.
        История очищается при начале новой задачи/автономного эпизода.
        """
        for message in session.get("history", []):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue

            decision = self._parse_decision(message.get("content", ""))
            if not isinstance(decision, dict):
                continue

            if decision.get("type") == "ask_player":
                return True

            if decision.get("type") != "tool_call":
                continue

            tool_name = decision.get("tool")
            if isinstance(tool_name, str) and tool_name not in OBSERVATION_ONLY_TOOLS:
                return True

        return False

    # ============================================================
    # ОСНОВНОЙ ЦИКЛ ПРИНЯТИЯ РЕШЕНИЙ
    # ============================================================

    def _decide(
        self,
        session: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:

        for attempt in range(MAX_REPLANNING_ATTEMPTS):

            if session["tool_calls_count"] >= MAX_TOOL_CALLS_PER_TASK:
                session["task_active"] = False
                session["waiting_for_player"] = False
                self._task_manager.mark_status("abandoned", "Достигнут лимит вызовов инструментов.")
                session["pending_call_id"] = None
                session["pending_tool"] = None
                session["autonomy_mode"] = False
                session["autonomy_tool_calls"] = 0

                return self._reply(
                    "Я уже достаточно накомандовала. Давай пересмотрим задачу.",
                    None,
                    task_status="abandoned"
                )

            try:
                messages = [
                    {
                        "role": "system",
                        "content": self._system_prompt
                    },
                    {
                        "role": "system",
                        "content": self._build_task_context_message(session)
                    }
                ]

                if session.get("autonomy_mode"):
                    messages.append({
                        "role": "system",
                        "content": AUTONOMY_SYSTEM_PROMPT
                    })

                messages.extend(
                    session["history"][-MAX_HISTORY_MESSAGES:]
                )

                prompt_chars = sum(
                    len(str(message.get("content", "")))
                    for message in messages
                )

                llm_started = time.perf_counter()

                answer = self._llm_call(
                    messages=messages,
                    options={
                        "temperature": 0.7,
                        "num_predict": 500
                    },
                    timeout=90
                )

                llm_elapsed = time.perf_counter() - llm_started

                print(
                    f"[AGENT][PERF] LLM: {llm_elapsed:.2f}s | "
                    f"messages: {len(messages)} | "
                    f"prompt_chars: {prompt_chars}"
                )

            except Exception as e:
                print("[AGENT] Ошибка LLM:", e)

                if session.get("autonomy_mode"):
                    session["task_active"] = False
                    session["waiting_for_player"] = False
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0
                    self._task_manager.mark_status(
                        "abandoned", "Автономный вызов LLM завершился ошибкой."
                    )
                    print(
                        "[AGENT][AUTONOMY] Эпизод закрыт после ошибки LLM; "
                        "следующая проверка сможет запуститься."
                    )

                return self._reply(
                    "Чёрт, мой мозг сейчас не отвечает. Попробуем позже.",
                    None,
                    task_status="abandoned" if not session.get("task_active") else "in_progress"
                )

            if not answer or not answer.strip():
                if session.get("autonomy_mode"):
                    session["task_active"] = False
                    session["waiting_for_player"] = False
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0
                    self._task_manager.mark_status(
                        "abandoned", "Автономный вызов LLM вернул пустой ответ."
                    )
                    print(
                        "[AGENT][AUTONOMY] Эпизод закрыт после пустого ответа; "
                        "следующая проверка сможет запуститься."
                    )

                return self._reply(
                    "Я получила пустой ответ. Что-то тут не так.",
                    None,
                    task_status="abandoned" if not session.get("task_active") else "in_progress"
                )

            print("[AGENT] Решение LLM:", answer)

            decision = self._parse_decision(answer)

            if decision is None:
                session["history"].append({
                    "role": "assistant",
                    "content": answer
                })

                session["history"].append({
                    "role": "user",
                    "content": (
                        "Твой ответ не является корректным JSON "
                        "с допустимым типом решения. "
                        "Повтори ответ строго по протоколу V-Agent."
                    )
                })

                continue

            decision_type = decision.get("type")

            if session.get("autonomy_mode"):
                decision.pop("goal_spec", None)

            # Поиск предметов — подготовительное действие.
            # Сначала получаем реальные игровые ID, затем выбираем
            # ближайшее полезное действие относительно исходной просьбы.
            is_object_discovery = (
                decision_type == "tool_call"
                and decision.get("tool") == "get_nearby_objects"
                and self._task_manager.get_goal_spec() is None
            )

            # ------------------------------------------------
            # РЕГИСТРАЦИЯ КОНЕЧНОЙ ЦЕЛИ
            # ------------------------------------------------

            declared_goal = (
                None if is_object_discovery
                else decision.get("goal_spec")
            )

            if declared_goal is not None:
                if (
                    self._task_manager.is_goal_spec_decided()
                    and self._task_manager.get_goal_spec() is None
                ):
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "Для этой составной задачи уже выбран режим "
                            "отслеживания по исходной просьбе и истории действий. "
                            "Не добавляй goal_spec задним числом и не превращай "
                            "промежуточное действие в конечную цель. Продолжай "
                            "по исходной просьбе игрока."
                        )
                    })
                    continue

                normalized_goal = (
                    self._task_manager.normalize_goal_spec(
                        declared_goal
                    )
                )

                if normalized_goal is None:
                    session["history"].append({
                        "role": "assistant",
                        "content": json.dumps(
                            decision,
                            ensure_ascii=False
                        )
                    })

                    session["history"].append({
                        "role": "user",
                        "content": (
                            "ОШИБКА ОПИСАНИЯ ЦЕЛИ. "
                            "goal_spec содержит неподдерживаемый "
                            "тип или некорректные аргументы. "
                            "Используй только поддерживаемые типы целей."
                        )
                    })

                    continue

                # Первоначальную цель нельзя незаметно менять.
                if self._task_manager.get_goal_spec() is None:
                    goal_result = self._task_manager.set_goal_spec(normalized_goal)
                    if goal_result.get("status") != "ready":
                        session["history"].append({
                            "role": "user",
                            "content": "Не удалось сохранить конечную цель: "
                            + json.dumps(goal_result, ensure_ascii=False)
                        })
                        continue

                elif normalized_goal != self._task_manager.get_goal_spec():
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "Ты попыталась изменить первоначальную "
                            "конечную цель. Продолжай работать "
                            "с уже установленной целью:\n"
                            + json.dumps(
                                self._task_manager.get_goal_spec(),
                                ensure_ascii=False
                            )
                        )
                    })

                    continue

            # ------------------------------------------------
            # ПЛАНЫ НЕ ЯВЛЯЮТСЯ ОБЯЗАТЕЛЬНЫМИ
            # ------------------------------------------------

            # AgentLoop намеренно не регистрирует статический plan.
            # Следующий шаг выбирается заново после каждого результата.
            if decision.pop("plan", None) is not None:
                print(
                    "[AGENT] Статический plan проигнорирован: "
                    "используется динамический выбор следующего действия."
                )

            # ----------------------------------------------------
            # TOOL CALL
            # ----------------------------------------------------

            if decision_type == "tool_call":
                tool_name = decision.get("tool", "")
                arguments = decision.get("arguments", {})
                speech = decision.get("speech", "")

                # Проверяем формат вызова.
                if not isinstance(tool_name, str) or not tool_name:
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "Поле tool должно содержать название "
                            "инструмента в виде непустой строки. "
                            "Повтори решение по протоколу V-Agent."
                        )
                    })
                    continue

                if not isinstance(arguments, dict):
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "Поле arguments должно быть JSON-объектом."
                        )
                    })
                    continue

                if not isinstance(speech, str):
                    speech = ""

                if session.get("autonomy_mode"):
                    if session.get("autonomy_tool_calls", 0) >= MAX_AUTONOMY_TOOL_CALLS:
                        session["history"].append({
                            "role": "assistant",
                            "content": json.dumps(decision, ensure_ascii=False)
                        })
                        session["history"].append({
                            "role": "user",
                            "content": (
                                "Предохранительный лимит вызовов свободного эпизода\n"
                                "достигнут. Не вызывай больше инструменты сейчас.\n"
                                "Заверши эпизод ответом final и task_status=completed.\n"
                                "Реплика необязательна: можешь оставить speech пустым."
                            )
                        })
                        continue

                    # В свободном режиме нет отдельного allow-list:
                    # выбор делается из полного реестра, а ToolExecutor
                    # по-прежнему проверяет схему, предусловия и безопасность.

                    if tool_name == "get_nearby_objects":
                        radius = arguments.get("radius_m", 8)
                        limit = arguments.get("limit", 25)
                        try:
                            arguments["radius_m"] = min(8.0, max(1.0, float(radius)))
                        except (TypeError, ValueError):
                            arguments["radius_m"] = 8.0
                        try:
                            arguments["limit"] = min(25, max(1, int(limit)))
                        except (TypeError, ValueError):
                            arguments["limit"] = 25

                # Для составной задачи goal_spec может отсутствовать:
                # исходная цель хранится в TaskManager.
                if (
                    not self._task_manager.get_original_goal()
                    and self._task_manager.get_goal_spec() is None
                    and tool_name not in (
                        "list_destinations",
                        "get_nearby_objects",
                        "get_game_state"
                    )
                ):
                    session["history"].append({
                        "role": "user",
                        "content": (
                            "Не вижу исходной игровой задачи. "
                            "Сначала уточни у игрока, чего он хочет добиться, "
                            "либо используй только инструменты наблюдения."
                        )
                    })
                    continue

                prepared = self._tool_executor.prepare_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    game_state=game_state
                )

                # Инструмент не прошёл проверку.
                if prepared.get("status") != "ready":
                    feedback = {
                        "type": "tool_call_rejected",
                        "tool": tool_name,
                        "arguments": arguments,
                        "reason": prepared.get("reason")
                    }

                    session["history"].append({
                        "role": "assistant",
                        "content": json.dumps(
                            decision,
                            ensure_ascii=False
                        )
                    })

                    session["history"].append({
                        "role": "user",
                        "content": (
                            "ТВОЙ ВЫЗОВ ОТКЛОНЁН:\n"
                            + json.dumps(
                                feedback,
                                ensure_ascii=False
                            )
                            + "\nНе повторяй тот же ошибочный вызов. "
                            "Проанализируй причину и выбери другой шаг."
                        )
                    })

                    print(
                        "[AGENT] Вызов отклонён:",
                        tool_name,
                        prepared.get("reason")
                    )

                    continue

                dispatch = prepared["dispatch"]
                call_id = dispatch["call_id"]

                if (
                    self._task_manager.get_goal_spec() is None
                    and tool_name not in (
                        "list_destinations",
                        "get_nearby_objects",
                        "get_game_state"
                    )
                ):
                    self._task_manager.mark_goal_spec_decided(True)

                # Сохраняем связь вызова с текущей задачей.
                session["pending_call_id"] = call_id
                session["pending_tool"] = {
                    "tool": tool_name,
                    "arguments": dispatch["arguments"],
                    # Нужен для страховочного тайм-аута коротких
                    # переключателей состояния, если C# callback потерялся.
                    "started_at": time.monotonic()
                }

                session["tool_calls_count"] += 1
                if session.get("autonomy_mode"):
                    session["autonomy_tool_calls"] = (
                        session.get("autonomy_tool_calls", 0) + 1
                    )
                self._call_sessions[call_id] = session["session_id"]

                # Сохраняем решение модели в истории.
                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(
                        decision,
                        ensure_ascii=False
                    )
                })

                print(
                    "[AGENT] Инструмент выбран:",
                    tool_name,
                    "call_id:",
                    call_id
                )

                return self._reply(
                    speech,
                    {
                        "call_id": call_id,
                        "tool": tool_name,
                        "arguments": dispatch["arguments"]
                    }
                )

            # ----------------------------------------------------
            # FINAL
            # ----------------------------------------------------

            if decision_type == "final":
                speech = decision.get("speech", "")
                task_status = decision.get(
                    "task_status",
                    "in_progress"
                )

                if session.get("autonomy_mode") and task_status not in (
                    "completed", "abandoned"
                ):
                    task_status = "completed"

                # Обычный разговор или информационный запрос не должен
                # оставлять task_active=True. В составных игровых задачах
                # без goal_spec мы сохраняем сессию активной, если уже было
                # была игровая команда или запрос уточнения.
                # Чтение состояния мира через observation-инструменты
                # само по себе таким действием не считается.
                if (
                    task_status == "in_progress"
                    and not session.get("autonomy_mode")
                    and self._task_manager.get_goal_spec() is None
                    and not self._has_game_action_or_pending_clarification(session)
                ):
                    task_status = "completed"
                    decision["task_status"] = "completed"
                    print(
                        "[AGENT] Разговорный/информационный ход завершён; "
                        "освобождаю сессию."
                    )

                # Составная задача завершается по исходной просьбе
                # и накопленному прогрессу, а не по закрытию статического плана.

                # --------------------------------------------
                # GOAL VERIFICATION
                # --------------------------------------------

                # --------------------------------------------
                # СТАРАЯ ЗАЩИТА ДЛЯ ЗАДАЧ БЕЗ goal_spec
                # --------------------------------------------

                if (
                    task_status == "completed"
                    and self._task_manager.get_goal_spec() is None
                    and session["unresolved_tool_failures"]
                    and not session.get("autonomy_mode")
                ):
                    unresolved = session["unresolved_tool_failures"]

                    session["history"].append({
                        "role": "assistant",
                        "content": json.dumps(
                            decision,
                            ensure_ascii=False
                        )
                    })

                    session["history"].append({
                        "role": "user",
                        "content": (
                            "ПРОВЕРКА ЗАВЕРШЕНИЯ НЕ ПРОЙДЕНА.\n"
                            "Остались незакрытые ошибки:\n"
                            + json.dumps(
                                unresolved,
                                ensure_ascii=False,
                                default=str
                            )
                            + "\nНе объявляй задачу completed."
                        )
                    })

                    print(
                        "[AGENT] Завершение отклонено: "
                        "остались незакрытые ошибки."
                    )

                    continue

                # --------------------------------------------
                # GOAL VERIFICATION
                # --------------------------------------------

                if (
                    task_status == "completed"
                    and self._task_manager.get_goal_spec() is not None
                ):
                    verification = self._task_manager.verify_structured_goal(
                        game_state
                    )

                    if verification["status"] != "achieved":
                        session["history"].append({
                            "role": "assistant",
                            "content": json.dumps(
                                decision,
                                ensure_ascii=False
                            )
                        })

                        session["history"].append({
                            "role": "user",
                            "content": (
                                "GOAL VERIFICATION FAILED.\n"
                                "Ты попыталась объявить задачу выполненной, "
                                "но проверка фактического состояния игры "
                                "не подтвердила достижение цели.\n"
                                + json.dumps(
                                    verification,
                                    ensure_ascii=False,
                                    default=str
                                )
                                + "\nНе используй completed. "
                                "Получи актуальную информацию или "
                                "продолжи выполнение задачи."
                            )
                        })

                        print(
                            "[AGENT] Goal verification:",
                            verification["status"],
                            verification["reason"]
                        )

                        continue

                    print(
                        "[AGENT] ЦЕЛЬ ПОДТВЕРЖДЕНА:",
                        self._task_manager.get_goal_spec()
                    )

                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(
                        decision,
                        ensure_ascii=False
                    )
                })

                if task_status in ("completed", "abandoned"):
                    if session.get("autonomy_mode"):
                        autonomy_progress = self._task_manager.get_progress(limit=4)
                        autonomy_history = session.setdefault("autonomy_history", [])
                        autonomy_history.append({
                            "speech": speech or "",
                            "actions": autonomy_progress.get("recent_actions", [])
                        })
                        if len(autonomy_history) > 6:
                            del autonomy_history[:-6]

                    session["task_active"] = False
                    session["waiting_for_player"] = False
                    self._task_manager.mark_status(task_status, speech)
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0

                return self._reply(
                    speech,
                    None,
                    task_status=task_status
                )

            # ----------------------------------------------------
            # ASK PLAYER
            # ----------------------------------------------------

            if decision_type == "ask_player":
                speech = decision.get("speech", "")

                if session.get("autonomy_mode"):
                    session["task_active"] = False
                    session["waiting_for_player"] = False
                    session["autonomy_mode"] = False
                    session["autonomy_tool_calls"] = 0
                    self._task_manager.mark_status(
                        "completed", "Свободный цикл завершён без запроса к игроку."
                    )
                    return self._reply(
                        speech,
                        None,
                        task_status="completed"
                    )

                session["waiting_for_player"] = True

                session["history"].append({
                    "role": "assistant",
                    "content": json.dumps(
                        decision,
                        ensure_ascii=False
                    )
                })

                return self._reply(
                    speech,
                    None
                )

            # ----------------------------------------------------
            # UNKNOWN TYPE
            # ----------------------------------------------------

            session["history"].append({
                "role": "assistant",
                "content": json.dumps(
                    decision,
                    ensure_ascii=False
                )
            })

            session["history"].append({
                "role": "user",
                "content": (
                    "Недопустимый тип решения. "
                    "Используй tool_call, final или ask_player."
                )
            })

        # Все попытки получить корректное решение исчерпаны.
        # Завершаем задачу, чтобы следующая команда начала
        # новую чистую сессию задачи.
        session["task_active"] = False
        session["waiting_for_player"] = False
        session["autonomy_mode"] = False
        session["autonomy_tool_calls"] = 0
        self._task_manager.mark_status("abandoned", "Исчерпаны попытки получить корректное решение.")
        session["pending_call_id"] = None
        session["pending_tool"] = None

        return self._reply(
            "Я запуталась в последовательности действий. "
            "Нужно пересмотреть план.",
            None,
            task_status="abandoned"
        )

    # ============================================================
    # РАЗБОР JSON ОТ МОДЕЛИ
    # ============================================================

    @staticmethod
    def _parse_decision(answer: str) -> Optional[Dict[str, Any]]:
        if not isinstance(answer, str):
            return None

        content = answer.strip()

        # Удаляем Markdown-обрамление, если модель его добавила.
        if content.startswith("```"):
            lines = content.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        # Сначала пробуем разобрать весь ответ.
        try:
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Если перед JSON случайно появился лишний текст,
        # пробуем извлечь внешний объект.
        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end <= start:
            return None

        try:
            parsed = json.loads(content[start:end + 1])

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            return None

        return None

    # ============================================================
    # ЕДИНЫЙ ФОРМАТ ОТВЕТА
    # ============================================================

    @staticmethod
    def _reply(
        speech: str,
        tool_call: Optional[Dict[str, Any]],
        task_status: str = "in_progress"
    ) -> Dict[str, Any]:

        return {
            "speech": speech or "",
            "action": "none",
            "target": "",
            "tool_call": tool_call,
            "task_status": task_status
        }

    # ============================================================
    # ИНФОРМАЦИЯ О ТЕКУЩЕЙ СЕССИИ
    # ============================================================

    def get_session_info(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session

            return {
                "session_id": session["session_id"],
                "goal": self._task_manager.get_original_goal(),
                "plan": None,
                "task_progress": self._task_manager.get_progress(limit=12),
                "task_active": session["task_active"],
                "waiting_for_player": session["waiting_for_player"],
                "pending_call_id": session["pending_call_id"],
                "has_pending_interrupt": session.get("pending_interrupt") is not None,
                "pending_aux_calls": len(session.get("pending_aux_calls", {})),
                "autonomy_mode": session.get("autonomy_mode", False),
                "autonomy_tool_calls": session.get("autonomy_tool_calls", 0),
                "tool_calls_count": session["tool_calls_count"],
                "history_length": len(session["history"])
            }
