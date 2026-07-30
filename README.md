# Cold-Chain MARL

Воспроизведение мультиагентного ядра из статьи Khanna et al., *Generative AI and
Blockchain-Integrated Multi-Agent Framework for Resilient and Sustainable Fruit
Cold-Chain Logistics* (Foods 2025, 14(17):3004). Синтетическая среда холодовой
цепи «ферма → ретейл», в которой обучается набор разнородных агентов по схеме
**централизованное обучение, децентрализованное исполнение (CTDE)**. Главная цель —
точно повторить алгоритмы статьи (Alg 1–6).

Что уже сделано:

- **MARL-ядро** — 16 агентов пяти типов (см. таблицу), обучены совместно на
  TorchRL и проверены сравнением trained-vs-random на отложенных сидах.
- **Модель на уровне отдельных ящиков** — груз задаётся для каждой машины
  индивидуально (общий `shipment` убран); routing, temperature и spoilage
  развёрнуты в 3 инстанса под общей политикой (Alg 4, param sharing).
- **GenAI-слой** — трансформер прогнозирует спрос (обучается офлайн, работает
  онлайн через буфер истории) и LLM разрешает конфликты слотов доставки (Alg 6,
  любой OpenAI-совместимый эндпоинт).
- **Веб-дашборд** — React/TS, живой SSE-стрим инференса, панели мира, агентов и
  переговоров, переключатель разрешения конфликтов greedy↔LLM.

## Статус агентов

| Агент | Алгоритм (бокс) | Действие | Инстансов | Метрика |
|-------|-----------------|----------|-----------|---------|
| temperature | DDPG (Alg 2) | сетпоинт рефрижератора | 3 (по машине) | temp_deviation |
| routing | DQN (Alg 1) | следующий узел | 3 (по машине) | route_cost |
| spoilage | frozen GraphSAGE + DDPG (Alg 3) | порог риска | 3 (по машине) | fn_rate |
| inventory | DDPG (Alg 4, shared) | объём заказа | 4 (по ретейлеру) | inventory_cost |
| delivery | MADDPG, shared critic (Alg 5) | слот расписания | 3 (по машине) | delivery_cost |

Последний совместный прогон (150 итераций): **temperature +97%**,
**spoilage +84%**, routing заметно обходит random. **delivery** и **inventory**
после перехода на модель отдельных ящиков просели (delivery — из-за связки
дедлайна со слотом, inventory — из-за насыщения актора на границе диапазона) и
сейчас донастраиваются. Подробности и план — в `docs/TODO.md` и `docs/status/`.

Общие для всех агентов механизмы:

- **Динамические Парето-веса** `ω_j = c_j / Σ c_k` в каждой награде.
- **Общий intention-buffer и штраф координации ρ** (`core/intention.py`):
  каждый шаг declare → detect → ρ → clear. Реальный источник конфликтов — слоты
  доставки.
- **CTDE**: агент действует по локальным наблюдениям; общий critic
  `Q(joint_obs, joint_act)` есть только у delivery (Alg 5).


## Установка

```bash
uv sync                    # рантайм-зависимости
uv sync --group notebook   # + Jupyter/pandas/matplotlib для ноутбуков
uv sync --group dev        # + ruff/mypy
```

## Запуск и работа с моделями

```bash
# полный совместный прогон всех агентов
uv run python -m training.marl.train

# подмножество типов
uv run python -m training.marl.train --agents temperature routing

# группа доставки (три машины — одна MADDPG-группа)
uv run python -m training.marl.train --agents delivery_0 delivery_1 delivery_2

# дообучение из существующих чекпоинтов + сохранение под суффиксом
uv run python -m training.marl.train --agents inventory_0 --load --tag scn05

# со сценариями сбоев из LLM-банка (доля эпизодов со сбоем)
uv run python -m training.marl.train --scenario-bank data/scenarios/bank.json --scenario-prob 0.5

# с онлайн-прогнозом спроса от трансформера
uv run python -m training.marl.train --forecaster

# rolling-мир (длинный горизонт, реститок) — для delivery/inventory
uv run python -m training.marl.train --rolling

# быстрый smoke (мало итераций, ловит коллапсы; не финальное качество)
uv run python -m training.marl.train --iters 40 --tag smoke
```

Оценка и диагностика:

```bash
uv run python -m training.marl.evaluate          # trained-vs-random по блокам
uv run python -m training.marl.fingerprint       # obs-хэши: контроль детерминизма
uv run python -m training.marl.stress_eval       # деградация по банку сбоев
uv run python -m training.marl.negotiation_eval --greedy --episodes 2   # медиатор Alg 6, без LLM
```

Медиатор переговоров (Alg 6) с реальной LLM: поднять локальный OpenAI-совместимый
сервер (например, LM Studio на `:1234`), затем:

```bash
export LLM_MODEL="<id-модели>"
uv run python -m training.marl.negotiation_eval --episodes 2   # без --greedy → путь через LLM
```

Веб-дашборд (сборка фронта, затем запуск API и статики):

```bash
cd viz/web && npm install && npm run build && cd ../..
uv run python -m viz.serve        # http://127.0.0.1:8000 (web :8000, api :8001)
```

В live-режиме дашборд стримит свежий инференс по SSE; переключатель разрешения
конфликтов (greedy↔LLM) на лету перезапускает стрим с новым медиатором.
Записать эпизод в JSONL и отрисовать gif без веб-стека:

```bash
uv run python -m viz.record --seed 90000 --episodes 1 > episode.jsonl
uv run python -m viz.dashboard episode.jsonl
```

## Структура

- `core/` — доменная логика без привязки к фреймворку: конфиг, состояние
  (`GlobalState` и `VehicleState` с грузом `load`), динамика, порча по Аррениусу,
  граф поставок, наблюдения, пространства действий, шум сбоев, `IntentionBuffer`,
  микро-климат по узлам.
- `env/` — `ColdChainParallelEnv` (PettingZoo) и `ColdChainTrainingEnv` (методы
  наград на классе среды, включая динамические Парето-веса).
- `training/` — `marl/` (агенты, MADDPG, GNN, цикл CTDE `loop.py`, реестр
  алгоритмов `config.py`, `train.py`, `evaluate.py`, `fingerprint.py`,
  `stress_eval.py`, `negotiation_eval.py`) и `forecaster/` (трансформер спроса).
- `llm/` — провайдер-независимый клиент, генератор банка сбоев, медиатор и
  протокол переговоров (Alg 6), реплей сценариев.
- `data/` — генерация и загрузка синтетического датасета, банк сценариев.
- `viz/` — раннер инференса, запись эпизодов, matplotlib-дашборд, SSE-API и
  React/TS-фронтенд (`viz/web/`).
- `docs/` — `DONE.md`, `TODO.md`, дизайн-доки, журнал состояния `status/` и
  `development_guide.md` (как собрать систему с нуля: архитектура по фазам,
  проектные решения, соответствие алгоритмам статьи).
- `notebooks/` — `dataset_report`, `training_report`, `agent_behavior`.

## Отклонения от статьи (задокументированы)

- **routing**: в статье табличный Q-learning (Alg 1), здесь **DQN** —
  дискретизация наблюдения в таблицу непрактична.
- **Форма динамических весов**: буквальное `−Σ ω_j·raw_j` в симуляции вырождается
  (доминируют эмиссии), поэтому используется `−Σ ω_j·c_j` со статическими
  приоритетами внутри `c_j = α_j·raw_j`.
- **Shaping вне статьи** (несущий, за пределами Парето-члена): бонус доставки у
  routing (награда Alg 1 — чистый штраф), член отклонения температуры и
  step-penalty у temperature.
- **Intention-buffer**: в статье он координирует инстансы одного типа; здесь
  реально конфликтуют только слоты доставки, остальные объявляют намерения, но по
  построению не сталкиваются.
