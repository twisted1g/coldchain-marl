# Cold-Chain MARL

Репродукция мультиагентного ядра из статьи Khanna et al., *Generative AI and
Blockchain-Integrated Multi-Agent Framework for Resilient and Sustainable Fruit
Cold-Chain Logistics* (Foods 2025, 14(17):3004). Синтетическая среда холодовой
цепи «ферма → ретейл», поверх которой обучается набор разнородных агентов по
схеме **централизованное обучение, децентрализованное исполнение (CTDE)**.
Приоритет — верность боксам-алгоритмам статьи (Alg 1–6), а не «красивые» игрушки.

Что уже собрано:

- **MARL-ядро** — 16 агентов пяти типов (см. таблицу), обучены совместно на
  TorchRL, проверены trained-vs-random на held-out сидах.
- **Per-crate мир** — груз стал first-class субъектом на каждой машине (singleton
  `shipment` устранён); routing/temperature/spoilage размножены до 3 инстансов
  под shared-политикой (Alg 4 param sharing).
- **GenAI-слой** — трансформер-форкастер спроса (offline-обучение, online через
  буфер истории) + LLM-медиатор слот-конфликтов доставки (Alg 6, любой
  OpenAI-совместимый эндпоинт).
- **Веб-дашборд** — React/TS, живой SSE-стрим инференса, панели мира / агентов /
  переговоров, переключатель conflict-solver greedy↔LLM.

## Статус агентов

| Агент | Алгоритм (бокс) | Действие | Инстансов | Метрика |
|-------|-----------------|----------|-----------|---------|
| temperature | DDPG (Alg 2) | сетпоинт рефрижератора | 3 (по машине) | temp_deviation |
| routing | DQN (Alg 1) | следующий узел | 3 (по машине) | route_cost |
| spoilage | frozen GraphSAGE + DDPG (Alg 3) | порог риска | 3 (по машине) | fn_rate |
| inventory | DDPG (Alg 4, shared) | объём заказа | 4 (по ретейлеру) | inventory_cost |
| delivery | MADDPG, shared critic (Alg 5) | слот расписания | 3 (по машине) | delivery_cost |

Последний совместный прогон (full 150 итераций): **temperature +97%**,
**spoilage +84%**, routing резко бьёт random. **delivery** и **inventory** после
per-crate ретрейна регрессируют (delivery — связка дедлайна со слотом,
inventory — bound-saturation актора) и донастраиваются — детали и план в
`docs/TODO.md` и `docs/status/`.

Механизмы, общие для всех агентов:

- **Динамические Парето-веса** `ω_j = c_j / Σ c_k` в каждой награде.
- **Общий intention-buffer + штраф координации ρ** (`core/intention.py`):
  declare → detect → ρ → clear каждый шаг; живой источник конфликтов — слоты
  доставки.
- **CTDE**: агент действует по локальным наблюдениям; общий critic
  `Q(joint_obs, joint_act)` — только у delivery (Alg 5).

Детерминизм: один сид → бит-в-бит идентичный прогон. Inventory-спрос и delivery
используют изолированные RNG-потоки — добавление агента не сдвигает миры
остальных.

## Установка

```bash
uv sync                    # рантайм-зависимости
uv sync --group notebook   # + Jupyter/pandas/matplotlib для ноутбуков
uv sync --group dev        # + ruff/mypy
```

Требуется Python ≥ 3.12. Стек: PyTorch + **TorchRL** (лоссы/replay/soft-update) +
рукописный MADDPG и CTDE-цикл, torch-geometric для spoilage-GNN. Без Ray/RLlib.

## Запуск и работа с моделями

Обучение (пишет кривые в `artifacts/reward_curve.csv`, модули в
`artifacts/modules/`, печатает trained-vs-random по каждому обучаемому):

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

# дешёвый smoke (few iters, ловит коллапсы, не финальное качество)
uv run python -m training.marl.train --iters 40 --tag smoke
```

Оценка и диагностика:

```bash
uv run python -m training.marl.evaluate          # trained-vs-random по блокам
uv run python -m training.marl.fingerprint       # obs-хэши: контроль детерминизма
uv run python -m training.marl.stress_eval       # деградация по банку сбоев
uv run python -m training.marl.negotiation_eval --greedy --episodes 2   # медиатор Alg 6, без LLM
```

LLM-медиатор переговоров (Alg 6) с реальной моделью — поднять локальный
OpenAI-совместимый сервер (например LM Studio на `:1234`), затем:

```bash
export LLM_MODEL="<id-модели>"
uv run python -m training.marl.negotiation_eval --episodes 2   # без --greedy → LLM-путь
```

Веб-дашборд (сборка фронта → запуск API + статики):

```bash
cd viz/web && npm install && npm run build && cd ../..
uv run python -m viz.serve        # http://127.0.0.1:8000 (web :8000, api :8001)
```

В live-режиме дашборд стримит свежий инференс по SSE; переключатель
conflict-solver (greedy↔LLM) на лету рестартит стрим под новым медиатором.
Для разработки фронта — `npm run dev` в `viz/web` рядом с `uv run python -m viz.api`.

Записать эпизод в JSONL и отрисовать gif без веб-стека:

```bash
uv run python -m viz.record --seed 90000 --episodes 1 > episode.jsonl
uv run python -m viz.dashboard episode.jsonl
```

## Структура

- `core/` — доменная логика без привязки к фреймворку: конфиг, состояние
  (`GlobalState` + per-vehicle `VehicleState` с `load`-грузом), динамика,
  Аррениусова порча, граф поставок, наблюдения, пространства действий, шум
  сбоев, `IntentionBuffer`, per-node микро-климат.
- `env/` — `ColdChainParallelEnv` (PettingZoo) и `ColdChainTrainingEnv`
  (методы наград на классе среды, включая динамические Парето-веса).
- `training/` — `marl/` (агенты, MADDPG, GNN, CTDE-цикл `loop.py`, реестр
  алгоритмов `config.py`, `train.py`, `evaluate.py`, `fingerprint.py`,
  `stress_eval.py`, `negotiation_eval.py`) и `forecaster/` (трансформер спроса).
- `llm/` — провайдер-независимый клиент, генератор банка сбоев, медиатор и
  протокол переговоров (Alg 6), реплей сценариев.
- `data/` — генерация и загрузка синтетического датасета, банк сценариев.
- `viz/` — инференс-раннер, запись эпизодов, matplotlib-дашборд, SSE-API и
  React/TS-фронтенд (`viz/web/`).
- `docs/` — `DONE.md` / `TODO.md`, дизайн-доки, журнал состояния `status/` и
  `development_guide.md` (как построить систему с нуля: архитектура фаза за фазой,
  дизайн-решения, карта на алгоритмы статьи).
- `notebooks/` — `dataset_report`, `training_report`, `agent_behavior`.

## Отклонения от статьи (задокументированы)

- **routing**: в статье табличный Q-learning (Alg 1), в реализации — **DQN**
  (табличная дискретизация наблюдения непрактична).
- **Форма динамических весов**: буквальное `−Σ ω_j·raw_j` вырождается в симуляции
  (эмиссии доминируют); реализовано как `−Σ ω_j·c_j` со статическими
  приоритетами, свёрнутыми в `c_j = α_j·raw_j`.
- **Не-из-статьи shaping** (несущий, вне Парето-члена): routing delivery-бонус
  (награда Alg 1 — чистый штраф), temperature deviation-член + step-penalty.
- **Intention-buffer**: в статье координирует инстансы одного типа; здесь живой
  конфликт — только слоты доставки, остальные декларируют, но по построению не
  конфликтуют.