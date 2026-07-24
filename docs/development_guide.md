# Гайд по разработке с нуля

Руководство по повторной сборке системы — от пустого репозитория до обученных
агентов, GenAI-слоя и веб-дашборда. Документ описывает, **что и в каком порядке
проектировать**, какие решения приняты и на каком основании, и как каждый
компонент соответствует боксам-алгоритмам статьи Khanna et al. (Foods 2025,
14(17):3004). Это не построчный пересказ кода, а инженерная карта, позволяющая
воспроизвести проект или собрать аналог осознанно.

Ведущий принцип — **fidelity-first**: сначала анализируется соответствующий бокс
статьи (Alg 1–6), затем реализуется механика под него; каждое отклонение
фиксируется с обоснованием (см. раздел «Отклонения» в конце и `README.md`).
Декоративные агенты исключаются — каждый агент решает реальную задачу в среде.

Оглавление:

0. [Предпосылки и раскладка](#0-предпосылки-и-раскладка)
1. [Фаза A — мир среды (`core/`)](#фаза-a--мир-среды-core)
2. [Фаза B — наблюдения и действия](#фаза-b--наблюдения-и-действия)
3. [Фаза C — награды](#фаза-c--награды)
4. [Фаза D — агенты и алгоритмы](#фаза-d--агенты-и-алгоритмы)
5. [Фаза E — CTDE-обучение](#фаза-e--ctde-обучение)
6. [Фаза F — GenAI-слой](#фаза-f--genai-слой)
7. [Фаза G — per-crate мир](#фаза-g--per-crate-мир-singleton-elimination)
8. [Фаза H — оценка и детерминизм](#фаза-h--оценка-и-детерминизм)
9. [Фаза I — визуализация](#фаза-i--визуализация)
10. [Порядок сборки и карта на алгоритмы](#порядок-сборки-и-карта-на-алгоритмы-статьи)

---

## 0. Предпосылки и раскладка

**Стек.** Python ≥ 3.12, `uv` для окружения. PyTorch + **TorchRL** (готовые
`DDPGLoss` / `DQNLoss` / `TensorDictReplayBuffer` / `SoftUpdate`), рукописный
MADDPG (в TorchRL нет turnkey-лосса) и рукописный CTDE-цикл. torch-geometric для
spoilage-GNN. PettingZoo как интерфейс мультиагентной среды. Без Ray/RLlib —
управляемость и детерминизм важнее фреймворкового автопилота.

**Раскладка пакетов** (её же и стройте по порядку):

```
core/       доменная логика без привязки к RL-фреймворку
env/        обёртки PettingZoo + reward shaping
training/   агенты, алгоритмы, CTDE-цикл, форкастер, оценка
llm/        GenAI: клиент, банк сбоев, переговоры (Alg 6)
data/       генерация/загрузка синтетического датасета
viz/        инференс-раннер, запись, matplotlib- и веб-дашборд
docs/       DONE/TODO, дизайн-доки, журнал состояния
```

Правило слоёв: `core` ничего не знает про RL и torch-агентов; `env` шейпит
награды поверх `core`; `training` знает про `env`; `viz`/`llm` — сверху. Обратных
зависимостей нет.

---

## Фаза A — мир среды (`core/`)

Мир строится до введения RL. Цель — детерминированная по сиду симуляция холодовой
цепи, в которую впоследствии интегрируются агенты. Все константы вынесены в
`core/config.py` (единый источник размеров, диапазонов и калибровочных весов).

### A1. Граф поставок

Направленный граф «ферма → ретейл»: `N_FARMS=2`, `N_HUBS=3`, `N_DCS=2`,
`N_RETAILERS=4` (`N_NODES=11`). Рёбра несут `distance_km` (20–300),
`base_transit_time` (1–3 тика), `emissions = distance × EDGE_BASE_EMISSIONS_PER_KM`.
Плюс синтетическое **wait-ребро** (петля: transit 1, эмиссии 0) — чтобы у routing
всегда было легальное действие ожидания. Реализация — `core/world/graph.py`
(`source_nodes` = фермы, `sink_nodes` = ретейлеры). Граф строится из сида
детерминированно.

### A2. Календарь, погода, ambient-температура

Один тик = один день. Погода моделируется устойчивой марковской цепью
(`WEATHER_TRANSITION`, диагональ доминирует → период солнечной погоды или шторм
сохраняется несколько дней, а не пересэмплируется каждый тик). Ambient-температура
каждый день = база погоды +
сезонный размах (`AMBIENT_SEASONAL_AMPLITUDE_C=6`) + дневной шум, клампится в
`(-5, 42)°C`. Погодный шум изолирован своим RNG-оффсетом (`WEATHER_RNG_OFFSET`),
чтобы добавление агента не сдвигало погоду. Это «changing/extreme weather» из
Sec 3 статьи.

### A3. Генератор спроса

Мультипликативная модель (`core/world/demand.py`): `base × сезон(±30%) ×
день-недели(×1.2 в выходные) × погода × событие × логнормальный шум(σ=0.1)`.
События-пики: `p=0.02` за день, длительность 1–3 дня, множитель ×1.5–2.5.
Календарные ковариаты (день года, день недели) — детерминированы; шумы и события
— свой RNG-поток. Эта же модель порождает и офлайн-датасет для трансформера
(Фаза F), и онлайн-спрос в среде — одна модель, две точки использования.

### A4. Per-node микро-климат (Design F)

Каждый узел графа держит **свой** storage-климат (температура/влажность),
mean-reverting к kind-специфичному сетпоинту (`farm 20°C`, `hub 11°C`, `dc 3°C`,
`retail 5°C`) и подтянутый к ambient с kind-специфичной силой (холодная камера DC
почти не чувствует улицу: pull 0.01; ферма на открытом воздухе — 0.15), с
жёстким клампом в физичный диапазон. Груз в узле наследует климат этого узла.
Это внешняя среда, с которой борется охлаждение, и вход в spoilage-GNN
(«риск по узлам» рассуждает над реальными per-node условиями, Table 1 статьи), а
не плоская копия ambient. Свой RNG-оффсет (`NODE_CLIMATE_RNG_OFFSET=777`).

### A5. Состояние

`core/state.py`:

- `GlobalState` — тик, календарь, погода, климат узлов, граф, RNG-потоки;
  per-инстанс списки inventory (`inventory_levels`, `unmet_demand`,
  `inventory_order`, `demand_today`, `inventory_arrival_emissions`), окна истории
  спроса (`histories`, 28 дней), очереди заказов/грузов, `fault_signals`.
- `VehicleState` — одна машина: `current_node`, `route`, накопленные
  `route_transit`/`route_emissions`, слот-поля доставки (`delay`,
  `sla_violated`, `conflict`, `emissions`) и **`load`** — груз-`Consignment`
  этой машины (основа per-crate модели, см. Фазу G).
- `Consignment` — груз: `fruit_type`, `sensor_temperature_c`, `energy`,
  `spoilage_risk`, `spoilage_prediction`, `instance`, `vehicle`, `arrival_tick`.

`init_state(seed, ...)` собирает всё детерминированно. Важный инвариант: порядок
`rng.choice`-дро фиксирует стрим погоды/спроса/сбоев — менять его нельзя без
инвалидации обученных весов (см. Фазу H про fingerprint).

### A6. Динамика

`core/dynamics.py` — шаг мира: применить действия агентов per-vehicle
(`_apply_routing_actions` / `_apply_temperature_actions` /
`_apply_spoilage_actions`), продвинуть машины по маршруту (`_advance_vehicles`),
доставить прибывший груз (`_deliver_vehicle`), запустить реститок-поток
(`_dispatch_orders`, заказ едет реальной машиной, Фаза G/W), продвинуть климат и
порчу грузов (`_advance_loads`), собрать infos.

**Порча — Аррениус.** Скорость порчи растёт с температурой по
`k = A·exp(−Ea/(R·T))` (`core/world/spoilage.py`, `R_GAS`, `KELVIN_OFFSET`),
плюс влажностный и задержечный факторы. `spoilage_risk` — накопленная вероятность
порчи; `risk_to_label` бинаризует по `RISK_LABEL_THRESHOLD=0.5` (метка для Alg 3).
Движущийся груз дополнительно теряет `TRANSIT_SPOILAGE_RATE=0.05 × risk` за тик
(порча в пути); груз в очереди или ожидающий окна находится в холодильнике и не
портится.

### A7. Шум сбоев (disruptions)

`DISRUPTION_PROB_PER_TICK=0.05`: `BLOCKED_NODE` (0.30), `INCREASED_TRANSIT` (0.50),
`RISK_FLAG` (0.20). Это встроенный стохастический стресс мира (в отличие от
LLM-банка сценариев из Фазы F — тот подменяет/усиливает сбои сгенерированными
сценариями). Свой RNG-поток.

---

## Фаза B — наблюдения и действия

`core/interfaces/observations.py` + `core/config.py` (поля obs). Каждый тип агента
видит **только локальное** наблюдение (CTDE-децентрализация на исполнении).

| Агент | obs-поля (dim) | действие |
|-------|----------------|----------|
| routing | traffic/weather/perishability/route_status/risk/loc/target + 5×(edge features) | дискретный следующий узел (`N_NEXT_NODES=5`) |
| temperature | current T/H, desired T, energy, fault_signals (5) | непрерывный сетпоинт `[-30, 30]°C` |
| spoilage | по узлам: T/H/delay/fruit_type (`N_NODES×4`) | непрерывный порог риска `[0,1]` |
| inventory | level, on_order, demand_forecast, shelf_life (4) | непрерывный объём заказа `[0,1]` |
| delivery | vehicle_id, availability, window, risk, breakdown, delays (6) | дискретный слот (`N_DELIVERY_WINDOWS=4`) |

Тонкости, влияющие на обучаемость (задокументированы в `config.py`):

- **inventory obs = 4, не 6.** Убраны shipment-global `zone_energy_usage` и
  `peer_stock` — это был шум, нерелевантный per-instance решению о заказе;
  политика цеплялась за него и училась вредной state-зависимости, проигрывая
  random. Держим ровно state из Alg 4 box `[level, forecast, shelf_life, carbon]`
  + `on_order` (pipeline-позиция).
- **idle-slot masking.** Машина без груза → routing/temperature/spoilage-инстанс
  отдаёт obs-нули, no-op действие, нулевую награду (Фаза G).

Пространства действий — `core/spaces.py`. Дискретные (routing/delivery) —
`Discrete`; непрерывные — `Box` с клампом из констант `*_ACTION_LOW/HIGH`.

---

## Фаза C — награды

Награды живут **на классе среды** (`env/training_env.py`), не в отдельном
`rewards.py` — reward shaping неотделим от механики среды. Каждый learner имеет
метод `_<agent>_reward(i)`, зарегистрированный в `self._reward_methods`; frozen-агенты
получают нулевую награду из core.

### C1. Динамические Парето-веса — общий каркас

Ядро всех наград (`_dynamic_pareto`): `w_j = a_j·ctx_j / Σ_k a_k·ctx_k`, стоимость
`= Σ_j w_j·c_j`. Боксы 2–5 берут `ctx` = сами компоненты стоимости; Alg 1
считает веса из отдельного контекст-вектора (`ctx=[dT, traffic, sla]` у routing).
Это «Context-Aware Weights» из статьи. Отклонение формы: буквальное `−Σ w_j·raw_j`
вырождается (эмиссии доминируют в сим-единицах) → реализовано `−Σ w_j·c_j` со
статическими приоритетами, свёрнутыми в `c_j = α_j·raw_j`.

### C2. Формулы по агентам

- **temperature (Alg 2):** cost = Pareto над `[deviation×2, energy×1,
  spoilage_delta×25]` минус `STEP_PENALTY`. deviation = |T_сенсора − оптимум
  фрукта|. deviation-член и step-penalty — не-из-статьи shaping (Alg 2 перечисляет
  только energy+spoilage), несущий: без него нет позитивного сигнала на удержание
  температуры.
- **routing (Alg 1):** cost = Pareto над `[Δtransit×1, risk×10, Δemissions×0.1]`
  с `ctx=[deviation, traffic, sla_priority]`; при прибытии — крупный
  `DELIVERY_BONUS=100`. Бонус — не-из-статьи (награда Alg 1 чисто штрафная), но
  единственный позитивный сигнал, предотвращающий бесконечное ожидание.
- **spoilage (Alg 3):** cost = Pareto над `[pred_error×3, false_negative×5,
  inspection×0.2]`. pred_error = (предсказание − метка)²; FN = предсказан низкий
  риск при фактически высоком. Штрафуется именно FN — пропущенная порча дороже
  ложной тревоги.
- **inventory (Alg 4):** cost = Pareto над `[Lspoil×5, holding×1, E×1]` **плюс
  аддитивно** `STOCKOUT_WEIGHT×shortage`. Ключевые калибровки (все с причиной в
  коде): E начисляется на **прибытие** груза, не на действие-заказ (иначе
  мгновенный штраф при отложенной выгоде → коллапс order=0);
  `EMISSIONS_SCALE=0.01` (сырой delivery-carbon ~50 доминировал бы Pareto);
  shortage вынесен из Pareto (внутри размывался в convex-вес и не мог перевесить
  holding); `STOCKOUT_WEIGHT=0.5` (при 1.0 резкий асимметричный обрыв штрафа
  смещал политику к избыточному заказу).
- **delivery (Alg 5):** cost = Pareto над `[delay×1, sla×5, E×0.05]` +
  `CONFLICT_PENALTY=5` при коллизии слота. `EMISSIONS_SCALE=0.01` — та же логика,
  что у inventory (эмиссии в основном route-fixed, не рычаг агента; ими владеет
  routing/Alg 1). Отдельная метрика `slot_cost` (delay+SLA+conflict) отражает
  подконтрольную агенту величину; на ней и проводится сравнение с random.

Приём вынесения не-масштабируемого или несущего члена за пределы Pareto (shortage,
conflict, bonus) применяется неоднократно: self-normalising Pareto подавляет
любой член, который должен доминировать, поэтому такие члены задаются аддитивно,
где градиент сохраняется.

---

## Фаза D — агенты и алгоритмы

`training/marl/agents.py` (+ `maddpg.py`, `gnn.py`). Единый протокол `Agent`:
`act(obs, explore) → action`, `observe(transition)`, `update()`, `save/load`.
Реализации:

- **`DDPGAgent`** — непрерывные действия (temperature, inventory, spoilage-head).
  TorchRL `DDPGLoss` + `TensorDictReplayBuffer` + `SoftUpdate` (tau=0.005),
  гауссов exploration-шум (σ=0.1). Конфиг — `DDPG_CFG`.
- **`DQNAgent`** — дискретный routing. `DQNLoss`, ε-greedy (1.0→0.05 за 2000
  шагов). Отклонение: в статье routing — табличный Q-learning (Alg 1); табличная
  дискретизация непрактична → DQN.
- **`SpoilageAgent`** — frozen GraphSAGE-энкодер (`gnn.py`, отдельный pretrain,
  Фаза F/претрейн) + DDPG-голова над эмбеддингом графа. Alg 3: «GNN over network,
  predict risk».
- **`MADDPGDelivery`** (`maddpg.py`) — shared centralized critic
  `Q(joint_obs, joint_act)` над V машинами (Alg 5), per-vehicle акторы с
  Gumbel-softmax дискретизацией (tau 1.0→0.3). Рукописный — turnkey-MADDPG в
  TorchRL нет.
- **`FrozenAgent` / `RandomAgent`** — backdrop и baseline.

### D1. Param sharing (Alg 4) — `SharedHandle`

routing/temperature/spoilage/inventory симметричны по инстансам → **одна**
политика на тип и V обёрток (`SharedHandle(shared_net, i)`). Все обёртки
обращаются к одной и той же сети; обёртка index-0 владеет градиентным шагом и
чекпоинтом, replay общий. Это `S={s(1)..s(n)}` из Alg 4 line 5. Delivery — отдельная
MADDPG-группа (`DeliveryHandle`).

### D2. Intention buffer + ρ (Sec 4.1)

`core/intention.py`: каждый шаг `declare → detect conflicts → штраф ρ → clear`.
Основной источник конфликтов — слоты доставки (две машины в один слот). Остальные
типы декларируют намерения в буфер, но по построению не конфликтуют (в текущем
scope). У inventory явный ρ убран: очередь заказов уже откладывает contended-заказ
на следующую свободную машину (Alg 4 line 10 «reassign»), поэтому явный ρ
сводился к постоянному штрафу.

---

## Фаза E — CTDE-обучение

### E1. Реестр агентов

`training/config.py` — единый registry: `ALGO` (тип→алгоритм), `METRIC`
(тип→метрика для кривой), `COMPARE_METRIC` (для честного trained-vs-random там,
где per-step метрика несравнима: random routing редко достигает цели → сравнение
на `return`; delivery → на `slot_cost`), `SHARED_GROUPS` (какие типы делят
политику).
`build_agents(env, learners)` собирает: delivery → MADDPG-группа, симметричные
типы → shared-политики, остальные learners → свой алгоритм, прочие → frozen.

### E2. Цикл

`training/marl/loop.py` — компактный CTDE-цикл: `reset → {act всеми → step →
observe+update каждым} до done`. Централизация обучения — внутри агентов (общий
critic у delivery, общий replay у shared-групп); децентрализация исполнения — в
`act` по локальным obs.

`training/marl/train.py` — оркестрация: `NUM_ITERATIONS=150`,
`EPISODES_PER_ITERATION=40`, периодическая eval-кривая (`EVAL_EVERY=5`,
`EVAL_EPISODES=10`) в `artifacts/reward_curve.csv`, финальный trained-vs-random на
`COMPARE_EPISODES=30`. Флаги: `--agents` (подмножество), `--load`/`--tag`
(дообучение/варианты), `--forecaster`, `--scenario-bank`/`--scenario-prob`,
`--rolling`, `--iters` (smoke), `--seed`.

Изолированные RNG-потоки (спрос, delivery, климат, погода, сбои имеют свои
оффсеты) → один сид даёт бит-в-бит прогон, а добавление агента не возмущает миры
остальных.

---

## Фаза F — GenAI-слой

Строится поверх обученного MARL-ядра. Три независимых компонента.

### F1. Трансформер-форкастер спроса

- **Датасет:** `data/generate_demand.py` — 50 рядов × 2000 дней прямо из
  генератора A3 → `data/demand/` (npz + manifest с сидами/sha256/снапшотом
  конфига). Сплит по времени 70/15/15.
- **Модель:** `training/forecaster/model.py` — `TransformerEncoder` 2 слоя,
  d_model 64, ~69k параметров; окно 28 дней × 10 фич + токен target-дня.
  `training/forecaster/train.py` учит офлайн → `artifacts/modules/forecaster/`.
- **Онлайн:** `GlobalState.histories` держит окно 28 дней (backfill при reset);
  с `config["forecaster"]` замороженный трансформер каждый шаг заполняет
  `demand_forecast[i]`, целясь в **день прибытия заказа** (lead time вперёд), не в
  завтра (`_apply_forecast`). Оценка — `training/forecaster/evaluate.py`.

### F2. LLM-клиент и банк сценариев сбоев

- **Клиент** `llm/client.py` — провайдер-независимый, любой OpenAI-совместимый
  эндпоинт, structured output + retry, `LLMConfig.from_env` (модель через
  `LLM_MODEL`, base-url через env/флаг). Обрабатывает `reasoning_content`
  reasoning-моделей.
- **Банк** `llm/generate_bank.py` — генерирует сценарии сбоев (7 категорий × 6
  эффектов `{kind, magnitude, target_role}`, диапазоны проверяет валидатор) →
  `data/scenarios/bank.json`. Реплей в эпизодах — `env/scenarios.py`
  (`scenario_bank`/`scenario_prob`/`scenario_id`). Доза сбоев критична: `prob=1.0`
  (буквальный рецепт статьи) вызывает catastrophic forgetting у temperature;
  `prob=0.5` работает как регуляризатор. Обучение с банком →
  `--scenario-bank ... --scenario-prob 0.5`.

### F3. LLM-переговоры (Alg 6)

`llm/negotiation.py` + `llm/mediation.py`: медиатор слот-конфликтов доставки.
Протокол по букве бокса — последовательные counter-offers (уступка только если
переезд дешевле конфликтного штрафа), `S_t = L(H)` (LLM-суммаризация истории с
win-win предложением), agreement = сами оферты, utility против порогов τ =
статус-кво с конфликтом; обобщён на n-стороннюю коллизию. `SlotMediator.resolve`
перехватывает конфликтующие claims перед `env.step` и подменяет их
согласованным назначением; провал переговоров оставляет конфликт (платит штраф
среды). Оценка — `training/marl/negotiation_eval.py` (`--greedy` для правила без
сервера; без флага — реальный LLM). `build_mediator` даёт `off`/`greedy`/`llm` —
общий вход и для eval, и для дашборда.

---

## Фаза G — per-crate мир (singleton elimination)

Изначально мир содержал **singleton** `state.shipment` — один глобальный груз, к
которому были привязаны routing/temperature/spoilage (по одному агенту на тип).
Это упрощение расходилось с многомашинной реальностью и препятствовало Alg 4 param
sharing для этих типов. Редизайн (см. `docs/singleton_elimination.md`):

- Груз переехал в **`vehicle.load`** — `Consignment` на каждой `VehicleState`.
- routing/temperature/spoilage стали **first-class per-crate** агентами: 3 инстанса
  на тип (V=`N_VEHICLES=3`), симметричные как inventory → одна shared-политика на
  V обёрток (Alg 4 param sharing). Агентов стало **10→16**.
- Мир — всегда rolling-restock (груз респавнится, а не завершает эпизод).
- Простаивающая машина (нет груза) маскируется: obs-нули, no-op, нулевая награда.

Порядок редизайна был поэтапный: **S1** ядро+obs (per-vehicle obs-векторы,
masking), **S2** env+rewards (per-instance награды, shared-группы), **S3** viz
(env сам роутит/предсказывает per-vehicle в `step`, overlay-драйверы viz сняты).
Инвариант детерминизма: оба `rng.choice`-дро, что потреблял singleton (source +
target узлы), **сохранены** — мировой стрим бит-в-бит идентичен сидам, на которых
обучались политики. При сборке с нуля per-crate модель следует закладывать сразу,
не вводя singleton-груз; данный раздел приведён как иллюстрация недостатков
singleton-подхода.

---

## Фаза H — оценка и детерминизм

- **`training/marl/evaluate.py`** — trained-vs-random по блокам (симметричные
  инстансы группируются под одним именем: routing/temperature/spoilage/inventory/
  delivery).
- **`training/marl/fingerprint.py`** — obs-хэши прогона: инструмент контроля
  детерминизма. После любого рефакторинга, не меняющего поведение, хэши блоков
  обязаны совпасть бит-в-бит. Это страховка, что чистка/редизайн не
  инвалидировали обученные веса.
- **`training/marl/stress_eval.py`** — деградация метрик блоков на чистых эпизодах
  vs каждая категория банка сбоев (согласована с физикой: temperature реагирует
  на термальные категории, spoilage — на power outage, и т.д.).
- **`training/marl/negotiation_eval.py`** — метрики Alg 6 (conflict, slot_cost,
  SLA) baseline vs mediated + статистика медиатора.

**Правило детерминизма** (несущее для всего проекта): один сид → бит-в-бит
идентичный прогон. Достигается изолированными RNG-потоками на каждый источник
шума (погода/спрос/климат/сбои/inventory) и стабильным порядком `rng`-дро. Любое
изменение, трогающее порядок дро, инвалидирует чекпоинты — проверяйте
fingerprint'ом.

---

## Фаза I — визуализация

- **Инференс-раннер** `viz/inference.py` — `run_inference(...)` гоняет обученные
  политики в rolling-мире; env сам роутит/предсказывает per-vehicle (после S3
  overlay-драйверов нет), `build_mediator(off/greedy/llm)` выбирает решатель
  слот-конфликтов. `_tick_record` сериализует тик (машины, грузы, склад,
  переговоры) в JSON.
- **Запись** `viz/record.py` — эпизод в JSONL; **`viz/dashboard.py`** рисует gif
  через matplotlib без веб-стека.
- **Веб** `viz/web/` (React/TS, Vite): панели мира (граф + позиции машин +
  per-crate груз), агентов (per-vehicle действия/награды), переговоров (раунды,
  S_t, статистика медиатора). Данные — live-SSE через `viz/api.py`
  (`run_inference` стримит тики) + статика через `viz/server.py`; оба поднимает
  `viz/serve.py`. Переключатель conflict-solver (`MediatorSwitch`, greedy↔LLM)
  в live-режиме на лету рестартит стрим под новым медиатором.

Проектируйте события мира как отдельный поток, а рендер — как подписчика: это
позволяет добавлять слои (напр. будущий ledger) без переделки.

---

## Порядок сборки и карта на алгоритмы статьи

**Порядок разработки** (каждая фаза опирается на предыдущую):

```
A. core: граф → погода/спрос → климат → состояние → динамика/порча → сбои
B. наблюдения + пространства действий
C. награды (Pareto-каркас + per-agent формулы) на классе среды
D. агенты (DDPG/DQN/GNN+DDPG/MADDPG) + param sharing + intention buffer
E. CTDE-цикл + реестр + оценка + fingerprint  ← MARL-ядро функционально
F. GenAI: форкастер → LLM-клиент/банк → переговоры (Alg 6)
G. per-crate мир (или сразу закладываем в A/B/C/D)
H. оценка/стресс/детерминизм
I. визуализация
```

**Команды и порядок запуска артефактов** — см. `README.md` (раздел «Запуск и
работа с моделями»): `uv sync` → `data.generate_demand` →
`training.forecaster.train` → (опц. `llm.generate_bank`) →
`training.marl.pretrain_spoilage` → `training.marl.train` → `evaluate`/`fingerprint`
→ дашборд.

**Карта на боксы статьи:**

| Бокс | Что | Где |
|------|-----|-----|
| Alg 1 | routing (path/next-hop) | `_routing_reward`, `DQNAgent` |
| Alg 2 | temperature (reefer setpoint) | `_temperature_reward`, `DDPGAgent` |
| Alg 3 | spoilage (GNN risk prediction) | `_spoilage_reward`, `SpoilageAgent`+`gnn.py` |
| Alg 4 | inventory + param sharing `S={s(i)}` | `_inventory_reward`, `SharedHandle` |
| Alg 5 | delivery (SLA-расписание, shared critic) | `_delivery_reward`, `MADDPGDelivery` |
| Alg 6 | LLM-переговоры слот-конфликтов | `llm/negotiation.py`, `llm/mediation.py` |
| Sec 3 | GenAI: спрос-форкаст, changing weather | `training/forecaster/`, `WEATHER_TRANSITION` |
| Sec 4.1 | intention buffer + ρ, dynamic Pareto ω | `core/intention.py`, `_dynamic_pareto` |

**Отклонения от статьи** (несущие, задокументированы здесь и в `README.md`):
routing DQN вместо табличного Q-learning; форма динамических весов `−Σ w·c`;
не-из-статьи shaping (routing-бонус, temperature deviation/step-penalty);
intention-buffer конфликтует только на delivery в этом scope. Blockchain-слой
(Alg 8–18) из плана снят.

Актуальное состояние работ и приоритеты — `docs/DONE.md`, `docs/TODO.md`,
`docs/status/`.
