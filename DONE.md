### Среда холодовой цепи
Построить ядро в `core/`: динамика температуры, порча фруктов, граф маршрутов, шумы, конфиг.

### Пакет среды и адаптер
Собрать `env/` с `training_env.py`, reward shaping внутри среды; адаптер под PettingZoo.

### Temperature-агент (DDPG)
Обучить контроль температуры рефрижераторов. Чекпоинт в `artifacts/modules/temperature/`.

### Routing-агент (DQN)
Обучить выбор маршрутов. Чекпоинт в `artifacts/modules/routing/`.

### Spoilage-агент (GNN + DDPG)
Обучить предсказание порчи по графу сети, отдельный pretrain энкодера. Чекпоинты в `artifacts/modules/spoilage*/`.

### Inventory-агент (DDPG)
Обучить управление запасами. Чекпоинт в `artifacts/modules/inventory/`.

### Delivery-агенты (MADDPG, ×3)
Обучить координацию трёх машин доставки. Чекпоинты в `artifacts/modules/delivery_*/`.

### Динамические Парето-веса ω
Добавить балансировку конфликтующих целей (скорость / энергия / порча) в общий reward.

### Intention buffer (ρ)
Добавить обмен намерениями между агентами — каждый учитывает планы остальных.

### Полный цикл обучения CTDE
Прогнать 150 итераций на TorchRL, все агенты сошлись. Кривые в `artifacts/reward_curve.csv`.

### Оценка против baseline
Сравнить с random baseline: temperature +98, spoilage +81, inventory +76, routing +60, delivery +43.

### Ноутбуки
Собрать `dataset_report`, `training_report`, `agent_behavior`.

### Рефакторинг
Вынести MADDPG в отдельный модуль, удалить мёртвое состояние, среду собрать в `env/`.