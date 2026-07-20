# Phase V — визуализация

### V1. Дашборд мира
Живая картина эпизода поверх `world_state`: граф маршрутов с позициями vehicles, склад (stock / on_order / прибытия из W1–W2), температура vs пороги, spoilage risk по цепочке, delivery slots и их дедлайны. Тик-за-тиком проигрывание эпизода + скраббинг. Отправная точка — существующие notebooks, целевая форма — standalone (Plotly Dash / Streamlit, решить при дизайне).

### V2. Панель агентов
Что решает каждый агент и почему: действия по тикам, награды по компонентам shaping, динамика Pareto ω, intention-buffer/ρ. Кривые обучения из `artifacts/reward_curve*.csv` рядом с trained-vs-random маржами.

### V3. Визуализация переговоров (Alg 6)
Раунды offer/counter-offer, S_t summary, пороги τ_i и утилиты сторон, итоговый assignment; после W5 — те же виды для re-stock/route конфликтов. Статистика медиатора (cache hits, rounds, failures) из `SlotMediator.stats`.

### V4. Стресс-витрина
Таблица `stress_eval` как heatmap (блок × категория сценария, деградация vs clean), drill-down в эпизод конкретного сценария через V1.

### V5. Доработки после blockchain-фазы
Слой поверх V1/V3: лента транзакций ledger'а, подписанные контракты как исходы переговоров (agreement → contract), DIDs участников на графе мира, статус smart-contract вызовов (Alg 8–18). Делать только после blockchain, но V1–V3 проектировать так, чтобы слой добавлялся без переделки (события мира — отдельный поток, рендер — подписчик).

# Phase W — world fidelity (после визуализации)

### W4. Multi-instance — DONE (см. DONE.md)
Осталось: финальный trained-vs-random на full 150 — покрыто W6.

### W5. Alg 6 на re-stocks и routes
Section 4.1: переговоры про "delivery times, re-stocks, and change of routes" — сейчас только delivery slots. После W2 появляются реальные конфликты re-stock (два склада, одна машина) и route (смена маршрута ради SLA). Расширить `SlotParty`/`negotiate` на эти типы конфликтов.

### W6. Stress + compare после W1–W5
Перепрогнать `stress_eval` и trained-vs-random на новом мире, обновить baseline-профиль. Fingerprint-инструмент (`training/marl/fingerprint.py`) — для контроля детерминизма при каждом инкременте.

# Отложенные неточности vs статья

### Обратный путь машины (Phase W)
W2 делает обратный путь мгновенным: round-trip при 3 машинах и transit ~4.5 даёт ~1.1 поставки за эпизод против спроса ~2.25 — структурный stockout. Вернуть round-trip только вместе с рекалибровкой (больше машин / короче маршруты / крупнее заказы).

### Contract signing после agreement (blockchain-фаза)
Section 4.1: после соглашения контракт подписывается через smart contracts. Ждёт Alg 7/9/14 (Solidity/JSON-RPC стек из статьи). `Agreement` уже несёт всё нужное (assignment + summary).

### Ledger + DIDs (blockchain-фаза)
Alg 8–18: журнал транзакций, decentralized IDs участников. Отдельная фаза после GenAI/W.

### Repro заявленных чисел (финальная интеграция)
−50/−35/−25/−30 (и несоответствие 40-vs-50 в тексте статьи), −60% SLA. Проверять только на полном мире: MARL + GenAI + W + blockchain вместе.
