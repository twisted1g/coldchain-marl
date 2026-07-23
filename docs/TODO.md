# Phase V — визуализация

### V4. Стресс-витрина
Таблица `stress_eval` как heatmap (блок × категория сценария, деградация vs clean), drill-down в эпизод конкретного сценария через дашборд V1.

# Phase W — world fidelity (после визуализации)

### W5. Alg 6 на re-stocks и routes
Section 4.1: переговоры про "delivery times, re-stocks, and change of routes" — сейчас только delivery slots. После W2 появляются реальные конфликты re-stock (два склада, одна машина) и route (смена маршрута ради SLA). Расширить `SlotParty`/`negotiate` на эти типы конфликтов.

### W6. Stress + compare после W1–W5
Перепрогнать `stress_eval` и trained-vs-random на новом мире, обновить baseline-профиль. Fingerprint-инструмент (`training/marl/fingerprint.py`) — для контроля детерминизма при каждом инкременте.

# Отложенные неточности vs статья

### Обратный путь машины (Phase W)
W2 делает обратный путь мгновенным: round-trip при 3 машинах и transit ~4.5 даёт ~1.1 поставки за эпизод против спроса ~2.25 — структурный stockout. Вернуть round-trip только вместе с рекалибровкой (больше машин / короче маршруты / крупнее заказы).

### Delivery: развязать дедлайн от слота (после singleton-elimination)
Ретрейн per-crate дал delivery −58%: `sla_deadline=(slot+1)/N·max_steps` И `departure=max(tick, slot_start)` оба растут со слотом → slot сокращается из `sla_violated ⟺ T_transit > window_width`, единственный рычаг — conflict, а он при 3veh/4win/cap1 почти не биндит. Фикс: `deadline = departure + expected_lead_time·margin`, затем ретрейн delivery.

### Inventory: bound-saturation политики (после W4)
Коллапс order=0 снят, но обученная политика схлопнулась в order=1.000 (std 0): tanh-актор насыщается у верхнего бортика action-range. Награда корректна (внутренний оптимум ~0.4, симуляция бьёт random). Кандидаты: регуляризация pre-activation актора, узкий action-range, gradient clipping.

### Repro заявленных чисел (финальная интеграция)
−50/−35/−25/−30 (и несоответствие 40-vs-50 в тексте статьи), −60% SLA. Проверять только на полном мире: MARL + GenAI + W вместе.
