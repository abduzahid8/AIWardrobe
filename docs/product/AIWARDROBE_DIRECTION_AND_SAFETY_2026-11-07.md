# AIWardrobe — Product Direction, Architecture Reality & Safe Roadmap

**Дата фиксации:** 2026-11-07  
**Статус:** стратегическое решение + проверенный read-only аудит; НЕ инструкция автоматически менять production.  
**Контекст:** история обсуждения пользователя (загруженная переписка, около 2905 строк), GitHub main на коммите 5784fc5cff24381cdae982e937d841fca611c7cd и живой Supabase (агрегированные запросы только на чтение).  
**Что НЕ делалось:** изменения исходного кода, SQL-миграции, записи в Supabase, деплой Edge Functions, изменение конфигурации production, тестирование реальных пользовательских потоков на устройстве.

> Главный принцип продукта: **AI that helps you buy fewer, better clothes for less money**. AIWardrobe должен превратиться из генератора образов в контекстный гардеробный и покупательский decision engine. При этом нельзя ломать существующие правила генерации, ротации и безопасность ради новых возможностей.

## 1. Как менялось направление: что считали главным и почему пересмотрели

### Этап A — Digital Closet / AI Personal Stylist
Изначальная ценность: оцифровать одежду, уменьшить усталость от ежедневного выбора, предложить комплекты по погоде, формальности и случаю. Основа — React Native/Expo, Supabase, Zustand и rule-based suggestion engine. Дополнения: AI chat, wear logs, календарь, travel, virtual try-on, shopping.

### Этап B — Personal Style OS / Style Autopilot
Обсуждалось переходить от ручного «подбери образ» к проактивным предложениям на завтра/неделю с календарём и погодой. Style Memory/Style Graph были предложены как долгосрочный персональный актив, а не очередной промпт к LLM.

### Этап C — Wardrobe Decision Engine / Purchase Intelligence
Рассматривались BuyCheck, Fit DNA, референсные вещи с идеальной посадкой, evidence по сохранённым и возвращённым покупкам. Ключевое различие: «подойдёт ли мне размер?» и «нужна ли вещь моему гардеробу?» — разные измерения. Не сводить всё в выдуманный единый процент. Fit Feedback должен быть **ненавязчивым и полностью отключаемым**; body scanning не обязан быть условием запуска.

### Этап D — Shopping Intelligence становится текущим приоритетом
Самый свежий запрос пользователя: повысить полноту каталога, реальную разнородность выдачи, лёгкость и экономическую пользу покупок. Выделены:
- **Catalog Intelligence:** реальные product identities, физические вещи, product families, варианты, источники, retrieval и разнообразие.
- **Shopping Intelligence:** Find Better, Same Look Less Money, Maximum Wardrobe Gain, Price Intelligence, сравнение полной цены предложения.
- **Fit Intelligence:** BuyCheck и Fit DNA как следующий полезный слой поверх исправленного каталога.

**Текущее решение:** сначала измерить и исправить входные данные/поиск/разнообразие, затем построить экономические сценарии покупок; параллельно подготовить лёгкий, opt-in фундамент для Fit DNA. Autopilot и сложный 3D остаются перспективными, но не должны отвлекать от качества выбора товаров.

## 2. Что подтверждено в актуальном GitHub

**Репозиторий:** https://github.com/abduzahid8/AIWardrobe  
**Аудированный main:** 5784fc5 (последний commit, который вернул GitHub при проверке: 2026-11-02).

- docs/ARCHITECTURE.md, ADR-001: основным backend surface является Supabase Postgres/Auth/Edge Functions; backend/api/ — legacy/deprecated, создавать новые Express-маршруты нельзя. При этом документация ещё содержит исторические фрагменты об Express: текущие решения всегда проверять по коду/действующему деплою.
- src/services/outfitGenerationService.ts вызывает Supabase Edge generate-outfits; существуют дополнительные локальные пути подбора кандидатов, но их нельзя считать эквивалентом production Edge.
- backend/supabase/functions/generate-outfits/index.ts содержит многоступенчатые фильтры, quality gates, rotation/diversity, budget constraints, shop supplement, four-slot validation и механизмы работы с физическими продуктами. Это **не простой random shuffle**.
- src/services/stylistCatalogInventory.ts делает по нескольким фильтрам запросы к shop_catalog с **limit(100)** на фильтр, сортируя по id. Такой клиентский slice может не отражать весь каталог.
- src/services/shoppingService.ts: часть локальных fallback-путей делает запросы с **limit(30)** на слот перед ранжированием. Это ограничение не описывает все пути production генератора, но показывает риск плохой локальной выборки.
- src/utils/shopCatalogOrder.ts уже различает схожесть названий, семейства обуви, дедупликацию по изображению и косвенно определяет цвет из названия/описания. Отдельного канонического family/variant слоя в проверенной схеме пока нет.
- src/lib/fit/fitEngine.ts — действующий чистый, офлайн-совместимый движок сравнения body/garment measurements, fit intent, stretch, зон посадки и confidence; это сильная база для будущей персонализации, но не эквивалент полноценной обученной Fit DNA.
- src/services/suggestionEngine.ts — отдельный rule-based движок. В исходнике: формальность 30%, новизна 25%, цветовая гармония 25%, погода 20%; preference weight 0 с комментарием «folded into formality tier selection». Не переносить этот вывод автоматически на каждый серверный scoring path.
- Есть специализированные тесты: shoeRotation, outfitRotation, outfitDiversityInvariant, outfitStructure, fourSlotRequirement, outfitQuality, stylistAgentAdversarial и др. Новая логика обязана проходить релевантную матрицу.
- docs/product/product.md описывает прежнюю отдельную инициативу **App Stabilization & App Store readiness**; это не полноценный актуальный Shopping Strategy. Известные там проблемы нельзя автоматически считать неустранёнными на 2026-11-07.

## 3. Что подтверждено в production Supabase (только агрегаты, без персональных данных)

На дату аудита:
- shop_catalog: **7 912** записей, **7 020** активных, **40** различных брендов, **49** различных источников.
- Из активных: **5 267 tops**, **1 638 bottoms**, **115 shoes**.
- Активная обувь — только **6 брендов**; поле garment_type у этих 115 записей — обобщённое значение shoes (детальную разновидность приходится выводить из других полей).
- Из 7 020 активных записей **7 008** не имеют заполненного external_id. Это НЕ значит, что товары невозможно идентифицировать по URL/ID/названию; значит, что надёжный source-specific ID не предоставлен отдельным полем почти для всех записей.
- У активных товаров в проверенных агрегатах нет пропусков в price, image_url и source_url. Это проверка заполненности, **не** проверка того, что цена/наличие/ссылка актуальны.
- Основные sources включают zara-uz-white-bg и множество других; неверно утверждать, что текущая база состоит почти только из Zara.
- Схема имеет плоскую shop_catalog с одним listing-подобным рядом. Отдельных production-таблиц product families, variants, merchant offers, price history и FitEvidence в проверенном списке public tables нет.
- Есть production таблицы profiles, clothing_items, saved_outfits, wear_logs, body_profiles, subscriptions, usage_counters и другие. Агрегированные estimated rows в list_tables не являются точной аналитикой пользовательской активности.
- Активные Edge: generate-outfits (v192), ai-process (v53), ai-chat, try-on и др. Факт ACTIVE не доказывает безошибочность конечного пользовательского пути или равенство деплоя последнему GitHub-коду.
- Security Advisor выдаёт предупреждения о некоторых SECURITY DEFINER RPC и о leaked-password protection; Performance Advisor — об оптимизации некоторых RLS-политик. Не делать автоматических security-миграций без оценки фактических прав/контрактов/регрессий. В общей продуктовой документации не публикуются секреты и внутренние exploit-инструкции.

**Вывод:** «не хватает одежды» — слишком широкая формулировка. Реальная комбинация: малая представленность обуви, слабая нормализация identifiers, неодинаковые client/Edge retrieval paths, отсутствие явной variant/offer модели, ограничения candidate budget и уже существующие сложные правила diversity/rotation.

## 4. Центральная архитектурная поправка: пять различных уровней идентичности

1. **Listing:** запись конкретного merchant/source и его URL/прайса.
2. **Physical Product:** реально различимая вещь для показа, подбора и outfit rotation. Коричневые и чёрные лоферы иногда должны быть ДВУМЯ разными физическими объектами.
3. **Product Family:** одна модель/линейка, под которой UI может группировать цветовые варианты для удобства просмотра.
4. **Variant:** конкретный colour/size/SKU, связанный с physical product/family; доступность и цена могут быть вариантно-зависимыми.
5. **Visual Family:** визуально/стилистически близкие, но не обязательно одинаковые вещи для выдачи без похожих карточек.

**Нельзя** просто объявить все цвета одной product-family «одним продуктом» в механизме outfit rotation: так можно потерять физически разные альтернативы, сломать LRU и снизить разнообразие. **Нельзя** игнорировать цвета при дедупликации источника, если от этого деактивируется корректный товар. Группировка карточек, подавление визуальной похожести и физическая идентичность — разные операции.

В первой итерации делаем исключительно **read-only mapping/диагностику**, не меняя таблицу и серверные identity keys.

## 5. Целевая продуктовая модель

**One-sentence promise:** «AIWardrobe знает, что у меня уже есть, помогает находить подходящие вещи и показывает, как купить лучше и дешевле».

Четыре понятных пользовательских входа:
- **Complete my wardrobe:** какая одна вещь реально повысит полезность гардероба?
- **Find something:** найти несколько качественных, разных вариантов по задаче, стилю, бюджету и доступности.
- **Improve an outfit:** собрать более подходящий комплект; по возможности использовать уже купленное.
- **Save money:** Find Better, Same Look Less Money, best total price и позже Wait for Sale.

Каталог — инфраструктура решения; интерфейс не должен превращаться в бесконечную ленту однотипных вещей. Для сравнения показывать **Best match / Best value / Budget** с объяснениями, а не ложную точность непроверенного «95% match».

## 6. Product roadmap: последовательность с учётом рисков

### Phase 0 — Baseline & contract audit (сейчас, read-only / local-only)
**Цель:** получить воспроизводимую оценку проблем без риска для production.

- Картировать все пути: Home picker, stylist, shop-only, shoppingService fallback, generate-outfits Edge, catalog sync, sources. Для каждого указать SQL filters, сортировку, лимиты, доступный pool, dedup, cache и особенности server/local.
- Вывести в локальных тестах: active/eligible/unique-physical/unique-visual/unique-family counts, coverage по категориям/стилям/ценам и причину отсечения каждого кандидата.
- Зафиксировать golden outputs/тесты для 3-card/4-slot, weather, occasion, budget, rotation, anchored item, shop-only, empty/low inventory.
- На тестовом срезе проверить, какие физически разные цвета сейчас подавляются и какие одинаковые карточки повторяются.
- Определить показатели latency, catalog completeness, availability freshness, 0 regression. Не объявлять улучшения до замеров.

**Выход:** архитектурная карта, диагностический отчёт, test fixtures и diff проектируемых изменений; production untouched.

### Phase 1 — Client-only improvements, без нарушения текущего Edge
**Цель:** поработать с качеством выдачи безопасно в локальной ветке.

- Разделить display-grouping от actual selection identity (если возможно без contract changes).
- Улучшать только явно client-only выборку/кеширование/пагинацию там, где это не противоречит действующему Edge контракту.
- Ввести детерминированную diversity диагностику: shoe family, silhouette, цвет, сочетания, повтор физического продукта между карточками.
- Все изменения под feature flag или выключенным по умолчанию режимом; existing tests зелёные, новый режим сравнивается с baseline.
- **Не** менять RLS, RPC, schema, Edge, sync, auth, subscription, paid generation.

**Выход:** локальный PR + тесты и замеры до/после. Не мержить автоматом, не деплоить.

### Phase 2 — Canonical Catalog & Retrieval (только после отдельного approval)
Спроектировать merchant adapters и canonical сущности: products, families, variants, offers, price-history; mapping на текущие shop_catalog IDs; независимую «визуальную семью»; verified source attribution и обновление актуальности. Retrieval: жесткие фильтры → широкая выборка (search) → personal ranking → family-aware selection → diversity → компактный LLM-context. LLM **не** должен получать тысячи строк без фильтра.

Это **потенциальный**, а не утверждённый план миграции. Никаких новых SQL/Edge/API без согласования контракта, рисков, RLS, обратной совместимости, rollback и budget. Не мигрировать/не удалять имеющиеся данные до подтверждения crosswalk IDs.

### Phase 3 — Money-saving shopping UX
**Find Better:** похожий, но выгоднее/лучше по реальной цене, доступности и посадке.  
**Same Look, Less Money:** альтернативы по каждому item, сохранение силуэта/назначения и суммарный budget.  
**Maximum Wardrobe Gain:** какие покупки открывают наиболее **полезные, отличающиеся** outfits; не считать голые декартовы комбинации.  
**True Price / Wait for Sale:** стоимость с доставкой/налогами/скидками, история цены и размерная доступность. Никаких «сэкономишь RM X», если цена не подтверждена.

Эти сценарии требуют подтверждённых каталогов и offer feeds; сначала прототипировать без обещаний real-time stock и автоматических покупок.

### Phase 4 — Fit Memory / BuyCheck (параллельно проектировать, включать постепенно)
- Опциональный, ненавязчивый «How does it fit?» на карточке вещи; **Not now**, закрыть и глобально **OFF** без санкций и повторяющихся попапов.
- Reference garments из фактического «perfect fit», а не обязательная анкета или body scan.
- FitEvidence: garment_id/brand/category/cut/size, source, зональная оценка, keep/return, confidence, версия профиля. Сырые факты отделены от выводного Brand Fit Memory.
- Лёгкие серверные агрегаты в перспективе, no photo duplication, low local cache, event-driven updates.
- Fit Check опирается на реальные размеры и confidence; неизвестный размерный chart → честное **insufficient data**.
- BuyCheck показывает **раздельно** Fit, Wardrobe Value, Duplicate Risk, Price/Offers, а затем BUY / CONSIDER / SKIP с объяснением. Пользователь принимает решение сам.
- Share → AIWardrobe из внешних магазинов — после доказанного quality, отдельно для iOS/Android native builds.

### Phase 5 — Future compounding advantage
Wardrobe Graph (владение, ношение, предпочтения, посадка, покупки и возвраты); Calendar/Weather Autopilot и wardrobe gaps; затем ассистент агентной покупки **только с явным пользовательским подтверждением действия**. Сложный 3D/body reconstruction и social feed не являются текущим P0.

## 7. Защищённые инварианты: нельзя ломать

- Существующий production generate-outfits — источник истины для server deck; **не заменять** его client-side shuffle.
- Сохранить правила формальности, weather, required slots/four-slot, hot-weather exclusions, color/palette, budget/currency, occasions, shop-only, anchored garment, full user-wardrobe protection.
- Сохранить физическую идентичность, LRU fairness, rotation, запрет истинных duplicate listings/photos, и корректные исключения при малом pool.
- Нельзя подменять валидный fail состоянием «красивого, но неправильного» outfit; качество выше мнимого разнообразия.
- Не ломать Supabase Auth, quotas/rate limit, subscriptions/RevenueCat, user-owned wardrobe RLS, storage, offline queues, admin controls, App Store compliance.
- Никаких провайдерских секретов в EXPO_PUBLIC переменных; migration/deploy/server-side change — отдельно согласовывать.

## 8. Обязательный approval gate для будущего Codex/агента

Если нужна новая таблица, SQL migration/index/RPC/policy, Edge Function, изменение live ingestion/sync, production auth или cloud provider — **СТОП, ничего не применяй**. Сначала спроси пользователя и предоставь:

1. Зачем изменение невозможно или хуже сделать local-only.
2. Какую измеримую пользу получим.
3. Точный список затронутых файлов, контрактов, таблиц и функций.
4. Что может сломаться (включая отдельно четыре слота, rotation, dedup, quotas и App Store).
5. Уровень риска и существующие зависимости.
6. Backward compatibility для текущих пользователей и уже сохранённых outfits.
7. Изолированный тестовый план, контрольные метрики и rollback.
8. Цена/лимиты/нагрузка, если изменение платное.
9. Явный вопрос: **«Разрешаете выполнить именно эти изменения?»**

Отсутствие ответа = **нет согласия**. Ни скрытых миграций, ни деплоя «для проверки». Локальная ветка и документация допускаются в пределах конкретной задачи.

## 9. Критерии приёмки (предлагаемые, не измеренные SLA)

| Критерий | Требование |
| --- | --- |
| Correctness | Ни одной регрессии инвариантов в существующей test matrix |
| Variety | Меньше визуально одинаковых карточек при сохранении нужного размера candidate pool |
| Rotation | Physical-product fairness не ухудшается; разные цвета не исчезают без причины |
| Coverage | Покрытие обуви и ключевых категорий растёт **после** валидного увеличения inventory |
| Budget | Подбор проверяет реальный полный budget; не подменяет валюты |
| Data quality | У каждого внешнего offer известен источник и момент обновления |
| Latency | p50/p95 measured before/after; не ухудшается молча |
| Privacy | Fit Feedback опционален; история не передаётся сторонним merchant без consent |
| Safety | Нет production SQL/Edge writes или релиза без разрешения |

## 10. Нерешённые вопросы, требующие доказательств

1. Существуют ли контролируемые или лицензированные merchant feeds для полноценного multi-brand stock/size/price? Правила источников/ToS нужно проверить.
2. Каков реальный end-user shopping journey — telemetry, conversion, save/click rate, return outcomes? Сейчас на основании одной базы нельзя доказать продуктовый retention.
3. Какие именно страницы дают дубли — Home, Shop, stylist deck или локальный fallback? Повторные выдачи надо воспроизводить по endpoint и входным параметрам.
4. Какие критерии физической идентичности для разных colorways и размеров используются в каждом действующем helper? Нужна единая карта, не переписывание на глаз.
5. Как сравнить deployed Edge v192 с git-источниками и не допустить непреднамеренного перезаписывания live hotfixes?
6. Нужно отдельно ревью текущих Security/Performance Advisor findings и прав на exposed RPC; считать предупреждение поводом для проверки, а не доказательством эксплуатации.
7. Как разрешать конфликты между прошлым App Stabilization/VTON roadmap и новым Shopping-first направлением, не ломая обязательные release gates?

## 11. Ближайшая задача для Codex (без серверных изменений)

**Task:** Inventory & Shopping Retrieval Truth Audit, local-only.

1. Прочитать docs/ARCHITECTURE.md, docs/API_RETIREMENT_PLAN.md, текущие services, Edge source, tests.
2. Нарисовать flow каждого shopping path (client + Edge), зафиксировать пределы DB fetching и candidate filtering.
3. Сохранить локальные тестовые fixtures с реалистичными разными цветами одного loafer, двумя листингами одной модели, разными моделями с одним силуэтом, дефицитом обуви.
4. Написать диагностику/bench tests и предложить **только локальный** change-set с diff, ожидаемым эффектом, метриками и rollback.
5. Не менять production Supabase, не деплоить, не менять действующий generation contract. Если потребуется — выполнить approval gate раздела 8.

## 12. Источники и уровень уверенности

**Verified (2026-11-07):**
- GitHub README, docs/ARCHITECTURE.md, docs/API_RETIREMENT_PLAN.md, docs/product/product.md.
- src/services/stylistCatalogInventory.ts, src/services/shoppingService.ts, src/services/outfitGenerationService.ts, src/services/suggestionEngine.ts, src/lib/fit/fitEngine.ts, src/utils/shopCatalogOrder.ts.
- backend/supabase/functions/generate-outfits/index.ts (исходник в main), соответствующие тесты.
- Supabase: public tables, колонка shop_catalog, агрегированные SELECT, индексы, list_edge_functions, advisors.

**Source-derived but not independently proven:** бизнес-оценки конкурентов, рыночные цифры и предыдущие идеи из загруженной истории переписки. Это исторический контекст, не проверенная статистика рынка. Упомянутый в истории большой production-safe prompt не был отдельным доступным исходником: его наличие/полнота не считаются проверенными.

**Not verified:** реальные ошибки на устройствах, успешность App Store review сегодня, реальные цены/наличие у merchants, performance p95, пользовательские конверсии, полнота deployed-vs-repo diff.

**Decision rule:** если стратегия конфликтует с фактически действующими контрактами, побеждают контракт + тесты + user approval, а не прежняя формулировка идеи.

---
**Версия документа:** 1.0, documentation-only audit. Следующая ревизия должна добавлять замеры и ссылки на regression fixtures, а не придумывать статус «implemented» по одному обсуждению.
