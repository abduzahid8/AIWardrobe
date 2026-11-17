# Styling quality v193 — 2026-11-07

Работа выполнена в отдельном worktree `AIWardrobe-styling-v193`, ветка
`codex/styling-quality-v193-20261107`. Незакоммиченные изменения основного
`AIWardrobe` не переносились и не изменялись.

Релиз выполнен: generate-outfits **v193 ACTIVE**, source commit
`141c83b9c7b3ff523a9d849a038f43b4acec6caf`, [PR #2](https://github.com/abduzahid8/AIWardrobe/pull/2).
Проверено 2026-11-07 18:42:15 Asia/Tashkent: скачанный original source совпадает
с committed inputs, SHA-256 `05ab555c747f9b5b65f99da24fc2bcb99ca16b95f19675eb48b8842e1c46c2a7`.
EZBR fingerprint: `201bf3034d34340269048b3bae40c2e1d530bfc1ecb25d7d8568a2b099b8d1d9`.
Auth, rateLimit, JWT config и остальные функции не изменились. Live OPTIONS=200,
POST без авторизации=401. Аутентифицированная live generation и simulator QA
не запускались; semantic smoke выполнен офлайн на настоящем handler.

## Проверенный источник

`baseline.json`: live ACTIVE v192, `verify_jwt=false`, пересборка
`52a97e3f35957e8492c6d1200bc0dc1375b31388` совпала с скачанным оригиналом
побайтово. SHA-256 исходного bundle:
`6e126cc5944d41f5666cfcec9dca36bc57b62877acb87266408a886e9a603d39`.
Live EZBR: `07bd476dc1e698f6e4ea3a127efb06de8d677586a2aecbe9767f9511be4102af`.
`main` остался на `5784fc5c`; ветка `release/stylist-quality-v4-20261107`
ответвилась на `d9d7ffc` и не содержит всех изменений v192. Она не использовалась.
Стратегические документы из `docs/aiwardrobe-direction-audit-2026-11-07` прочитаны.

## Где принимаются решения

| Этап | Источник и функция |
|---|---|
| Авторизация, квота, лимиты | Edge `generate-outfits/index.ts`: auth перед обработкой provided wardrobe; `_shared/rateLimit.ts` |
| Budget и валюты | Edge `strictestOutfitBudget`, `budgetCandidatesFromTextEdge`, `outfitShopSpend`, `fitsBudgetAmount`; request-scoped CBU FX; общий `_shared` переэкспорт доменных правил |
| Источник, ownership, anchor | Edge загрузка wardrobe/catalog, `eligibleCandidate`, `requiredChoiceIds`, `checkCompleteChoice`, финальная валидация |
| Occasion, climate, preferences | `garmentContextReason`, `rejectForStyleOccasion`, `rejectForClimate`, `rejectForRequestIntent`, `rejectForActivity`; перед caps и в финальном judge |
| TOP/LAYER/BOTTOM/SHOES | `outfitStructure`, canonical macro/category, `fourSlotOutfitReason` |
| Материал, силуэт, узор | `garmentDescriptor.describeGarment`, `layerCompatibility`, `outfitCoherence` |
| Whole-look quality floor | `occasionRules.assessOutfitQuality` → `outfitQualityFloor`; оценка не позволяет отменять hard constraints |
| Reserve → shortlist | `validatedQuartets`: ≤384 попытки, ≤48 принятых до trim, reserve ≤24; `curatedOutfitShortlist`: ≤9, включая feasible seed |
| NVIDIA | `generateCuratedChoices` ранжирует готовые ID из shortlist; неизвестные ID отклоняются `resolveOutfitChoices`; новые вызовы не добавлены |
| Тройка и ротация | `assembleDiverseDeck` ≤512 шагов, physical identity, account history/LRU; `assignRotatingShoes` с повторной проверкой совместимости |
| Последняя защита | `signalsFor(...).acceptable` после replacement/shoe assignment; неполная тройка возвращает явную ошибку |
| Локальный fallback | `outfitGenerationService.buildLocalOutfits`, `suggestionEngine.generateCandidates` используют `passesStyleJudge` и `validatedQuartets`; здесь нет NVIDIA и лимит перебора отличается от Edge |

## Причины и исправления

1. `describeGarment` передавал paisley/jacquard в `patternOf`, который относил
   jacquard к texture. `outfitQualityFloor` сравнивал только разные семьи узоров:
   две крупные клетки также обходили проверку. Теперь motif и prominence
   определяются отдельно: paisley — print, обычный tonal jacquard — texture;
   явные subtle-мотивы могут сочетаться, конкурирующие заметные мотивы снижают
   coherence и отклоняются для smart/formal/date. Размер посадки и слово fine
   у шерсти сами по себе не доказывают размер/контраст узора.
2. `fadedDenim` не распознавал «washed light-blue slim jeans». Добавлен отдельный
   `paleDenim`, который не выдаёт светлый цвет за доказательство distress.
   Узкое whole-look правило связывает светлые джинсы + denim base + tailoring +
   dress Oxford/monk/boot для smart/formal. Чистый denim, suede loafers,
   casual/travel и denim jacket остаются доступны.
3. `layerCompatibility` штрафовал heavy слой при `weather=null`. Теперь этот
   температурный штраф требует известной температуры >22°C. Явная комбинация
   lightweight linen и heavyweight winter остаётся несовместимой по ткани,
   даже без прогноза; обычный wool/cashmere без доказательства веса не получает
   выдуманный зимний вес. Не вводились общие запреты linen/denim/cardigan.
4. Локальный `passesStyleJudge` проверял hard reasons, но не acceptance floor.
   Он и `validatedQuartets` теперь применяют тот же floor. Диагностический
   `styleJudgeReason` сохранён отдельно: Edge ремонтирует слабые карточки из
   reserve, сохраняя хорошие, вместо лишнего вызова провайдера.
5. Изменение Edge entrypoint — только идентификатор
   `stylist-whole-look-coherence-v193-20261107`; основная логика общая для Edge и
   приложения. Авторизация, rate limiting, subscription, БД и backend/api не менялись.

## Проверки и воспроизведение

Точные результаты и команды: `checks.json`. Обязательные suites: 217/217,
15 suites, включая 26 новых тестов. До изменений новые tests: 6 падений / 20.
Базовые обязательные suites: 176 passed / 15 failed; все 15 — устаревшее ожидание
шаблонного описания в adversarial. Проверки injection/source сохранены, ожидание
обновлено на уже существующую garment-based подпись.

Расширенный прогон: 468 passed / 6 failed, 21 suites. Все шесть воспроизведены
на нетронутом `52a97e3f`: ожидание shortlist=3 (фактически v192=9), два Moner
comfort-sole + dress-trousers (v192 уже отвергает), три hosted scenarios:
snow/Work с generic boots; partial single-card provider deck; anchored top
discovery. Эти отдельные старые failures не исправлялись изменениями production
логики. Они не скрыты и не считаются успешными проверками.

Общий `npm run typecheck`: 1059 diagnostics, все внутри старого архивного bundle
`docs/reports/running-stylist-repair/20261106/catalog-identity-fix/deployment/...`;
точно те же 1059 diagnostics на baseline. Полный app typecheck с исключением
архивных `docs/` прошёл. Deno-check действующего Edge entrypoint прошёл.
Targeted ESLint: 0 errors, 5 существующих warnings в `occasionRules.ts`.

`fixtures-before.json` / `fixtures-after.json`: 14 именованных сочетаний.
Было 4 ошибочных допуска, стало 0; после — 5 rejection / 9 acceptance,
тройка без повторов товаров. Это анализ текстовых fixtures, не человеческая
оценка фотографий и не измерение всего production-каталога.

```sh
npx jest --runInBand __tests__/services/stylingV193.test.ts __tests__/backend/stylingV193Hosted.test.js
npx tsx scripts/validation/styling-v193-report.ts /tmp/styling-v193-report.json
CPU_BUDGET_MS=500 CPU_REPORT=/tmp/styling-v193-cpu.json node scripts/validation/post-validation-cpu.cjs
```

CPU replay реального post-provider handler: 14/14 HTTP 200, по три полных образа,
budget/anchor/identity/recency checks прошли. Максимум CPU: baseline 464.132 ms,
после 453.546 ms; медиана (верхний центральный элемент) 239.434 / 258.282 ms.
Это локальный CPU replay, не production latency; компиляция модулей исключена.
Старый replay сначала не запускался из-за отсутствующего `useCuratedChoices`,
затем из-за прежнего требования 0 reserve-search на fresh deck. Оба дефекта
воспроизведены на v192; harness исправлен под уже существующий bounded shoe
audit. Лимиты ≤1 search, ≤24 reserve, четыре слота и <500ms сохранены.

## Release и rollback

`scripts/validation/styling-v193-release.cjs` выполняет prepare/deploy/verify.
Токен передаётся через `SUPABASE_ACCESS_TOKEN_FILE` или `SUPABASE_ACCESS_TOKEN`.
Prepare повторно сверяет live v192, сохраняет точный rollback, фиксирует hashes
всех bundle inputs, проверяет неизменность auth и rateLimit. Deploy требует
закоммиченные inputs и совпадение remote HEAD, а также прежние live version и
fingerprint. Verify скачивает опубликованный original source и сравнивает
его hash; проверяет ACTIVE, JWT config и неизменность остальных функций.

Фактический результат релиза находится в `deployment.json` и `verification.json`.
Live smoke ограничен CORS и отказом без авторизации: нет новых AI-запросов,
изменения пользовательских данных или схемы. Семантические smoke tests выполнены
на реальном handler с локальными auth/catalog/provider fixtures.

Rollback: `backend/supabase/rollback/generate-outfits-styling-v193-20261107/`.
Не удалять прежний `generate-outfits-selection-quality-20261107` с v191.

## Оставшиеся вопросы и визуальный QA

Дескрипторы опираются на catalog metadata; изображения могут ей противоречить.
Погода отсутствует явно, никакой forecast не добавлен. UI строит budget из
`promptRef/userPrompt` → `parseStylistIntent/parseBudgetFromText` →
`withOutfitBudget`, не из placeholder. Hosted tests проверяют UZS cap для всего
образа (structured и текст), отказ при 4×450000 >1500000 и отсутствие cap при
непереданном budget. Это не доказательство payload из исходного скриншота.

1. Semi-Classic / Date night: 3 полных образа; нет двух конкурирующих крупных узоров.
2. Нет светлого double-denim + tailored blazer + polished Oxford; чистый smart-casual denim остаётся.
3. Casual / Travel: graphic tee, denim jacket, sneakers и разные физические товары сохраняются при повторной генерации.
4. Пустой budget и явно введённые 1 500 000 UZS: сверить request payload и суммарную цену четырёх вещей.
5. Weather unavailable: нет выдуманной температуры; отдельно сравнить запрос с реально известной погодой и anchor из wardrobe.
