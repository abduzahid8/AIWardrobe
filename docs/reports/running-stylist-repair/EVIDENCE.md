# Simulator verification — November 6, 2026

Resumed the existing isolated worktree and Metro **8082**. Screens confirmed client `033a21d41d3c`. Only the failing Casual + Formal scenario was retried after targeted corrections; its final passing client was **`41ad036d3c8b`**. All results below are real Expo Go/iOS Simulator results, not mocked output.

| Scenario | Displayed result | Screenshot |
|---|---|---|
| Classic + Work, before restart | 3 complete looks; smart layers and shirt/polo bases | [Work 1](20261106/work-before-restart.png) |
| Classic + Work, after cold restart | 3 complete looks; 0 product overlap with Work 1 | [Work 2](20261106/work-after-restart.png) |
| Classic + Casual | 3 complete looks; field/knit jackets and mostly trainers | [Casual](20261106/classic-casual.png) |
| Classic + Travel | 3 complete looks; removable layers and trainers in all three | [Travel](20261106/classic-travel.png) |
| Classic + Formal | 3 complete looks; shirts, suit blazers, tailored bottoms and dress shoes | [Classic Formal](20261106/classic-formal.png) |
| Casual + Formal, corrected retry | 3 complete looks; collared bases, coherent tailoring and dress footwear | [Casual Formal](20261106/casual-formal-final.png) |

Every passing look has exactly **TOP + LAYER + BOTTOM + SHOES**. Work's cold restart retained its 12 displayed IDs on disk and supplied 12 previous products to the next request: **24 placements, 24 distinct normalized IDs, zero repeats**. [Persisted history](20261106/cold-restart-history.json).

The four Classic occasion decks had **48/48 distinct IDs**. Across all six passing decks there were **72 placements and 69 distinct IDs**: the final Formal deck reused three older Work shoe IDs. Immediate previous-deck overlap was zero. This is documented older-item reuse; catalog-wide exhaustion was not established. [Actual IDs and per-request results](20261106/displayed-decks.json).

Casual + Formal initially failed visual review despite valid slots. Four failed attempts are retained in [events](20261106/simulator-events.json). Necessary corrections covered competing patterns (including pinstriped/geometric evidence), Henley/unverified merino bases, conflicting tailoring and hybrid footwear. No passing scenario was rerun after these Formal-only changes. Three focused regression cases and client/Edge typechecks passed; no full Jest suite or new benchmark ran.

Uzbek product descriptions and separate advice were checked on screen; no duplicated generic placeholder appeared. English/Russian UI and an actual inventory shortfall were not exercised in this session; all tested requests ultimately found three suitable looks.

No ordinary cold-launch crash occurred. A native crash during development source reload was captured separately: [native error](20261106/development-reload-crash.json). This is not evidence of a release failure; no release build was tested.

No remaining scenario failure is observed. Development-reload instability and older-footwear reuse remain documented limitations. Hosted rollout is still unexecuted; the last verified Edge was v179, and the read-only version recheck timed out. Main, production configuration and billing were not modified. [Updated candidate artifact and rollback plan](ROLLOUT.md).
