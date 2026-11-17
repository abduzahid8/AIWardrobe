# Critical journeys, beta defaults and fallbacks

Local unit/contract tests are not proof that a device journey passed. All installed end-to-end statuses below remain MANUAL REQUIRED. The RC full-regression log establishes the local automated status separately.

| Tier | Journeys | Local evidence | Beta policy |
|---|---|---|---|
| A | Account/session startup; add clothing; wardrobe browsing; generate outfit; save/log outfit; basic Style/Fit; BuyCheck; family/variant; offline; subscription recovery | Auth/store/media/calendar/backend generation/purchase/catalog suites; Q1–Q8 still required | Core UI enabled in existing client; premium remains server-authoritative and requires restore validation |
| B | Recreate look; Fit DNA; reference garments; Find Better; Same Look Less Money; Maximum Wardrobe Gain; gaps; weekly plan; evidence-backed price intelligence | purchase/styleOS/visualOS/growth suites; many advanced services are domain-level and do not prove shipped UI wiring | Enable only a reviewed reachable journey after Stage 0 proof; advanced price history OFF initially |
| C | Advanced multimodal; autonomous stylist; deep proactive; price watches; advanced VTON; experimental ranking | Domain/provider mocks only; remote controls not drilled | OFF; optional limited internal visual/VTON or ranking shadow requires approval; never a beta entry dependency |

Actual platform boolean defaults live in `src/services/platform/experimentation.ts` and are extracted into the manifest. fit_dna_v2=false, buycheck_v2=false, visual_parser=false, recreate_look=false, autonomous_stylist=false, proactive_recommendations=false, price_intelligence=false, wardrobe_graph=true, daily_stylist=false, new_retrieval=false, new_ranking=false. They govern that helper service; existing core BuyCheck, Fit feedback, family/variant UI, outfit and wardrobe paths are direct routes, not automatically disabled by these experimental flag names. launchFlagDefaults() is planning metadata with advanced defaults off; do not treat it as a live rollout controller. New ranking remains off; runShadow() is available for future approved shadow comparisons and is not automatically scheduled.

Owners: mobile owns core routes and local fallback; shopping owns discovery/family/price; fit owns evidence; backend owns paid-provider deny/budgets; release owner approves cohorts. No names or remote control destinations have been assigned. Kill overrides in the platform helper are in-memory only. The six metered Edge entries now have shared server environment stop controls (PAID_WORK_CONTROLS.md), locally tested but not deployed; remote kill drills and complete route inventory remain OPEN (RC-007). Do not claim universal or persistent remote shutdown from their unit tests.

| Advanced feature | Primary | Fallback | Offline | Kill/coverage |
|---|---|---|---|---|
| Catalog discovery | catalog_discovery_candidates | Counted bounded windows on missing RPC | Cached inventory / honest unavailable | Missing-RPC tested; helper new_retrieval is not the existing caller's kill |
| Visual parse | Vision provider | Metadata/manual or insufficient evidence | Manual | visual_parser/provider deny; platform flag wiring and remote drill pending |
| Price intelligence | Observed offer/history | Insufficient evidence; no guessed discounts | Cached stale label, no live-price claim | price_intelligence default false; existing assessment services need caller coverage audit |
| VTON | Provider render | Outfit board/no render; retain history | Unavailable | Provider-level deny required; no universal platform VTON flag exists |
| Autonomous stylist | Orchestrator | Existing deterministic workflows | Partial local | autonomous_stylist=false; actual remote route coverage pending |
| Recreate/reference | Visual parsing + catalog retrieval | Manual selection/partial match | Saved references where available | recreate_look=false; UI path availability must be checked |
| Fit DNA | Evidence-derived profile | Unknown/low confidence and reference garment input | Local recorded evidence | Experimental fit_dna_v2 off; core fit feedback stays available |
| Find Better / Less Money / Gain / gaps | Local ranked catalog evidence | Explain missing candidates/constraints | Cached subset only | Existing core tools; provider independence tested locally, UI manual |
| Weekly plan / proactive | Plan service | Manual outfit calendar | Local saved plan | daily_stylist/proactive OFF; notifications/jobs never enabled here |
| Experimental ranking | Candidate ranker shadow | Stable existing ranker | Stable cached output | new_ranking=false; approved runShadow only; failures must leave production result unchanged |

No new provider is a launch dependency. A hidden feature is acceptable only if every entry point, background job and provider call is actually gated; UI hiding alone does not cap cost. Tier C can remain deferred, but shared Tier A provider paths still need verified rate/cost/kill controls before release.
