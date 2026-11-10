# Stylist 6 twelve-scenario pilot — gate decision (2026-11-10)

**Branch:** `codex/stylist6-visual-first-20261110`
**Verdict: NO-GO for staging integration.** The pilot apparatus is complete and
measured; the acceptance gate itself is NOT passed. Details and exact unblock
steps below. No production system was touched.

## What was built and measured (all verified this session)

- `backend/stylist6/catalog_adapter.mjs` (`catalog-adapter-v1`, 9 tests green):
  maps the committed `data/` snapshot (40,989 rows) to prototype items —
  38,388 admitted real shoppable SKUs; 2,652 knit-base-layer projections with
  per-item provenance; 2,108 unknowns honestly excluded (shorts, suits, socks…).
- Wearing-role fix is visually grounded: audited `photo_3` (white turtleneck
  base under fair-isle cardigan) and `photo_5` (brown quarter-zip mockneck base
  under camel overcoat). ~27/46 references wear knit as the BASE top —
  the old `top/fine-knit`-only mapping starved retrieval while 623
  pullover-classified knits sat in the wrong slot.
- `backend/stylist6/visual_critic_hosted.mjs` (6 tests green): REAL hosted
  multimodal critic (Gemini `generateContent`, reference + product photos,
  strict JSON contract, per-candidate abstention, cost/latency recorded).
  Implemented and contract-tested with recorded fixtures — NEVER FIRED
  (no vision key in this environment; see blockers).
- `scripts/stylist6/pilot_12.mjs`: 12/12 scenarios executed, all three arms,
  same snapshot, same hard budget rule (native currency only, zero invented FX).
  Raw output (private): `/tmp/s6-pilot-20261110/pilot-private.json`.
  Blind grids (private): `/tmp/s6-pilot-20261110/blind-*.jpg` (24 sheets) +
  `blind-key.json`. Report (committed):
  `benchmarks/stylist6/pilot_12_report.json`.
- Arm A is the REAL v207 kernel (`stylingContext`/`outfitConcepts`/gates/
  `createOutfitPlan` bundled locally, network forbidden), not a mock.

## Pilot results (committed report + private decks)

| Scenario | V5 baseline | Creative-first S6 | Reference-first S6 |
|---|---|---|---|
| S6-01…S6-07 non-budget (7) | 3 looks, distinct, ~1.9 s | 3 looks, distinct, ~0.55 s, visual NOT_RUN | 3 looks, distinct, ~0.5 s, visual NOT_RUN |
| S6-08, S6-11, S6-12 | 3 looks, distinct | 3 looks, distinct | 3 looks, **item reuse across looks (dup)** |
| S6-B03 (8M UZS) | 3/3 budget-compliant | 3/3 compliant | 3/3 compliant (dup ×1) |
| S6-B01 (1.5M UZS) | 0 compliant (3 proposals over cap, dropped) | 0 looks (honest abstention) | 0 looks (honest abstention) |

- 100% of SHOWN looks: real SKUs, HTTPS photo + shop links, native prices,
  budget-compliant. Zero fabricated products, zero invented FX.
- S6-B01 infeasibility is PROVEN inventory depth, not a retrieval bug:
  cheapest brief-compatible quartet = 1,584,702 UZS > 1,500,000 cap
  (fine-knit 279,900 + jacket 258,000 + trouser 396,902 + loafer 649,900).
- Image reachability from pilot host: 87/152 (65 × HTTP 403 from brand CDNs
  such as apc.com; 5 recovered on retry). Counts recorded per arm per scenario.
  Reviewers must weigh labeled missing cells accordingly.

## Decision table (per STYLIST6_DECISION_GATE.md)

| Candidate | Real SKU hard compliance | 12-scenario blind wins | Critical fashion issues | Looks distinct? | Est. cost | Latency (retrieval) | Decision |
|---|---|---|---|---|---|---|---|
| V5 | 11/12 (B01: 0 compliant) | NOT MEASURED | NOT ASSESSED | 12/12 | $0 offline | ~1.9 s / ~2.1 s | control |
| Reference-first | 11/12 shown (B01 abstains) | NOT MEASURED | NOT ASSESSED | 8/12 (dup in 4) | $0 offline | ~0.5 s / ~10.7 s budget | **NO-GO (gate not run)** |
| Creative-first | 11/12 shown (B01 abstains) | NOT MEASURED | NOT ASSESSED | 12/12 display | $0 offline | ~0.55 s / ~11.4 s budget | **NO-GO (gate not run)** |

The 8/12-wins requirement needs blind human votes that were never cast
(the implementer cannot judge their own decks), and the hosted visual critic
never fired. Declaring a winner now would repeat the exact failure mode the
gate was written to prevent (green tests mistaken for fashion quality).

## Bottleneck diagnosis (pilot failed ⇒ name the layer)

1. **Reference rights + archive integrity (BLOCKER):** authoritative
   `AIWardrobe_Style_References_46_Pilot.zip` (with `manifest.json` SHA-256 set)
   is absent from this Mac; 46 loose photos found in
   `~/Downloads/Telegram Desktop/` have 46 unique SHAs but nothing to verify
   against. Photos carry luxury brand callouts (Zegna, Ralph Lauren, Cartier,
   Loro Piana…) — hosted ML use needs explicit rights-scope confirmation.
   Upload NOT performed (correctly). Metadata-only evaluation used instead.
2. **Visual critic (BLOCKER):** implemented + contract-tested, never fired —
   no vision API key in this environment, no spend approval, no authorized
   reference bytes committed anywhere.
3. **Catalog depth (MEASURED, not fatal):** 1.5M UZS old_money/Work has no
   compliant quartet in the UZS slice; 15.9% of snapshot rows are variant
   duplicates; 65/152 pilot image URLs 403 from this host.
4. **Taxonomy/retrieval (FIXED this session):** knit-base-layer projection;
   budget-starved shortlists widened (perRole 24/perBrief 120 under budgets).
   Remaining: display-deck item reuse in 4/12 reference-first scenarios
   (pre-novelty-filter slicing; the post-visual novelty stage would address it
   once the critic fires).
5. **Staging/iOS (NOT STARTED, correctly):** no new Node preview, no
   `aiw-stylist6-staging` proxy, no quota/user test — all gated behind a
   passed pilot per the execution plan.

## Exact unblock steps (in order)

1. Owner supplies the authoritative 46-pilot ZIP (or confirms the 46 loose
   photos as canonical) AND confirms rights scope for private research upload
   + hosted vision evaluation.
2. Run `upload_private_reference_bundle.py` (dry-run → `--upload
   --confirm-private-research --create-private-bucket`) with server-side
   Supabase credentials; keep the signed receipt private.
3. Provide a vision API key + explicit spend approval: single dry-run first,
   then ≤108 judged calls (~$2–5 est.).
4. Three reviewers vote on `blind-*.jpg` via `blind-key.json`; record
   majority per scenario in this report; apply the 8/12 rule.
5. Only on a pass: build the isolated staging backend + 30+ scenario expansion.
