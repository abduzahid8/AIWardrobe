# Cleanup Candidates

Generated during the 2026-08-21 repo reorganization pass. These files were
**left in place, not deleted or moved** — this is a list for a human to
review and decide on. Nothing here was touched.

## Deleted (2026-08-21, confirmed by user)

These were generated/output artifacts or unreferenced scratch files —
`git rm`'d, still recoverable from git history if ever needed:

`AIWardrobe-2026-06-16-174012 (2).ips`, `AIWardrobe-2026-06-16-174012.ips`,
`AIWardrobe-2026-06-16-174014.ips` (crash logs), `report.xml` (test output),
`flux_step1-4_*.jpg`, `tryon_step1-4_*.jpg` (debug pipeline images),
`Group 7(3).svg` (unreferenced stray asset), `test-check.js` / `test-check.ts`
(scratch probe, one empty), `groq_prompt_enhancements.json` (contained only
`[]`), `multi-garment-vton-roadmap.md` (superseded by
`docs/MULTI_GARMENT_VTON_ROADMAP.md`).

`maestro_test_output/` and `SplashScreen.nib/` were also deleted
(2026-08-21) — confirmed unused: `scripts/run_maestro_tests.sh` recreates
`maestro_test_output/` itself via `mkdir -p` on every run, and
`SplashScreen.nib/` was a stray compiled Xcode Interface Builder artifact
with no references anywhere in the repo.

## Needs a decision — broken or ambiguous git submodules

| Path | Why |
|---|---|
| `alicevision-service/` | Tracked as a git submodule reference (gitlink, mode `160000`) but **has no entry in `.gitmodules`** — `git submodule status` errors on it (`fatal: no submodule mapping found`). It contributes zero files either way. Either re-add it properly with a `.gitmodules` entry, or remove the dangling gitlink. |
| `open-design/` | A real, valid git submodule (`.gitmodules` has a matching entry, pointing to `github.com/nexu-io/open-design`), but it has never been initialized/cloned locally (0 files checked out) and nothing else in the repo references it. Worth confirming it's still needed before carrying it forward. |

## Flagged, not touched — likely app bug (per your instruction to just flag it)

**Two separate Supabase client instances exist:**
- `lib/supabase.ts` — used by most of `screens/`, `store/`, `hooks/`, `components/`, and some of `src/services/*`.
- `src/lib/supabase.ts` — used by `src/services/apiHealthCheck.ts`, `externalAIService.ts`, `adminService.ts`, `feedbackService.ts`, `fitService.ts`, and `src/hooks/useAdminStatus.ts`, `src/hooks/useSessionGuard.ts`, `src/lib/api.ts`, `src/services/iapService.ts`, `shoppingService.ts`, `apiClient.ts`, `aiVisionService.ts`, `outfitGenerationService.ts`, `ai/mixedOutfitService.ts`.

Running two separate Supabase client instances in the same app can cause
auth/session state to desync (one client refreshes a token, the other still
holds the stale session). Worth a dedicated fix — consolidating on one
client — as a follow-up, separate from this file-organization pass.

## Done (2026-08-21, second pass) — script/test clutter reorganized

- `scripts/scrape-md-*.mjs` (9 variants) + `scrape-lyst.mjs` /
  `scrape-lyst-playwright.mjs` → moved into `scripts/experiments/`
  (zero cross-references confirmed before moving; still there for reference,
  not deleted — `scripts/sync-massimo-dutti.js` remains the working,
  package.json-wired script for that catalog).
- `scripts/test-*.{sh,js,mjs,ts}`, `benchmarkVton.js`,
  `runThreeGarmentBenchmark.py` → moved into `scripts/manual-tests/`.
  `test-tryon-flux-only.mjs` imported from the old `../api/...` path —
  this had actually been silently broken since the `backend/` move earlier
  in this session (an `.mjs` file my first reference-audit missed); fixed
  to `../../backend/api/...` while moving it.
- `scripts/generate_pitch_deck.py` (unrelated to the app — an investor deck
  generator) → moved into `scripts/misc/`.
- `backend/mobile-vton-service/test_*.py`, `debug_masks.py`,
  `demo_color_correction.py`, `build_overall.py`, `verify_tryon_100.py`,
  `result_run.txt` → moved into `backend/mobile-vton-service/tests/`.
  Several of these compute their own repo-root path via `Path(__file__)`
  parent-traversal or `sys.path.insert` — all recomputed for the extra
  nesting level and verified to resolve to real paths. `build_overall.py`
  additionally had a hardcoded absolute path from a different machine/user
  (`/Users/zohidvohidjonov/...`) that could never have worked here — replaced
  with the same portable `Path(__file__)`-based computation as its siblings.
  `deploy_and_test.sh` (kept at the top level — it's the real orchestration
  script) had inline Python with the same hardcoded-path bug; fixed the same
  way, plus updated its `test_new_code_live.py` comments to the new
  `tests/test_new_code_live.py` path.

## Safe to delete — unreferenced asset duplicates (found in second pass)

| Path | Why |
|---|---|
| `assets/image.png` | Byte-for-byte identical (same MD5 checksum) to `assets/AIWardrobe-mainlogo.png` / `-square.png` / `-square-1024.png` — same file under a generic, unhelpful name. Unreferenced anywhere in code. |
| `assets/friends/` (`friend1.jpg`, `friend1.png`, `friend2.jpg`, `friend2.png`, `friend3.jpg`, `friend3.png`) | Entire folder (536KB) with zero references anywhere in the codebase. |

## Flagged, not touched — a real-looking API key in a tracked template file

`.env.example` (tracked in git) has a live-looking value on its last line —
`GEMINI_API_KEY=AIzaSyDw1si0_iDALhLHvh7CNOgD0f_pBRr15-0` — sitting directly
under a comment block titled "PROVIDER TOKENS — DO NOT PUT HERE" that
explains these belong in Supabase `app_config`, not here. This looks like
a real key accidentally committed into what's supposed to be a safe,
values-free template. Recommend rotating the key and removing the line;
not done here since it wasn't asked for and rotating a live key is your
call to make.
