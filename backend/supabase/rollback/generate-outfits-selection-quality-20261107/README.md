# generate-outfits selection quality — 2026-11-07

Baseline: deployed ACTIVE v191 / client and source `088432be532b`.

The candidate adds construction-aware whole-outfit rejection for dress trousers
with hybrid/comfort dress shoes, jersey drawstrings with driving loafers,
washed denim with refined tailoring, and lightweight linen with heavy winter
clothing. Casual/travel chinos, summer linen drawstrings, fine merino layering
and slim rubber dress soles retain positive regressions.

NVIDIA receives up to nine approved alternatives from the existing 24-look
reserve, including a feasible seed deck. Its ranking participates in final
selection after hard identity/diversity, rotation and material quality-gap
checks. Brand novelty cannot lower candidate quality or the deck score.

Essential verification: 75 targeted regression checks across the new selection
quality/curated-handler checks, human QA, repetition/history, hosted quality
floor and operation-bound suites. Targeted TypeScript and generate-outfits Deno
checks passed. Existing limits remain: 384 quartet trials, 48 admitted reserve
looks trimmed to 24, 512 deck search steps, and unchanged provider calls/retries.

## Controlled rollout

Compare production version/fingerprint with manifest baseline immediately
before deployment. Verify every bundled input against committed source and the
candidate bundle hash. Deploy only `generate-outfits`, preserving `verify_jwt`
and the exact application authentication block. Re-list functions to confirm
ACTIVE, expected version/fingerprint and unchanged other functions. Download
the deployed body and compare the embedded original source byte-for-byte with
the verified candidate bundle.

## Rollback

`v191-source.tar.gz` contains the downloaded, verified v191 original as
`index.ts`. Its SHA-256 is recorded in `manifest.json` and matches the prior
verified deployment artifact. Extract to isolated staging, verify its hash,
and redeploy only `generate-outfits` with entrypoint `index.ts` and the recorded
`verifyJwt` setting, after checking the live candidate fingerprint. Supabase
assigns a new monotonic version on rollback; verify ACTIVE and downloaded
original source against the archived bytes. Retain all data and configuration.
