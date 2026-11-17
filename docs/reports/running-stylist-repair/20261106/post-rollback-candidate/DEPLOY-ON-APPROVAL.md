# Deploy on explicit approval only — post-rollback candidate P1+P2

Do NOT deploy without the user's explicit approval for this exact archive.

Archive: `aiwardrobe-candidate-p1p2-edge-source.tar.gz`

SHA-256: `a2f77c340fe1e1e5af8703989d1e9edb839c8205cdf5de8d42358eb6a4c8d3e7`

Bundle SHA-256: `f5ab8fb32eecdebff40a61855177f997116463469b3f1071817a3658d9a2f593`

Contents: v182 source (catalog identity fix intact) + offline P1 CPU bounds,
per-request quality memo, timing diagnostics, P2 whole-outfit formality
clashes. No billing, authentication, source, anchor, RLS, secret or
database changes.

Steps (after approval, from an isolated directory):

1. Re-verify the archive hash and the live baseline: project
   `fyqpifmrsftsfqibhwhy`, function `generate-outfits`, ACTIVE v183,
   JWT verification disabled, bundle hash
   `ff4b5f02772d999c3ec57ccb973599ec7fbe75044a9e0f7e4c30ced3ba3570df`.
   Stop for any intervening deployment.
2. Extract this archive into another isolated directory. Deploy only the
   explicit `generate-outfits` name with
   `--remote=fyqpifmrsftsfqibhwhy --use-api --no-verify-jwt` (JWT stays
   disabled). No `--prune`, no other function.
3. Confirm the new ACTIVE numeric version, `verify_jwt: false`, and that the
   hosted `ezbr_sha256` equals the bundle hash above. Download the hosted
   source and diff it against the extracted bundle entry.
4. Run one Classic Work wardrobe generation and one catalog generation with
   the authorized test account; record HTTP status, latency, product overlap
   and four-slot identity. Any HTTP 546 reverts to the v179 rollback archive
   (`aiwardrobe-hosted-v179-source.tar.gz`, SHA-256
   `25e33ee8415ad048a4b2513fd723d5d1812891639793a462c86ec5353d81bf21`).
