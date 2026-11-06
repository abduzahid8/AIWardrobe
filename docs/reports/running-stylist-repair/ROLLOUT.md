# Runtime repair — deployment remains unexecuted

Client code: `41ad036` (Formal verification correction after `033a21d`) (isolated branch based on `4ce2023`). Metro: `/Users/v.zahid/AIWardrobe-running-repair`, port 8082. Native host: Expo Go (`host.exp.Exponent`), booted device `BBE8D6A5-D0B5-4DE2-8808-EC5BBEB1B133`.

Confirmed original mismatch: port 8081 belonged to `/Users/v.zahid/AIWardrobe`, HEAD `1f0b6b2` plus extensive dirty changes. The clean published repair worktree was not serving the simulator. An actual pre-repair screenshot and accessibility observation captured duplicated Uzbek descriptions.

Hosted function was last verified as **v179**, updated October 30, 2026, 20:06:58 Asia/Tashkent. Management metadata and downloaded source confirm it omits the November 5 domain-quality changes. Its six-per-slot reserves, shoe prepend and final 40 cap explain the small effective bottom pool. Its diversity flag indicates equal scores, not proof that stylist quality was applied meaningfully.

The client now judges legacy proposals with its current occasion rules and builds a bounded validated reserve from live inventory. Fallback decks use physical-product matching, account-wide product recency and persisted displayed-deck history. All displayed products now cross occasion/style changes within the account; a legacy account window is retained during migration. The legacy soft-sorting cache is account scoped; hydration cannot replace products just displayed. The November 6 simulator checks exposed and corrected Formal pattern/base/footwear admission gaps; the passing retry is recorded in EVIDENCE.md. A specific three-language shortfall explains when the checked inventory cannot form three complete Formal looks.

## Versioned artifacts

- Candidate Edge source: `20261106/aiwardrobe-stylist-41ad036-edge-source.tar.gz`, Git commit `41ad036`, generation version `stylist-runtime-repair-v2`.
- Candidate archive SHA-256: `6668a954747c71af99aaee413548f09ecd84e1a4f32d66d1d8d9d52e6154cd99`.
- Downloaded v179 rollback source: `aiwardrobe-hosted-v179-source.tar.gz`.
- Rollback archive SHA-256: `25e33ee8415ad048a4b2513fd723d5d1812891639793a462c86ec5353d81bf21`.
- Original hosted bundle SHA-256 (Supabase): `7249e5b3d639789bee75c0117de382286d19acf656fbd12e37c47e82879967db`.

## Narrow rollout, only after approval

1. Verify the archive hashes; re-read active hosted version to detect intervening changes. Preserve existing environment secrets and JWT setting (`verify_jwt:false`; the function authenticates through its admission gate).
2. In an extracted candidate tree, deploy **only** `generate-outfits` to project `fyqpifmrsftsfqibhwhy`, using `backend` as the CLI working directory. No migrations, data changes, secrets, billing changes or other functions are part of this rollout.
3. Confirm responses report `generationVersion:stylist-runtime-repair-v2`. Verify three four-slot looks for Work/Casual/Travel/Formal; inspect consecutive displayed product IDs, request latency and catalog-read counts. The provider prompt stays capped at 40 products, balanced up to ten per slot. Missing discovery RPC uses a cached, bounded six-page-per-slot fallback; an incomplete search cannot silently establish global exhaustion.
4. On a generation/validation/latency regression, deploy only the downloaded v179 source with the original JWT setting. Supabase assigns a new numeric deployment version; verify its source matches the archived v179 artifact. No data rollback is needed.

No hosted deployment, main merge, force-push or PR creation has been performed. The local Edge handler was not run against production admission counters; retrieval measurements execute only the real candidate-read functions. Docker is unavailable, so those measurements are explicitly not a local Supabase runtime test. Simulator results use hosted v179 and the repaired client.
