# `api/` Retirement Plan

**Status:** In progress. `api/` is deprecated per `docs/ARCHITECTURE.md`
ADR-001 but still has runtime callers in the mobile app.

## Consumers of `Config.api.url` (mobile bundle)

| File                                | Endpoint(s) hit              | Migration target                          | Complexity |
| ----------------------------------- | ---------------------------- | ----------------------------------------- | ---------- |
| `components/WeatherWidget.tsx`      | POST `/weather/coords`       | **Done** → Edge `weather`                 | ✅         |
| `src/services/analyticsService.ts`  | POST `/api/analytics/events` | Edge `ingest-analytics` (batch → storage) | Low        |
| `src/services/ai/healthService.ts`  | GET `/health`                | Delete; rely on supabase status page      | Low        |
| `src/services/ai/chatService.ts`    | POST `/ai-chat`              | Edge `ai-process` (already exists)        | Medium     |
| `src/services/ai/outfitService.ts`  | POST `/ai/outfit`            | Edge `generate-outfits` (already exists)  | Medium     |
| `src/services/ai/scanService.ts`    | POST `/ai/scan`              | Edge `ai-process` (vision route)          | Medium     |
| `src/services/aiProviderService.ts` | Multi                        | Edge `ai-process`                         | Medium     |
| `src/services/llmService.ts`        | POST `/ai/chat`              | Edge `ai-process`                         | Low        |
| `src/services/shoppingService.ts`   | GET `/shop/catalog`          | Direct Supabase `shop_catalog` query      | Low        |
| `src/services/flashSalesService.ts` | GET `/shop/flash-sales`      | Direct Supabase `flash_sales` query       | Low        |
| `src/lib/api.ts`                    | Axios wrapper                | Delete after callers migrated             | —          |

## AliceVision (external repository, NOT `api/`)

The Python implementation was removed from this repository in commit
`d7b11a2` and moved to the separate `abduzahid8/alicevision` repository.
This repository must not create or host an `alicevision-service/` Vercel
project: the current tree contains no Python implementation, Dockerfile, or
deployable directory for it. `backend/api/routes/alicevision.js` is only a
legacy Express proxy and is not the Python service.

The active Stylist 5 path does not depend on AliceVision. Outfit generation
and image analysis used by current application flows go through Supabase
Edge Functions (`generate-outfits` and `ai-process`). The remaining
`src/services/visionService.ts` and AliceVision branches in
`src/services/ai/{scanService,chatService,outfitService,healthService}.ts`
are legacy exports and must not be used as evidence that AliceVision is a
release dependency.

If those legacy flows are reactivated, configure `EXPO_PUBLIC_ALICEVISION_URL`
to the separately hosted Python service and validate that service in its own
repository. Do not point Vercel at this repository or at `backend/api` for
AliceVision.

## Execution order

1. `shoppingService`, `flashSalesService` → direct Supabase queries. No new
   Edge Function required. ~30 min each.
2. `analyticsService` → Edge `ingest-analytics` writes to a new `analytics_events`
   table. Idempotency by event UUID. ~1 hour.
3. `healthService` → delete. Health is checked via supabase connectivity.
4. `llmService`, `chatService`, `outfitService`, `scanService`,
   `aiProviderService` → consolidate onto `ai-process` with a discriminated
   `op` field (`"chat" | "outfit" | "scan" | "vision"`). ~2-4 hours.
5. Delete `src/services/apiClient.ts`, `src/lib/api.ts`, `Config.api.url`,
   and the whole `api/` directory. Run `npm run typecheck && npm test`.
6. Remove `EXPO_PUBLIC_API_URL` from `.env.example` and EAS Secrets.

## Guardrail

ESLint `no-restricted-imports` already blocks any mobile-side import from
`api/**` (see `.eslintrc.js`). New callers cannot be added.
