# AIWardrobe — Backend Integration Guide

## Architecture Overview

```
React Native App (Expo)
        │
        │  HTTPS (Supabase JWT in Authorization header)
        ▼
┌─────────────────────────────────────────────────────┐
│  Fastify Backend  (backend/)  port 3001             │
│                                                     │
│  Routes: /api/v1/                                   │
│    user · wardrobe · outfits · wear-logs            │
│    ai · analytics · upload                          │
│                                                     │
│  Workers (BullMQ)                                   │
│    uploadProcessing · dailySuggestions              │
│    wearPrompt · streakCalculation                   │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  PostgreSQL       Redis          External APIs
  (Supabase)   (cache/queue)   Gemini · AliceVision
                                  · Weather
```

---

## Quick Start

### 1. Install backend dependencies
```bash
cd backend
npm install
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in all values — see comments in .env.example
```

### 3. Run Prisma migration
```bash
npm run db:generate   # Generate Prisma client types
npm run db:push       # Push schema to Supabase (dev)
# OR for production:
npm run db:migrate    # Run migration files
```

### 4. Start the server
```bash
npm run dev           # Development (tsx watch)
npm run build && npm start  # Production
```

### 5. Configure the React Native app
Add to your `.env` (at the workspace root):
```
EXPO_PUBLIC_API_URL=http://localhost:3001
```

---

## API Reference (`/api/v1/`)

All endpoints require `Authorization: Bearer <supabase_jwt>` unless noted.

### User

| Method | Path | Description |
|--------|------|-------------|
| GET | `/user/me` | Get current user profile |
| PATCH | `/user/me` | Update profile / style preferences |
| DELETE | `/user/me` | Delete account (GDPR) |
| GET | `/user/me/closets` | List user's closets |
| POST | `/user/me/closets` | Create a new closet |

### Wardrobe

| Method | Path | Description |
|--------|------|-------------|
| GET | `/wardrobe` | List all clothing items |
| POST | `/wardrobe` | Add a clothing item |
| PATCH | `/wardrobe/:id` | Update a clothing item |
| DELETE | `/wardrobe/:id` | Delete a clothing item |
| POST | `/wardrobe/:id/favorite` | Toggle favorite |

### Outfits

| Method | Path | Description |
|--------|------|-------------|
| GET | `/outfits` | List outfits (saved only by default) |
| POST | `/outfits` | Create an outfit |
| PATCH | `/outfits/:id` | Update outfit |
| DELETE | `/outfits/:id` | Delete outfit |
| POST | `/outfits/:id/save` | Toggle saved |
| POST | `/outfits/:id/rate` | Rate outfit (1–5) |

### Wear Logs

| Method | Path | Description |
|--------|------|-------------|
| GET | `/wear-logs` | List wear logs (newest first) |
| POST | `/wear-logs` | Log a wear event |
| DELETE | `/wear-logs/:id` | Delete a wear log |

### AI

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ai/chat` | Multi-turn stylist chat |
| POST | `/ai/analyze-clothing` | Detect clothing from base64 image |
| POST | `/ai/generate-outfits` | Generate outfit suggestions |
| GET | `/ai/daily-suggestion` | Get today's pre-generated outfit |

### Upload Queue

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload/enqueue` | Enqueue photo for async processing |
| GET | `/upload/status` | Get queue status for current user |
| DELETE | `/upload/:tempId` | Cancel a pending upload |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics` | Full wardrobe analytics snapshot |

### Health (no auth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Redis + DB health check |

---

## Frontend Integration

### API Client (`src/lib/api.ts`)

```ts
import { wardrobeApi, aiApi, wearLogApi, userApi } from './src/lib/api';

// Fetch wardrobe
const items = await wardrobeApi.list();

// Chat with AI stylist
const { response } = await aiApi.chat({ message: 'What should I wear today?' });

// Log a wear
await wearLogApi.log({ itemIds: ['id1', 'id2'], date: '2026-03-23', occasion: 'casual' });
```

All API calls automatically attach the Supabase JWT from the active session.
A `401` response triggers one automatic token refresh before rethrowing.

### wardrobeStore changes

`fetchItems()` and `rehydrateFromCloud()` now call the backend API first and fall back to the legacy Supabase direct queries if the backend is unreachable. `addItem()`, `removeItem()`, and `toggleFavorite()` use optimistic updates and call the API with offline fallback to the pending-actions queue.

### aiProviderService changes

| Method | Before | After |
|--------|--------|-------|
| `chat()` | Direct Gemini call | `POST /api/v1/ai/chat` |
| `analyzeImage()` | `POST /api/analyze-clothing` | `POST /api/v1/ai/analyze-clothing` |
| `generateOutfit()` | `POST /api/outfit-recommendation` | `POST /api/v1/ai/generate-outfits` |
| `processUpload()` | `POST /api/process-upload` | `POST /api/v1/upload/enqueue` (async) |

---

## Background Workers

Workers are auto-started when `server.ts` is loaded. They process jobs from Redis/BullMQ queues:

| Worker | Queue | Trigger | Purpose |
|--------|-------|---------|---------|
| `uploadProcessing` | `upload-processing` | On enqueue | AliceVision → ClothingItem creation |
| `dailySuggestions` | `daily-suggestions` | Cron 08:00 | Generate + cache + notify |
| `wearPrompt` | `wear-prompt` | Cron 20:00 | Evening wear-log reminder |
| `streakCalculation` | `streak-calculation` | After wear log | Recalculate streak |

To schedule the cron jobs, call these from a cron runner (e.g. `node-cron` or an external scheduler):
```ts
import { scheduleDailySuggestions, scheduleWearPrompts } from './backend/server';

// 08:00 daily
cron.schedule('0 8 * * *', scheduleDailySuggestions);
// 20:00 daily
cron.schedule('0 20 * * *', scheduleWearPrompts);
```

---

## Caching Strategy

| Data | Cache Key | TTL |
|------|-----------|-----|
| Wardrobe items | `wardrobe:{userId}` | 5 min |
| Outfits | `outfits:{userId}` | 60 min |
| Daily suggestion | `daily:{userId}` | 60 min |
| Analytics | `analytics:{userId}` | 30 min |
| Weather | `weather:{city}` | 10 min |
| Gemini context | `gemini:ctx:{userId}` | 15 min |

Cache is **invalidated** automatically on every mutating operation (add/update/delete item, log wear, etc.).

---

## Environment Variables Summary

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | Supabase PostgreSQL connection string |
| `SUPABASE_URL` | ✅ | Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | Server-side admin key |
| `REDIS_URL` | ✅ | Redis connection string |
| `GEMINI_API_KEY` | ✅ | Google AI Studio key |
| `ALICE_VISION_SERVICE_URL` | ✅ | AliceVision microservice URL |
| `WEATHER_API_KEY` | ✅ | OpenWeatherMap API key |
| `EXPO_ACCESS_TOKEN` | optional | For push notifications |
| `SENTRY_DSN` | optional | Error monitoring |

---

## Deployment Notes

- **Render**: Use `render.yaml` at workspace root — it already configures the Express API (`api/`). Add a second service for the Fastify backend pointing to `backend/`.
- **Redis**: Use [Upstash](https://upstash.com/) for serverless Redis (free tier available). Set `REDIS_URL` to the `rediss://` TLS URL.
- **AliceVision**: The Python microservice must be running separately. See `colab_tryon/` for the Colab notebook setup, or deploy `api/` Python service independently.
- **Node version**: Requires Node ≥ 20 (uses `--experimental-vm-modules` for Jest and ES2022 targets).
