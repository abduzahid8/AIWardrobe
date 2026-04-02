# AIWardrobe - Serverless Setup Complete

## What Changed

### Before (Server Required)
```
Mobile App → Your Server (4GB RAM) → HuggingFace/Replicate APIs
             ↓
           Docker, Deploy, Monitor, Pay $20-50/month
```

### After (Serverless - No Backend)
```
Mobile App → HuggingFace API (FREE)
         → Replicate API (~$0.01/image)
         → Supabase (Auth + Database)
```

**No server. No Docker. No deploy. Just mobile app + APIs.**

## Updated Files

| File | Change |
|------|--------|
| `.env.example` | Now only needs HF_TOKEN and REPLICATE_TOKEN |
| `src/services/externalAIService.ts` | NEW - Direct API calls from mobile |
| `src/features/wardrobe/useVideoAnalysis.ts` | UPDATED - Uses ExternalAIService |
| `screens/MyClosetScreen.tsx` | UPDATED - Uses ExternalAIService |
| `src/config/env.ts` | UPDATED - Removed ALICEVISION_URL, added AI config |

## Required API Keys

Get these free/pay-per-use API keys:

### 1. HuggingFace (FREE - 30,000 calls/month)
- Sign up: https://huggingface.co/join
- Get token: https://huggingface.co/settings/tokens

### 2. Replicate (Pay per use)
- Sign up: https://replicate.com
- Get token: https://replicate.com/account/api-tokens
- Cost: ~$0.002-0.01 per image processed

### 3. Supabase (already have this)
- Your existing Supabase project

## Environment Variables

Create `.env` file in project root:

```bash
# Supabase (REQUIRED)
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# AI Services (REQUIRED - called directly from mobile)
EXPO_PUBLIC_HF_TOKEN=hf_your-huggingface-token
EXPO_PUBLIC_REPLICATE_TOKEN=r8_your-replicate-token

# Optional
EXPO_PUBLIC_REVENUECAT_API_KEY=your-revenuecat-key
EXPO_PUBLIC_SENTRY_DSN=your-sentry-dsn
EXPO_PUBLIC_WEATHER_API_KEY=your-weather-key
```

## Install Dependencies

```bash
npm install @huggingface/inference replicate
```

## How It Works

When user uploads a photo:

1. **Classification** → HuggingFace CLIP (FREE)
   - Identifies: t-shirt, jeans, dress, etc.
   - Confidence score

2. **Background Removal** → Replicate ($0.002)
   - Removes background
   - Returns transparent PNG

3. **Studio Enhancement** → Replicate ($0.01)
   - Generates professional product photo
   - Massimo Dutti style

4. **Description** → HuggingFace BLIP-2 (FREE)
   - Auto-generates item description

## Cost Example

For 100 images/month:
- HuggingFace: FREE (30k calls/month)
- Replicate (bg removal): $0.20
- Replicate (studio): $1.00
- **Total: ~$1.20/month**

## Security

**Are API keys in the mobile app safe?**
- HuggingFace: Read-only, rate-limited, can rotate instantly
- Replicate: Usage-based billing, you control limits
- Both services allow instant token revocation

**If you need to hide keys:** Use Supabase Edge Functions as proxy (see docs/SERVERLESS.md)

## What Was Removed

The following are no longer needed (you can delete):
- `api/` folder - Node.js backend
- `backend/` folder - Fastify backend  
- `alicevision/` folder - Python AI service
- `docker-compose.yml` - Docker orchestration
- `nginx/` folder - Reverse proxy
- `scripts/` folder - Deployment scripts
- All server-related infrastructure

## Build & Publish

```bash
# iOS
eas build --platform ios

# Android
eas build --platform android

# Or run locally
npx expo start
```

## Troubleshooting

**"Module not found" errors:**
```bash
npm install @huggingface/inference replicate
```

**"Invalid API key" errors:**
- Check tokens are correct in `.env`
- Verify no extra spaces in keys
- Regenerate tokens if needed

**Network errors:**
- Check internet connection
- Verify API services status:
  - https://status.huggingface.co
  - https://replicate.com/status

## Files You Can Delete

Since we're serverless now, these are obsolete:

```bash
rm -rf api/ backend/ alicevision/ nginx/ scripts/
rm docker-compose.yml docker-compose.light.yml
rm -f docs/DEPLOYMENT.md docs/env-production-template
```

**Keep only:**
- Mobile app code (screens/, components/, src/, etc.)
- Supabase config (supabase/, lib/supabase.ts)
- `.env.example` (updated version)

## Next Steps

1. ✅ Get API keys (HuggingFace + Replicate)
2. ✅ Fill in `.env` file
3. ✅ Install dependencies: `npm install`
4. ✅ Run: `npx expo start`
5. ✅ Test image upload
6. ✅ Build and publish to App Store / Play Store

**No server maintenance ever again!**
