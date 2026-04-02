# AIWardrobe - Serverless Architecture

## Overview

**NO SERVER NEEDED.** The mobile app calls AI APIs directly:

```
Mobile App → HuggingFace API (FREE)
         → Replicate API (pay-per-use)
         → Supabase (Database + Auth)
```

## Why Serverless?

| Approach | Cost/Month | Complexity |
|----------|------------|------------|
| **Full Server** (4GB RAM) | $20-50 | High (Docker, deploy, maintain) |
| **Light Server** (1GB RAM) | $5-10 | Medium (Docker, deploy) |
| **Serverless** (this) | **$0-5** | **Low** (no server to manage) |

## How It Works

### Image Processing Pipeline (Direct from Mobile)

1. **Classification** → HuggingFace CLIP (FREE)
2. **Background Removal** → Replicate rembg ($0.002/image)
3. **Studio Enhancement** → Replicate Ghost Mannequin ($0.01/image)
4. **Description** → HuggingFace BLIP-2 (FREE)

### Cost Example (100 images/month)

| Service | Cost |
|---------|------|
| HuggingFace | FREE (30k calls/month) |
| Replicate (bg removal) | $0.20 |
| Replicate (studio) | $1.00 |
| **Total** | **~$1.20/month** |

## Setup

### 1. Get API Keys

**HuggingFace** (FREE):
- Sign up: https://huggingface.co/join
- Get token: https://huggingface.co/settings/tokens
- 30,000 free calls/month

**Replicate** (Pay-per-use):
- Sign up: https://replicate.com
- Get token: https://replicate.com/account/api-tokens
- Pay only for what you use (~$0.01/image)

### 2. Configure Mobile App

```bash
# .env file in mobile app root
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# AI Services (called directly from mobile)
EXPO_PUBLIC_HF_TOKEN=hf_your-huggingface-token
EXPO_PUBLIC_REPLICATE_TOKEN=r8_your-replicate-token
```

### 3. No Server Deploy Needed!

Just build and publish the mobile app:
```bash
eas build --platform ios  # or android
```

## Security Considerations

**API Keys in Mobile App?**
- HuggingFace tokens are safe (read-only, rate-limited)
- Replicate tokens are safe (usage-based billing, you control limits)
- Both can be rotated instantly if needed
- Alternative: Use Supabase Edge Functions as proxy (see below)

## Optional: Supabase Edge Functions

If you want to hide API keys, use Supabase Edge Functions:

```typescript
// supabase/functions/process-image/index.ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

serve(async (req) => {
  const { image } = await req.json()
  
  // Call HuggingFace/Replicate here
  // API keys stored server-side in Supabase
  
  return new Response(JSON.stringify({ result }))
})
```

**Deploy:**
```bash
supabase functions deploy process-image
```

## Architecture Comparison

### Before (Full Server)
```
Mobile → Your Server (4GB) → HuggingFace/Replicate
         ↓
       Docker, Nginx, Deploy, Monitor
```

### After (Serverless)
```
Mobile → HuggingFace (FREE)
     → Replicate ($0.01/img)
     → Supabase (Auth/DB)
```

## When to Use a Server?

You might want a server if:
- You need heavy caching (process same images repeatedly)
- You have complex business logic to hide
- You need WebSockets for real-time features
- You want to batch/process offline

**For most mobile apps, serverless is simpler and cheaper.**

## Monitoring

Track costs:
- HuggingFace: https://huggingface.co/settings/billing
- Replicate: https://replicate.com/account/billing

Set up alerts for unexpected usage spikes.

## Troubleshooting

**"Network Error" on mobile:**
- Check API tokens are correct
- Verify internet connection
- Check HuggingFace/Replicate status pages

**Slow processing:**
- HuggingFace: First call warms up (5-10s), then fast
- Replicate: Typically 2-5 seconds

## Migration from Server

If you had a server before:

1. Remove all server files: `docker-compose.yml`, `api/`, `backend/`
2. Install mobile packages: `npm install @huggingface/inference replicate`
3. Update `.env` with API tokens
4. Replace API calls with `ExternalAIService`
5. Delete server infrastructure

Done! No more server bills.
