# AIWardrobe - Final Setup Summary

## Architecture

**Serverless with Supabase Edge Functions**
```
Mobile App → Supabase Edge Function → NVIDIA (FREE) → Replicate ($0.002/img)
                     ↓
              API Keys Stored Here
              (Secure, never exposed)
```

## API Services Used

| Task | Service | Model | Cost |
|------|---------|-------|------|
| **Classification** | NVIDIA NIM | Granite Vision 3.1 | **FREE** (10k req/month) |
| **Description** | NVIDIA NIM | Granite Vision 3.1 | **FREE** (10k req/month) |
| **Background Removal** | Replicate | rembg | **$0.002/image** |

## Setup Steps

### 1. Get API Keys

**NVIDIA (FREE):**
1. Go to https://build.nvidia.com/explore/discover
2. Sign up / Log in
3. Find "Granite Vision" model
4. Click "Get API Key"
5. Copy token (starts with `nvidia_` or similar)

**Replicate (Pay-per-use):**
1. Go to https://replicate.com/account/api-tokens
2. Copy token (starts with `r8_`)

### 2. Add Keys to Supabase

Go to Supabase Dashboard → SQL Editor → Run:

```sql
-- Create table (if not exists)
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add your API keys (replace with real tokens)
INSERT INTO app_config (key, value) VALUES
    ('nvidia_token', 'nvidia_your_actual_token_here'),
    ('replicate_token', 'r8_your_actual_token_here')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    created_at = NOW();
```

### 3. Deploy Edge Function

```bash
# Navigate to project
cd /Users/zohidvohidjonov/Desktop/AIWardrobe

# Deploy the Edge Function
supabase functions deploy ai-process

# Verify it's deployed
supabase functions list
```

### 4. Mobile App Environment

Your `.env` file only needs Supabase:

```bash
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
# No API keys here - they live in Supabase!
```

### 5. Run the App

```bash
# Install dependencies
npm install

# Start development
npx expo start --clear
```

## Cost Breakdown

**Monthly estimate for 100 images:**
- NVIDIA (classification + description): **FREE** (10k requests/month)
- Replicate (background removal): **$0.20** ($0.002 × 100)
- **Total: ~$0.20/month**

**Monthly estimate for 500 images:**
- NVIDIA: **FREE**
- Replicate: **$1.00**
- **Total: ~$1.00/month**

## Files Changed

| File | Purpose |
|------|---------|
| `supabase/migrations/20260329_app_config.sql` | Creates secure table for API keys |
| `supabase/functions/ai-process/index.ts` | Edge Function (calls NVIDIA + Replicate) |
| `src/services/externalAIService.ts` | Mobile service calling Edge Function |
| `src/features/wardrobe/useVideoAnalysis.ts` | Updated to use ExternalAIService |
| `screens/MyClosetScreen.tsx` | Updated to use ExternalAIService |

## Testing

1. Open app on device/simulator
2. Go to "My Closet"
3. Tap "Upload" → "AI Studio Photo"
4. Select a clothing image
5. Should see:
   - "Analyzing with AI..." (NVIDIA classification)
   - "Saving to Your Wardrobe..." (Replicate bg removal)
   - Result: Clothing item with transparent background

## Troubleshooting

**"NVIDIA API key not configured"**
→ Run SQL in Supabase to add nvidia_token

**"Replicate API key not configured"**
→ Run SQL in Supabase to add replicate_token

**"Failed to send request to Edge Function"**
→ Deploy function: `supabase functions deploy ai-process`

**Slow responses**
→ First call to NVIDIA warms up (5-10s), subsequent calls are fast

## Security

- API keys stored in Supabase (server-side only)
- Mobile app never sees API keys
- Keys can be rotated instantly without app update
- Edge Function has service_role access only

## Support

- NVIDIA NIM Docs: https://docs.api.nvidia.com/
- Replicate Docs: https://replicate.com/docs
- Supabase Edge Functions: https://supabase.com/docs/guides/functions
