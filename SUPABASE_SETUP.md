# Setup API Keys in Supabase (Secure Method)

## Overview
API keys are stored in Supabase database, mobile app calls Edge Function.
**Keys never exposed to mobile app.**

## Step-by-Step Setup

### 1. Get Your API Tokens

**HuggingFace (FREE):**
- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Name: `AIWardrobe`
- Role: `read`
- Copy token: `hf_xxxxx`

**Replicate (Pay-per-use):**
- Go to https://replicate.com/account/api-tokens
- Copy token: `r8_xxxxx`

### 2. Add Keys to Supabase (Manual)

Go to your Supabase Dashboard → SQL Editor → New query → Run:

```sql
-- Create table for API keys (run once)
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add your API keys (replace with real tokens)
INSERT INTO app_config (key, value) VALUES
    ('hf_token', 'hf_your_real_huggingface_token'),
    ('replicate_token', 'r8_your_real_replicate_token')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    created_at = NOW();
```

**Or use the SQL file I created:**
```bash
cd /Users/zohidvohidjonov/Desktop/AIWardrobe
supabase db push  # Pushes the migration file
```

### 3. Deploy Edge Function

```bash
# Install Supabase CLI if not already
npm install -g supabase

# Login
supabase login

# Link your project
supabase link --project-ref your-project-ref

# Deploy the Edge Function
supabase functions deploy ai-process

# Set secrets (Edge Function needs these)
supabase secrets set SUPABASE_URL=https://your-project.supabase.co
supabase secrets set SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### 4. Update Your .env (Mobile App)

Your `.env` only needs Supabase (no API keys):

```bash
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

**No HF_TOKEN or REPLICATE_TOKEN in .env!** They live in Supabase now.

### 5. Test It

```bash
npx expo start --clear
```

Upload a photo → It will call Supabase Edge Function → AI processing happens securely.

## How It Works

```
Mobile App → Supabase Edge Function → HuggingFace/Replicate
                ↓
           API Keys stored here
           (secure, server-side)
```

## Security Benefits

| Before (Direct) | After (Supabase) |
|-----------------|------------------|
| Keys in mobile app | Keys in Supabase only |
| Can be extracted from APK/IPA | Never exposed to app |
| Users can see tokens | Users only see Supabase URL |
| Rotate = rebuild app | Rotate = update DB instantly |

## Files Created

| File | Purpose |
|------|---------|
| `supabase/migrations/20260329_app_config.sql` | Creates secure table |
| `supabase/functions/ai-process/index.ts` | Edge Function (proxies AI calls) |
| `src/services/externalAIService.ts` | Updated to use Edge Function |

## Cost Still $1-5/month

- Supabase Edge Functions: FREE tier (500k invocations/month)
- HuggingFace: FREE (30k calls/month)
- Replicate: ~$0.01/image (same as before)

## Troubleshooting

**"Failed to send a request to the Edge Function"**
→ Make sure you deployed: `supabase functions deploy ai-process`

**"API keys not configured"**
→ Run SQL in Supabase to insert keys

**"Authorization failed"**
→ Check SUPABASE_SERVICE_ROLE_KEY is set correctly
