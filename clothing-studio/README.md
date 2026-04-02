# 👔 Atelier Studio — AI Clothing Product Photography

Transform any clothing photo into a clean white-background e-commerce product shot using Claude Vision + Flux.

## How it works

1. **Upload** any photo — clothes on a hanger, store shelf, bed, floor
2. **Claude Vision** analyzes the garment (type, color, material, details)
3. **Flux (via Replicate)** generates a professional flat-lay product photo
4. **Download** your clean product shot

## Setup

### 1. Install dependencies
```bash
npm install
```

### 2. Set up API keys
```bash
cp .env.example .env.local
```

Edit `.env.local`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
REPLICATE_API_KEY=r8_your-key-here
```

**Get your keys:**
- Anthropic: https://console.anthropic.com → API Keys
- Replicate: https://replicate.com → Account → API Tokens (free tier available)

### 3. Run locally
```bash
npm run dev
```
Open http://localhost:3000

### 4. Deploy to Vercel (recommended)
```bash
npm install -g vercel
vercel
```
Add your environment variables in the Vercel dashboard.

## Cost
- **Claude Vision** (analysis): ~$0.003 per image
- **Flux Schnell** (generation): ~$0.003 per image
- **Total: ~$0.006 per product shot**

## Tips for best results
- Higher resolution input photos → better output
- Single garment per photo works best
- Good lighting in original photo helps
- Edit the generation prompt to add/remove details

## Stack
- Next.js 14 (frontend + API routes)
- Claude claude-opus-4-5 (garment analysis)
- Flux Schnell via Replicate (image generation)
- No external CSS frameworks
