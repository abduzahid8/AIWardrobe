# IDM-VTON on Google Colab — Step-by-Step Guide

## Prerequisites

1. **Google account** with access to [Google Colab](https://colab.research.google.com)
2. **ngrok account** (free) — sign up at [ngrok.com](https://dashboard.ngrok.com/signup) to get a public URL *(optional — you can use Gradio share instead)*

---

## Quick Start (5 Cells)

### Cell 1: Upload Files

Upload `colab_setup.py` and `tryon_api.py` to Colab. The easiest way:

```python
# In Colab, click the folder icon (📁) on the left sidebar
# Then click the upload button (⬆️) and upload both files
# OR use this to clone your whole repo:
!git clone https://github.com/YOUR_USERNAME/AIWardrobe.git /content/AIWardrobe
```

### Cell 2: Select GPU Runtime

> **Runtime → Change runtime type → T4 GPU**

Verify GPU:
```python
!nvidia-smi
```

You should see **Tesla T4** with **15 GB** memory.

### Cell 3: Run Setup (⏱️ ~10-15 minutes)

```python
!python /content/AIWardrobe/colab_tryon/colab_setup.py
```

This installs all dependencies and downloads the ~15 GB model. **Run this once per session.**

### Cell 4: Start the API Server

**Option A — Gradio Share URL (easiest, no signup):**
```python
!python /content/AIWardrobe/colab_tryon/tryon_api.py --use-gradio
```
You'll get a public URL like `https://xxxxx.gradio.live` — this is a visual UI you can test directly in browser.

**Option B — FastAPI + ngrok (for app integration):**
```python
# Replace YOUR_TOKEN with your ngrok auth token from https://dashboard.ngrok.com
!python /content/AIWardrobe/colab_tryon/tryon_api.py --ngrok-token YOUR_TOKEN
```
You'll get a URL like `https://xxxx-xx-xx.ngrok-free.app` — use this in your AIWardrobe app.

### Cell 5: Test It

**For Option A (Gradio):** Just use the web UI that opens.

**For Option B (API):** Test with curl from another terminal:
```bash
# Health check
curl https://YOUR_NGROK_URL/health

# Try-on (replace with actual base64 images)
curl -X POST https://YOUR_NGROK_URL/tryon \
  -H "Content-Type: application/json" \
  -d '{
    "person_image": "BASE64_PERSON_IMAGE",
    "garment_image": "BASE64_GARMENT_IMAGE",
    "garment_description": "Short sleeve round neck t-shirt",
    "denoise_steps": 30,
    "seed": 42
  }'
```

Or visit `https://YOUR_NGROK_URL/docs` for the interactive Swagger UI.

---

## API Reference

### `GET /health`
Returns GPU status and model loading state.

### `POST /tryon`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `person_image` | string (base64) | *required* | Photo of person |
| `garment_image` | string (base64) | *required* | Photo of garment |
| `garment_description` | string | `"Short sleeve round neck t-shirt"` | Garment text description |
| `auto_crop` | bool | `true` | Auto-crop to 3:4 ratio |
| `denoise_steps` | int (20-40) | `30` | Quality vs speed tradeoff |
| `seed` | int | `42` | Reproducibility seed |

**Response:**
```json
{
  "result_image": "BASE64_PNG_IMAGE",
  "elapsed_seconds": 25.3
}
```

---

## Tips & Troubleshooting

| Issue | Fix |
|-------|-----|
| **"No GPU detected"** | Runtime → Change runtime type → T4 GPU |
| **CUDA out of memory** | Reduce `denoise_steps` to 20, or restart runtime |
| **Session disconnected** | Re-run Cells 3-4 (setup + server) |
| **ngrok URL changed** | Update the URL in your app config |
| **Slow first inference** | First run takes longer due to JIT compilation, subsequent runs are faster |

## Performance Expectations

| Metric | T4 GPU |
|--------|--------|
| **Model loading** | 2-5 minutes |
| **Per image** | 20-40 seconds |
| **Session limit** | ~4-12 hours (free tier) |
