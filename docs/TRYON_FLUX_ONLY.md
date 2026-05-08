# Try-On Pipeline — FLUX-only (April 2026)

## What changed

The old pipeline ran a deterministic anchor-box compositor before FLUX (slow,
hard-coded per-garment geometry, looked pasted-on). That entire step is gone.

New flow per garment (top → layer → pants → shoes):

1. Snapshot `preStep` pixels of the current dressed mannequin.
2. Clean the garment image (RMBG background removal + crop, cached on disk).
3. Build a side-by-side composite: `[ current mannequin (768×1024) | clean garment (768×1024) ]`.
4. Send composite + dressing prompt to **FLUX.1-Kontext-dev**.
5. Take FLUX's left half = dressed mannequin candidate.
6. Compute a pixel-diff mask between `preStep` and the FLUX candidate.
   Threshold = 14 / 255 mean-abs RGB delta. Mask covers exactly the pixels
   FLUX changed = the garment region.
7. **Drift guard**: if the diff mask covers >85% of the canvas, the
   mannequin moved (FLUX shifted camera/proportions). Step fails hard.
8. Dilate (8 px) + feather (4 px) the mask, then merge:
   - Inside mask  → FLUX pixels (realistic drape, folds, hems).
   - Outside mask → snap back to `preStep` (mannequin pixel-locked).

Result: clothes look genuinely worn, mannequin pixels never drift.

## Why NVIDIA's hosted cloud endpoint cannot be used

Per NVIDIA's own API reference for `flux.1-dev` and friends:

> "Preview API NIM supports only a predefined set of images. The image
>  should be in form of `data:image/png;example_id,{example_id}` with
>  `example_id` in a range [0,3]."

The hosted `https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev`
endpoint is a 4-image demo. Any custom user image returns HTTP 500.
There is no `flux_1-kontext-dev-infer` reference page on docs.api.nvidia.com.
Cloud is dead-end for custom images.

## Provider — self-hosted NIM

Required env in `api/.env`:

```
FLUX_PROVIDER=nvidia_local
FLUX_LOCAL_URL=http://localhost:8000/v1/infer
```

Run the container (needs CUDA GPU — Hopper / Ada Lovelace / Blackwell):

```bash
docker run -it --rm --gpus all \
  --shm-size 32gb \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/black-forest-labs/flux.1-kontext-dev:latest
```

Wait for `Pipeline warmup: done` in the logs (≈3 min on first start).

## Run the end-to-end test

```bash
node scripts/test-tryon-flux-only.mjs                          # default top
node scripts/test-tryon-flux-only.mjs <garment.png> top
node scripts/test-tryon-flux-only.mjs <pants.png>  pants
```

Outputs in `scripts/out/`:

| File | What it shows |
|---|---|
| `tryon-input-mannequin.png` | starting mannequin |
| `tryon-input-garment.png` | cleaned garment we sent to FLUX |
| `tryon-flux-input-composite.png` | the side-by-side fed to FLUX |
| `tryon-flux-raw-step1.png` | FLUX's raw side-by-side response |
| `tryon-mask-step1.png` | diff mask (white = where FLUX changed pixels) |
| `tryon-final-step1.png` | **final dressed mannequin** |
| `tryon-metrics.json` | timing + coverage + preservation stats |

`tryon-metrics.json` contains:

```json
{
  "totalMs":            <full pipeline ms>,
  "fluxMs":             <just the FLUX network call ms>,
  "coveragePct":        <% of pixels FLUX changed>,
  "insideMaskPct":      <% of canvas inside merged mask>,
  "outsideMaskExactMatchPct":    <should be ≈100>,
  "outsideMaskMaxChannelDelta":  <should be 0 — proves mannequin is pixel-locked>,
  "drifted": false
}
```

`outsideMaskExactMatchPct = 100` and `outsideMaskMaxChannelDelta = 0` are the
hard proof that the mannequin/background did not change.

## Known weak points to expect after first test

- If FLUX rotates / re-frames the mannequin (camera drift), `coveragePct`
  spikes >85% and the drift guard fails the step. This is a prompt-strength
  problem; tighten `buildDressingPrompt` in `api/routes/tryon.js`.
- Mask edges around hair / hem boundaries can be slightly blurry due to the
  feather (4 px). If that becomes visible, drop feather to 2 or raise the
  diff threshold above 14.
- Multi-garment chains: each step uses the previous step's `basePx`, so any
  artifact compounds. Always inspect `tryon-final-step1.png` for the first
  garment before adding step 2.
