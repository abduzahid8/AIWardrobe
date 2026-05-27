# Multi-Garment Mobile VTON Engineering Roadmap

This document defines a 6-month production roadmap to replace the current sequential Mobile-VTON multi-garment loop with a faster, memory-bounded hybrid mobile + cloud GPU multi-garment system.

## 0. Executive Decision

Build a **hybrid mobile + cloud GPU** architecture:

- **Mobile:** capture, garment selection, low-resolution preview, on-device/body-prep hints, upload orchestration, cache warming, and graceful degraded UX.
- **Cloud GPU:** final multi-garment diffusion render, shared person encoding, batched/streamed garment conditioning, occlusion graph, seam refinement, and production quality control.

This is the right default because the target requires diffusion-level garment realism, fabric interaction, and multi-layer occlusion while being faster than N independent single-garment runs. A server GPU path gives enough VRAM and scheduling control to fuse the expensive work into one render. Mobile NPUs should reduce critical-path preprocessing but should not own the final 768x1024 or 1024x1024 multi-garment diffusion render in the next 6 months.

Rejected alternatives:

- **Mostly on-device final diffusion:** rejected for this 6-month goal because 3-5 garment diffusion at catalog quality will force aggressive quantization/resolution cuts, making the quality constraint the first thing to break.
- **Server-first with no mobile intelligence:** rejected because upload latency, cache misses, and repeated garment preprocessing will keep p95 latency unstable on mobile networks.

## 1. Current System Diagnosis

Current production-relevant path:

- `features/try-on/AITryOnScreen.tsx` sends selected `top`, `layer`, `pants`, and `shoes` slots as `garments` to `/api/tryon/mobile-vton` with a 180s client timeout.
- `api/services/strategies/mobileVton.js` normalizes garments with `GARMENT_RENDER_ORDER = ['top', 'layer', 'pants', 'shoes']` and calls `callMobileVtonMulti`.
- `api/services/mobileVtonClient.js` posts to `/tryon/multi` on the FastAPI Mobile-VTON service.
- `mobile-vton-service/main.py` implements `/tryon/multi` by looping over garments and calling `pipeline.inference(...)` once per garment.
- `mobile-vton-service/tryon.py` and `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py` recompute prompt embeddings, person tensors, person VAE latents, garment VAE latents, IP-adapter image embeddings, and garment UNet features per run.

The core issue is not just that garments are processed sequentially. The core issue is that each garment triggers a **full image-to-image diffusion pass** even though person identity, pose, background, body parsing, most denoising trajectory structure, output canvas, and many masks are shared.

## 2. Baseline Assumptions For All Numbers

All latency and memory numbers below are engineering targets or budgets under these assumptions unless a table row states otherwise.

| Area | Assumption |
|---|---|
| Deployment | Warm cloud GPU container, Modal or equivalent, one high-quality request per GPU worker by default |
| GPU class | A10G 24GB VRAM as baseline; L4 expected to be slower but usable; A100/H100 optional scale tier |
| Model precision | bf16/fp16 inference, no gradient tracking, `torch.inference_mode()` |
| Default render | 768x1024 final image, 10 denoising steps for production default, 6-8 steps for preview/distilled tier |
| High-quality render | 1024x1024 final image, 10-14 denoising steps |
| Max garment target | 5 garments simultaneously: top, bottom, outerwear, shoes, one accessory class |
| Current sequential baseline | One full Mobile-VTON inference per garment through `/tryon/multi` |
| Latency scope | Model compute plus local decode/encode unless marked model-only; excludes cold starts unless stated |
| Memory scope | Peak GPU VRAM includes weights and activations; process RAM includes decoded PIL/images, request buffers, masks, and response image |
| Quality scope | Multi-garment result must tie or beat manually composited single-garment outputs in blind review |

Month 1 must replace these estimates with measurements from your exact checkpoint, GPU, and request payloads. Until then, treat the numbers as strict targets, not facts about the current deployed service.

## 3. Pipeline Redesign

### 3.1 Target Inference Graph

Replace the current loop with one multi-garment request graph:

1. **Request intake and artifact resolution**
   - Accept `person_image_ref` or `mannequin_image_ref`, not large base64 JSON by default.
   - Accept `garments[]` with `id`, `image_ref`, `category`, `layer_class`, `opacity`, `material`, `length`, `fit`, `tuck_policy`, and optional `z_index_hint`.
   - Preserve the legacy shape temporarily by adapting current `garment_image` and `garments` payloads into the new schema.

2. **Mobile-side preparation**
   - Resize preview input to a known tier.
   - Compute or cache lightweight body/pose hints when available.
   - Warm garment preprocessing when a user selects an item, not when they press generate.
   - Upload image artifacts to object storage and send references to the API.

3. **Shared person analysis**
   - Decode the person/mannequin exactly once.
   - Compute person RGB tensor, VAE latent, pose/DensePose, human parsing, body-part masks, depth estimate, and frozen-region mask once.
   - Store these as a `PersonCondition` object for the request.

4. **Garment preprocessing and feature extraction**
   - Preprocess garment cutouts in parallel on CPU/GPU.
   - Encode garment image embeddings in a micro-batch.
   - Encode garment VAE latents either as a micro-batch or streamed one garment at a time into a fixed conditioning bank.
   - Cache garment embeddings by `(garment_image_hash, category, resolution_tier, encoder_version)`.

5. **Garment-to-body assignment**
   - Map each garment to semantic body regions: torso, arms, hips, legs, feet, neck, shoulder, waist, hands.
   - Produce `RegionSlot` masks with hard interior, soft boundary trimap, and conflict mask.
   - Use the same assignment code for 2 and 5 garments; empty slots are masked out.

6. **Occlusion graph construction**
   - Build a directed acyclic graph where nodes are garments and body regions.
   - Edges represent "must render above" or "must preserve below" constraints.
   - Topologically sort into render/composition priors, but do not run full diffusion per node.

7. **Batched warping**
   - Compute body-conditioned warp fields for all garments using shared pose/person features.
   - Use category-specific heads or tokens, not category-specific full pipelines.
   - Output warped garment latents, warped alpha masks, and confidence maps.

8. **Single fused diffusion render**
   - Run one denoising trajectory for the full output canvas.
   - Condition the denoiser with a fixed-size garment conditioning bank: up to 5 active slots, each with category token, layer token, pooled visual tokens, warped latent summary, and region mask.
   - Use region-gated cross-attention so each pixel attends mostly to garments assigned to that body region.
   - Preserve frozen background/person regions outside garment zones.

9. **Local refinement only when needed**
   - Run an optional crop-level seam/occlusion refiner on conflict regions only.
   - Never rerun the full canvas for the second, third, or fourth garment.

10. **Composition and validation**
   - Apply matte refinement, Laplacian boundary blending, contact shadows, and hard identity snap outside editable masks.
   - Run automated failure detectors before returning the image.

### 3.2 Parallel vs Sequential Processing

Use parallelism where it reduces critical path without multiplying full-frame diffusion memory.

| Stage | Execution | Reason |
|---|---|---|
| Image fetch/upload resolution | Parallel | Independent I/O, no GPU memory pressure |
| Garment background removal/cropping | Parallel with bounded worker pool | CPU/GPU preprocessing is independent; cap to avoid RAM spikes |
| Text prompt embedding | Batched and cached | Low memory, avoids repeated CLIP calls |
| Garment image embeddings | Batched micro-batch of 2-5 | Good GPU utilization, small compared with denoiser activations |
| Garment VAE encoding | Streamed or micro-batched by resolution | Avoid retaining full latent activations for all garments |
| Warping | Batched | Same person pose features, same output geometry |
| Diffusion denoising | Single fused pass | Primary speed win; avoids N full denoising loops |
| Seam fixes | Sequential small crops only if conflicts exist | Maintains quality while keeping memory bounded |

Rejected alternatives:

- **Full parallel single-garment runs:** rejected because latency improves only if you reserve N GPUs and memory/cost scale linearly with garments.
- **Strict sequential compositing with no fused pass:** rejected because it cannot model fabric interactions between garments and cannot beat N runs by enough.

### 3.3 Shared Encoder Strategy

Create explicit condition objects:

```text
PersonCondition
  rgb_tensor
  vae_latent
  pose_keypoints
  densepose_or_body_uv
  human_parse_masks
  depth_map
  frozen_region_mask
  editable_union_mask

GarmentCondition[i]
  garment_id
  category_token
  layer_token
  opacity_token
  text_embedding
  image_embedding
  cutout_mask
  garment_parse
  warped_latent_summary
  region_mask
  z_order_prior
```

Reusable computation:

- Person decode and resize.
- Person VAE latent.
- Body parsing and DensePose/pose.
- Background/person preservation mask.
- Global prompt negative embeddings.
- Garment text embeddings for repeated catalog descriptions.
- Garment image embeddings for repeated catalog items.
- Garment cutout and segmentation cache.
- Anchor/region masks for mannequin flows.
- Scheduler timesteps and latent allocation buffers.

Non-reusable computation:

- Garment-specific image embedding for new garment images.
- Garment-specific warp field and region mask.
- Garment-specific occlusion edges in overlapping regions.
- Final denoising interactions across all active garments.
- Local seam refinement for actual conflicts.

### 3.4 Dynamic Garment Count Handling

Use one graph with a fixed maximum slot count and active masks.

- `max_slots = 5` for months 1-6.
- Slot layout: `top`, `bottom`, `outerwear`, `shoes`, `accessory`.
- For 2 garments, only 2 slot masks are active; empty slots contain zero embeddings and zero region masks.
- For 5 garments, all slots are active; conditioning bank shape stays fixed.
- This keeps model memory mostly constant while compute grows sub-linearly through active-slot attention and conflict refinement.

Default behavior:

- 2-3 garments: one fused render, no crop refiner unless seam detector fails.
- 4 garments: one fused render plus seam checks around cuffs, waist, hems, and shoe/pant overlap.
- 5 garments: one fused render plus accessory-specific depth/refinement pass if the accessory crosses torso/arms.

## 4. Speed Architecture

### 4.1 Bottlenecks That Appear At N Garments

| Bottleneck | Current behavior | Scaling risk |
|---|---|---|
| Base64 JSON payloads | Client/API pass large encoded strings | Request size and RAM grow with garment count |
| API image fetching | Node fetches all garment URLs and converts to data URI | Duplicated buffers and slow third-party CDN failures |
| FastAPI decode/resize | Decodes person once but garments per loop | CPU time and RAM grow with N |
| Prompt encoding | Per garment inference | Small but unnecessary repeated CLIP cost |
| Person VAE encoding | Recomputed every garment because `result` becomes next `person_image` | Wastes GPU time and compounds visual drift |
| Garment VAE encoding | Recomputed per garment | Necessary per garment, but should be batched/cached/streamed |
| IP-adapter image embedding | Recomputed per garment | Cacheable for catalog items |
| `denoiser_garment` | Runs inside every timestep for every full run | Major per-garment overhead |
| Main try-on denoiser | Full denoising loop per garment | Dominant latency; must be fused |
| VAE decode and PNG encode | Once per loop in current path | Should happen once for final output |
| Quality correction | Currently implicit by rerunning full canvas | Must become targeted crop refinement |

### 4.2 Concrete Speed Techniques And Expected Deltas

Assumption for this table: A10G warm container, 768x1024, 10 denoising steps, bf16, Mobile-VTON-class architecture, one output image, no cold start. Deltas are target reductions against the current sequential `/tryon/multi` implementation.

| Technique | Expected latency impact | Why it reduces latency | Default? |
|---|---:|---|---|
| Replace base64 JSON with object refs or multipart upload | -150ms to -600ms per request | Removes duplicate string encoding/decoding and lowers request parsing overhead | Yes |
| Shared person VAE latent and body masks | -180ms to -260ms per extra garment | Person conditioning is identical across garments | Yes |
| Cache garment cutout and segmentation by hash | -250ms to -900ms per cached garment | Catalog items repeat frequently; preprocessing should not be critical path | Yes |
| Cache garment text/image embeddings | -120ms to -450ms per cached garment | Image encoder and CLIP output are deterministic for a given garment/version | Yes |
| Batched garment image encoder | Additional garment cost drops from +180ms-300ms to +35ms-80ms | Micro-batch improves GPU utilization | Yes |
| Batched/streamed garment VAE encoding | Additional garment cost drops from +120ms-220ms to +40ms-90ms | Avoids repeated framework overhead and controls activation retention | Yes |
| One fused denoising trajectory | Saves ~1.35s-1.90s per avoided extra garment run | Main try-on denoiser runs once instead of N times | Yes |
| Region-gated cross-attention | Keeps active-slot overhead to +50ms-110ms per added garment | Pixels attend to relevant garment slots only | Yes |
| Conflict-only crop refinement | +80ms-220ms only when needed instead of +2s full rerun | Fixes seams/overlaps locally | Yes |
| Speculative preprocess on garment selection | Removes 300ms-1200ms from generate critical path | Work starts before user taps generate | Yes |
| 512x768 preview tier | First preview in 900ms-1600ms | Smaller latent grid and fewer steps | Yes |
| 6-8 step distilled/default tier | -350ms to -900ms versus 10 steps | Reduces denoising iterations | Month 4+ only |

Rejected alternatives:

- **Lower all renders to 512px:** rejected because quality fails on logos, fabric texture, seams, and catalog-grade detail.
- **Always run crop refinement:** rejected because it adds latency when masks and seams are already correct.

### 4.3 Target Latency Model

Assumption: A10G warm GPU, 768x1024, 10 steps, object refs, cached mannequin/person masks, uncached new garments unless stated. Sequential baseline is current architecture: N full inferences.

| Garments | Current sequential E2E target baseline | Target fused E2E | Required speedup | Key reason |
|---:|---:|---:|---:|---|
| 1 | 2.6s-3.2s | 2.4s-3.0s | Tie or slight win | Shared architecture should not regress single garment |
| 2 | 5.1s-6.1s | 2.8s-3.4s | 1.7x-2.1x | One denoising pass, batched garment conditioning |
| 3 | 7.6s-9.0s | 3.1s-3.9s | 2.1x-2.8x | Person and denoiser work are fixed |
| 4 | 10.1s-11.8s | 3.5s-4.5s | 2.5x-3.2x | Added cost is gated attention/refinement, not full reruns |
| 5 | 12.6s-14.6s | 4.0s-5.2s | 2.8x-3.4x | Accessory may trigger small crop refiner |

Production p95 targets at month 6 should be higher than p50 but still bounded:

- 2 garments: p95 <= 4.8s.
- 3 garments: p95 <= 5.6s.
- 4 garments: p95 <= 6.8s.
- 5 garments: p95 <= 8.5s.

### 4.4 Theoretical Minimum Latency Floors

Assumption: model-only lower bound on A10G, 768x1024, warm worker, no network, no PNG encode, cached garment embeddings, precomputed person condition, 10 steps, no crop refiner. This is not the user-visible production latency; it is the floor used to judge optimization headroom.

| Garments | Theoretical floor | Components that remain |
|---:|---:|---|
| 2 | 2.05s-2.25s | One denoising loop, two region-gated garment slots, one VAE decode |
| 3 | 2.25s-2.50s | One denoising loop, three slots, slightly more attention/conflict logic |
| 4 | 2.45s-2.80s | One denoising loop, four slots, more mask-gated attention |

If measured month-1 single-garment compute is much slower than 2.6s-3.2s E2E on A10G, keep the same relative targets: fused 4-garment p50 must be **less than 45% of 4 sequential runs**.

### 4.5 Caching Strategy

Use four cache layers:

1. **Mobile cache**
   - Cache selected garment thumbnails and upload handles.
   - Cache body/pose hints per captured photo.
   - TTL: current session plus 24 hours for user-approved assets.

2. **API cache**
   - Cache normalized request artifact metadata.
   - Avoid converting remote images to base64 if the VTON service can pull from object storage.

3. **Preprocess cache**
   - Key: `garment_hash + category + preprocessing_version`.
   - Stores cutout PNG/WebP, garment mask, crop box, material/opacity estimate.

4. **GPU embedding cache**
   - Key: `garment_hash + encoder_version + resolution_tier`.
   - Stores compact image embedding, text embedding, and optional pooled garment latent summary on CPU pinned memory or fast local disk.
   - Do not store full UNet activations across requests.

## 5. Memory Architecture

### 5.1 Where Memory Pressure Occurs

| Layer | Pressure source | Current risk |
|---|---|---|
| Mobile client | Full base64 mannequin and garment payloads | App memory spike and upload timeout |
| Express API | JSON body, fetched buffers, data URIs, repeated copies | RAM grows with garment count and image size |
| FastAPI service | PIL images, decoded base64, resized copies | CPU RAM grows with garment count |
| GPU weights | VAE, decoder, text encoders, image encoder, try-on UNet, garment UNet | Fixed high baseline |
| GPU activations | Denoising activations, attention tensors, garment feature maps | Explodes with naive batching |
| Condition tensors | Person latent, garment latents, masks, image embeddings | Should be bounded by compact bank |
| Output buffers | VAE decoded image, PNG encode buffer, response data URI | Avoid repeated encode/decode loops |

### 5.2 Required Memory Strategy

Default strategy: **constant-weight memory + sub-linear per-garment conditioning memory**.

Concrete rules:

- Keep one copy of model weights per GPU worker.
- Keep one `PersonCondition` per request.
- Store up to 5 garments in a fixed conditioning bank.
- Store masks as uint8 or bit-packed tensors; do not store float masks unless actively used.
- Stream garment VAE encoding into compact slot tensors; release raw garment tensors immediately.
- Batch image encoder work in micro-batches of 2-5 because embeddings are small.
- Do not retain full per-garment `denoiser_garment` activations across all steps.
- Use SDPA/xFormers/FlashAttention-compatible attention kernels where supported.
- Use `torch.inference_mode()` and explicit `del` for large tensors after every request.
- Call `torch.cuda.empty_cache()` only after OOM recovery or known fragmentation events; routine calls can hurt latency.
- Limit high-res worker concurrency to 1 request per 24GB GPU unless memory profiling proves safe.

Rejected alternatives:

- **Naively batch N complete single-garment pipelines:** rejected because activation memory grows nearly linearly and 4 garments can OOM a 24GB GPU.
- **CPU offload everything except active layer:** rejected for default path because PCIe transfers add unpredictable latency. Keep it as emergency fallback for 1024px/5-garment renders.

### 5.3 GPU VRAM Budget Table

Assumption: A10G 24GB, bf16/fp16, warm worker, one active request, shared person condition, fixed-size conditioning bank, region-gated attention, compact garment slot tensors, no full activation retention per garment. Values include model weights and activations.

| Resolution | Garments | Target peak VRAM | Technique |
|---|---:|---:|---|
| 512x768 preview | 1 | 8.8GB | Shared person, compact bank |
| 512x768 preview | 2 | 9.2GB | +1 active garment slot |
| 512x768 preview | 3 | 9.5GB | Region-gated attention |
| 512x768 preview | 4 | 9.8GB | Fixed slot bank |
| 512x768 preview | 5 | 10.2GB | Accessory slot active |
| 768x1024 default | 1 | 12.4GB | Shared person, no refiner |
| 768x1024 default | 2 | 13.0GB | Batched garment conditioning |
| 768x1024 default | 3 | 13.6GB | Compact slot bank |
| 768x1024 default | 4 | 14.3GB | Conflict masks active |
| 768x1024 default | 5 | 15.2GB | Accessory/depth conditioning |
| 1024x1024 high quality | 1 | 15.6GB | High-res latent grid |
| 1024x1024 high quality | 2 | 16.4GB | Shared person condition |
| 1024x1024 high quality | 3 | 17.1GB | Slot bank + gated attention |
| 1024x1024 high quality | 4 | 17.9GB | Optional crop refiner budget excluded |
| 1024x1024 high quality | 5 | 18.8GB | Requires no concurrent request on A10G |

Emergency fallback budgets:

- If peak VRAM exceeds 21GB on A10G, downgrade from 1024x1024 to 768x1024 before starting denoising.
- If peak VRAM exceeds 18GB at 768x1024, disable crop refiner and return fused output with seam warning telemetry.
- If peak VRAM exceeds 16GB on L4 for 5 garments, use 512x768 preview plus queued high-quality render.

### 5.4 Process RAM Budget Table

Assumption: object refs instead of base64 JSON, streaming decode, one request per worker, no repeated data URI conversion, decoded images capped at render tier.

| Resolution | Garments | Current base64-style RAM risk | Target RAM budget | Technique |
|---|---:|---:|---:|---|
| 512x768 | 2 | 450MB-700MB | <= 450MB | Object refs, stream decode |
| 512x768 | 5 | 900MB-1.4GB | <= 650MB | Bounded preprocess pool |
| 768x1024 | 2 | 650MB-950MB | <= 650MB | No duplicate data URI buffers |
| 768x1024 | 5 | 1.5GB-2.2GB | <= 900MB | Release raw PIL/buffers after tensorization |
| 1024x1024 | 2 | 900MB-1.3GB | <= 850MB | Decode directly to target size |
| 1024x1024 | 5 | 2.0GB-3.0GB | <= 1.2GB | Stream garments, no all-images-in-RAM invariant |

### 5.5 Garbage Collection And Cleanup Timing

Per request:

1. Decode one image at a time when possible.
2. Convert to tensor or cached artifact.
3. Delete raw buffers immediately after conversion.
4. Keep only compact conditions through denoising.
5. Decode final latent once.
6. Encode output once.
7. Delete request tensors before returning the response object.
8. Emit peak RAM and peak VRAM telemetry.

OOM recovery:

- Catch CUDA OOM at the service boundary.
- Clear references and call `torch.cuda.empty_cache()`.
- Retry once at the next lower resolution tier or fewer refinement steps.
- Return explicit degraded metadata; never silently return a lower-quality result as if it were full quality.

## 6. Occlusion And Composition Engine

### 6.1 Layering Order

Use metadata first, segmentation/depth second, fixed priors only as fallback.

Recommended z-prior order from back to front:

| Priority | Class | Examples | Notes |
|---:|---|---|---|
| 10 | Body/background locked | skin/mannequin/background | Frozen outside editable masks |
| 20 | Inner base | undershirt, tucked tee | Usually visible at collar/cuffs only |
| 30 | Top | shirt, sweater, blouse | Can be under outerwear or over waistband |
| 40 | Bottom | pants, skirt, shorts | Waist conflict depends on tuck policy |
| 45 | Dress/one-piece | dress, jumpsuit | Owns torso + lower body unless layered |
| 50 | Shoes | sneakers, boots, heels | Pants may overlap boot shaft |
| 60 | Outerwear | jacket, coat, cardigan | Covers top sleeves/torso; open-front can reveal top |
| 70 | Neck/waist accessories | scarf, belt, tie | Region-specific front layer |
| 80 | Cross-body accessories | bag strap, sash | Requires depth split across torso/arms |
| 90 | Hand-held accessories | purse, object | May be in front of hands/body |

Occlusion graph rules:

- `outerwear -> top` means outerwear is above top in overlapping torso/sleeve regions.
- `top -> bottom` at waist is conditional: untucked top above pants; tucked top below waistband/belt.
- `pants -> shoes` if pant hem covers shoe upper; `shoes -> pants` if high boots cover pants.
- `arm/body depth -> accessory` can split a single accessory mask into front and back segments.
- Transparent garments create blend edges rather than hard z edges.

### 6.2 Region Assignment

For each garment, compute:

- Source garment segmentation: garment pixels, holes, transparent regions, logos/details.
- Target body regions: torso, left/right arm, hips, left/right leg, feet, neck, shoulder, waist.
- Fit constraints: sleeve length, hem length, waist height, collar opening, shoe footprint.
- Conflict mask: where two or more garment region masks overlap.

Default region mapping:

| Garment | Target regions | Conflict zones |
|---|---|---|
| Top | torso, shoulders, upper arms, optional waist | collar, cuffs, hem, waistband |
| Bottom | hips, thighs, knees, calves, waist | waistband, top hem, shoe boundary |
| Outerwear | shoulders, torso, arms, optional hips | top sleeves, collar, open front, cuffs |
| Shoes | feet, ankles | pant hem, socks/skin, floor contact |
| Belt | waist contour | top hem and pants waistband |
| Scarf | neck, upper chest, shoulders | outerwear collar, hair/face if real person |
| Bag strap | shoulder-to-hip path | arm crossing, jacket lapel, torso |

### 6.3 Boundary Blending Without Artifacts

Use a three-zone trimap per garment boundary:

- **Interior:** hard garment ownership; preserve product texture and logos.
- **Boundary band:** 8-24px adaptive blend based on resolution and edge confidence.
- **Exterior:** frozen original or lower-priority garment.

Composition steps:

1. Refine alpha matte around garment cutout and warped target mask.
2. Use hard ownership inside the confident mask.
3. Use Laplacian pyramid blending only in the trimap band.
4. Add contact shadows under hems, collars, cuffs, waistband, shoe sole, and outerwear overlap.
5. Run seam detector over boundary band.
6. If seam score fails, run crop-level refiner on the failing boundary only.

Do not use uniform feathering across the whole mask. Uniform feathering causes logos, stripes, garment edges, and cuffs to look washed out.

### 6.4 Edge Cases

| Edge case | Handling |
|---|---|
| Jacket over shirt | Outerwear owns sleeve/torso overlap; shirt visible at collar, open front, cuffs, and hem if geometry supports it |
| Long coat over pants | Coat region extends into hips/thighs; pants remain visible below coat mask and at openings |
| Tucked shirt | Waist graph sets pants/belt above top hem; top texture terminates under waistband |
| Untucked shirt | Top hem above pants; add soft shadow at waist overlap |
| Transparent shirt | Render underlayer first, then transparent layer with alpha/material token and color attenuation |
| Mesh/lace garment | Preserve high-frequency pattern with stronger product-fidelity loss and limited blur at boundary |
| Cross-body bag | Split strap by depth: behind arm where arm is foreground, above torso where torso is background |
| Scarf over coat | Scarf above coat near neck/chest; coat remains above top sleeves |
| Boots with pants | Boot shaft may cover pant hem; graph depends on detected boot height |
| Accessories crossing multiple regions | Decompose into sub-masks per body region and assign independent z edges |

## 7. Six-Month Roadmap Summary Table

| Month | Engineering focus | Research tasks | Benchmark targets | Go / No-Go gate | Top risks and mitigations | Team allocation |
|---|---|---|---|---|---|---|
| 1 | Instrument current pipeline, build eval set, define artifact schema | Validate baseline bottlenecks and objective quality metrics | Baseline current 1-4 garment latency/RAM/VRAM; 300-case eval set; stage timing coverage >=95% | Proceed only if current sequential baseline and eval suite are reproducible | Risk: subjective quality only. Mitigation: locked metric suite + blind review. Risk: telemetry too coarse. Mitigation: stage-level timers and CUDA peaks | ML/CV 35%, backend 25%, mobile 15%, MLOps 15%, QA/eval 10% |
| 2 | Prototype shared person condition and garment cache/bank | Validate fused conditioning does not lose single-garment fidelity | 2 garments p50 <=3.8s; VRAM <=14GB at 768x1024; outside-mask SSIM >=0.985 | Proceed only if 2-garment fused prototype beats 2 sequential runs by >=30% | Risk: garment identity loss. Mitigation: product-fidelity loss/metrics. Risk: cache invalidation bugs. Mitigation: versioned keys | ML/CV 45%, backend 20%, MLOps 15%, mobile 10%, QA/eval 10% |
| 3 | Build occlusion graph, region masks, batched warping | Validate layering and body-region assignment across top/bottom/outerwear/shoes | 3 garments p50 <=4.2s; z-order accuracy >=92%; region spill <=5%; VRAM <=15GB | Proceed only if top+bottom+outerwear passes blind review tie/win >=55% | Risk: wrong z-order plausible failures. Mitigation: graph validators. Risk: mask seams. Mitigation: trimap refiner | ML/CV 40%, diffusion 20%, backend 15%, QA/eval 15%, mobile 10% |
| 4 | Add fabric interaction, seam/crop refiner, 5-slot accessory support, step optimization | Validate transparent/cross-body accessories and 6-8 step distilled tier | 4 garments p50 <=4.8s; 5 garments p50 <=5.8s; seam severe artifact <=3%; VRAM <=18.8GB at 1024 | Proceed only if crop refiner fixes seams without full reruns | Risk: refiner changes identity. Mitigation: hard frozen mask. Risk: accessory complexity. Mitigation: accessory beta gate | Diffusion 35%, ML/CV 30%, backend 15%, QA/eval 15%, mobile 5% |
| 5 | Production integration, fallback policy, observability, mobile cache warming, beta rollout | Validate real-device latency and production failure recovery | 3 garments p95 <=5.6s; 4 garments p95 <=6.8s; OOM recovery success >=98%; crash rate <0.1% | Proceed only if 5% beta has stable p95 and no Sev1 quality class | Risk: infra cold starts. Mitigation: warm pool/autoscaling. Risk: network payload spikes. Mitigation: object refs | Backend 25%, MLOps 25%, mobile 20%, ML/CV 20%, QA/eval 10% |
| 6 | Scale, harden, launch, governance for model updates | Validate regression gates and cost/throughput | 4 garments p50 <=4.5s, p95 <=6.8s; 5 garments p95 <=8.5s; cost/render target met; zero critical regressions | Launch only if metrics pass for 2 consecutive weeks | Risk: model update regressions. Mitigation: blocking eval. Risk: GPU cost. Mitigation: tier routing/cache hits | MLOps 30%, backend 25%, QA/eval 20%, ML/CV 15%, mobile 10% |

## 8. Month-by-Month Deep Dive

### Month 1: Measurement, Evaluation, And Architecture Contract

Engineering focus:

- Add stage-level benchmarking around current `/api/tryon/mobile-vton` and `/tryon/multi` path.
- Record timing for image fetch, API request parsing, FastAPI decode, resize, prompt encode, person encode, garment encode, denoising loop, VAE decode, PNG encode, and response serialization.
- Record peak CPU RAM and CUDA VRAM per request.
- Define a v2 request/response schema that supports object refs, garment metadata, active slots, quality tier, and returned diagnostics.
- Build an evaluation harness that can run the same garment/person set through current sequential and future fused pipelines.

Research tasks:

- Validate which current model modules are safe to cache: prompt embeddings, image embeddings, garment cutouts, person latents.
- Measure whether current `denoiser_garment` features can be batched or need architectural changes.
- Establish visual metrics that correlate with human review for this product: outside-mask preservation, product identity, seam artifacts, z-order correctness.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Baseline trace report for 1, 2, 3, and 4 garments on A10G/L4-equivalent workers |
| 2 | Evaluation dataset v1: 300 cases, including mannequin and real-person captures if available |
| 3 | Artifact schema and cache key spec for person/garment conditions |
| 4 | Automated benchmark runner with pass/fail thresholds and dashboard |

Benchmark targets:

- Current sequential p50/p95 measured for 1-4 garments at 768x1024.
- CUDA peak memory measured for single and multi-garment sequential path.
- Eval harness reproducibility: repeated same-seed score variance <=2% for deterministic metrics.
- Body preservation metric implemented: outside editable mask SSIM target >=0.985 for accepted renders.
- Product color metric implemented: garment crop DeltaE2000 target <=6 for non-transparent garments.

Go / No-Go gate:

- Go if the team can reproduce current latency, memory, and quality numbers on a fixed dataset and fixed GPU class.
- No-Go if quality decisions still rely only on ad hoc visual inspection.

Risk log:

- **Risk:** current subjective quality hides silent failures.
  - **Mitigation:** require metric reports and blind review labels before accepting future model changes.
- **Risk:** benchmark harness diverges from production request path.
  - **Mitigation:** benchmark through the same API/service path plus a model-only harness.

Team allocation:

- ML/CV: 35% for metric design and model instrumentation.
- Backend: 25% for API tracing and artifact schema.
- Mobile: 15% for payload and cache-warming requirements.
- MLOps: 15% for GPU telemetry and benchmark automation.
- QA/eval: 10% for dataset labeling and review rubric.

### Month 2: Shared Person Encoding And Garment Conditioning Prototype

Engineering focus:

- Implement `PersonCondition` extraction once per request.
- Implement `GarmentCondition` cache for cutouts, text embeddings, image embeddings, and compact visual tokens.
- Build a prototype `/tryon/multi-fused` path behind a feature flag.
- Replace repeated full person encoding with shared person latent and shared body masks.
- Start with 2 garments: top + bottom and top + outerwear.

Research tasks:

- Determine whether garment-specific UNet features can be computed in a batched call per timestep or approximated by cached compact features.
- Test region-gated conditioning so top pixels do not attend to shoe features and leg pixels do not attend to shirt details.
- Validate that single-garment quality does not regress under the new condition bank.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Shared `PersonCondition` extraction prototype with telemetry |
| 2 | Garment embedding/cache service with versioned keys and hit/miss metrics |
| 3 | Two-garment fused conditioning prototype, no advanced occlusion yet |
| 4 | A/B report: current sequential vs fused two-garment prototype |

Benchmark targets:

Assumption: 768x1024, A10G, warm worker, 10 steps.

- 2 garments p50 <=3.8s end-to-end.
- 2 garments peak VRAM <=14GB.
- Cache hit removes >=300ms from repeated catalog garment requests.
- Single-garment output under new path ties current single-garment output in >=60% blind review.
- Outside-mask SSIM >=0.985.

Go / No-Go gate:

- Go if 2-garment fused path is at least 30% faster than 2 sequential runs and does not materially degrade garment identity.
- No-Go if the fused condition bank causes frequent garment mixing or product identity loss.

Risk log:

- **Risk:** conditioning bank blends garment identities.
  - **Mitigation:** add slot/category embeddings, region-gated attention, and per-slot product-fidelity checks.
- **Risk:** cache keys become invalid after model updates.
  - **Mitigation:** include encoder/model/preprocess versions in every cache key.

Team allocation:

- ML/CV: 45%.
- Backend: 20%.
- MLOps: 15%.
- Mobile: 10%.
- QA/eval: 10%.

### Month 3: Occlusion Graph, Region Assignment, And Batched Warping

Engineering focus:

- Build the production occlusion graph with metadata priors and segmentation/depth overrides.
- Implement body-region assignment for top, bottom, outerwear, and shoes.
- Add batched warping using shared person pose/body features.
- Add z-order validation and conflict mask outputs.
- Support 3 garments as the default development target.

Research tasks:

- Validate rule-based z-priors against learned depth/segmentation results.
- Measure when outerwear should expose or hide top regions.
- Evaluate tucked/untucked waistband interactions.
- Determine confidence thresholds for automatic recovery when region assignment is uncertain.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Body-region mask generator and garment-to-region assignment report |
| 2 | Occlusion graph resolver with cycle detection and explainable z-order output |
| 3 | Batched warp fields for top/bottom/outerwear/shoes |
| 4 | Three-garment quality benchmark: top + bottom + outerwear |

Benchmark targets:

Assumption: 768x1024, A10G, warm worker, 10 steps, 3 garments.

- 3 garments p50 <=4.2s.
- Peak VRAM <=15GB.
- Region assignment IoU >=0.90 for torso/legs/feet on benchmark masks.
- Z-order pairwise accuracy >=92% on labeled overlap cases.
- Region spillover <=5% outside expected body zones.
- Blind review tie/win against manual single-garment composite >=55%.

Go / No-Go gate:

- Go if 3-garment outputs have correct layer order in at least 92% of labeled overlaps and p50 latency is less than 55% of current 3 sequential runs.
- No-Go if plausible-looking wrong-layer outputs remain common and undetected.

Risk log:

- **Risk:** wrong z-order can look plausible and pass casual review.
  - **Mitigation:** add pairwise z-order classifier and graph explanation logs.
- **Risk:** warping errors cause boundary seams.
  - **Mitigation:** route low-confidence warp boundaries into Month 4 seam refiner.

Team allocation:

- ML/CV: 40%.
- Diffusion/model engineering: 20%.
- Backend: 15%.
- QA/eval: 15%.
- Mobile: 10%.

### Month 4: Fabric Interaction, Seam Refinement, And 5-Slot Support

Engineering focus:

- Add crop-level seam and occlusion refiner.
- Add contact shadow generation and boundary trimap refinement.
- Add support for one accessory slot: belt, scarf, bag, tie, or simple jewelry.
- Add transparent/semi-transparent garment handling.
- Optimize denoising step count through distillation or validated 6-8 step quality tier.

Research tasks:

- Validate local crop refiner against full rerun quality.
- Test transparent fabrics with underlayer preservation.
- Determine when accessories need depth splitting versus simple top-layer composition.
- Validate whether 6-8 steps can meet product fidelity thresholds.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Seam detector and trimap-based local refiner |
| 2 | Contact shadow and fabric overlap rules for cuffs, collars, hems, waistband, shoes |
| 3 | Transparent material and accessory slot prototype |
| 4 | Four- and five-garment benchmark with quality/latency report |

Benchmark targets:

Assumption: A10G, warm worker, default 768x1024; high-quality 1024x1024 measured separately.

- 4 garments p50 <=4.8s at 768x1024.
- 5 garments p50 <=5.8s at 768x1024.
- 1024x1024 5-garment peak VRAM <=18.8GB before optional crop refiner.
- Severe seam artifact rate <=3% on benchmark set.
- Transparent garment opacity correctness >=85% on labeled transparent/semi-transparent cases.
- Crop refiner changes frozen outside-mask pixels by <=1% of boundary-adjacent area.

Go / No-Go gate:

- Go if crop-level refiner fixes seams without full-canvas reruns and without identity/background drift.
- No-Go if fabric interactions require repeated full-frame diffusion passes to look acceptable.

Risk log:

- **Risk:** crop refiner introduces local body or background drift.
  - **Mitigation:** hard frozen masks and pixel snap outside crop trimap.
- **Risk:** accessories explode scope.
  - **Mitigation:** launch accessory support as one slot with strict supported classes and fallback messaging.

Team allocation:

- Diffusion/model engineering: 35%.
- ML/CV: 30%.
- Backend: 15%.
- QA/eval: 15%.
- Mobile: 5%.

### Month 5: Production Integration, Reliability, And Beta Rollout

Engineering focus:

- Integrate fused path behind server-side and client-side feature flags.
- Add fallback routing: fused default, sequential legacy fallback, lower-resolution retry, queued high-quality render.
- Add mobile cache warming when users select garments.
- Replace large data URI request path with object refs for production VTON artifacts.
- Add production observability for latency, memory, cache hit rate, failure category, and quality detectors.

Research tasks:

- Validate production traffic patterns and cache hit rates.
- Tune resolution tier routing by device class, network quality, garment count, and subscription tier.
- Validate failure detector precision/recall to avoid blocking good renders or shipping bad renders.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Feature-flagged API route and mobile request integration |
| 2 | Object-ref artifact path, cache warming, and response diagnostics |
| 3 | OOM/degraded-retry policy and autoscaling warm-pool config |
| 4 | 5% beta rollout with daily quality review and rollback playbook |

Benchmark targets:

Assumption: production-like mobile network, warm GPU pool, 768x1024 default.

- 3 garments p95 <=5.6s.
- 4 garments p95 <=6.8s.
- 5 garments p95 <=8.5s or queued high-quality fallback.
- OOM recovery success >=98%.
- Crash rate <0.1% of try-on requests.
- Cache hit rate >=50% for catalog garments after warm-up.
- User-visible generic failure rate <=2%.

Go / No-Go gate:

- Go if 5% beta holds p95 latency and failure targets for one full week without Sev1 visual regressions.
- No-Go if production memory or cold-start behavior invalidates lab benchmarks.

Risk log:

- **Risk:** GPU cold starts dominate p95.
  - **Mitigation:** warm pool, min containers, health probes, and queue admission control.
- **Risk:** network/upload path dominates latency.
  - **Mitigation:** object refs, mobile compression, and background upload/cache warming.

Team allocation:

- Backend: 25%.
- MLOps: 25%.
- Mobile: 20%.
- ML/CV: 20%.
- QA/eval: 10%.

### Month 6: Scale, Hardening, Cost Control, And Launch

Engineering focus:

- Lock model update governance with blocking regression tests.
- Optimize cost per render through cache hit improvements, GPU routing, and optional lower-step default tier.
- Harden autoscaling and admission control for peak demand.
- Launch full multi-garment support with explicit supported categories and fallbacks.
- Build post-launch monitoring for silent quality failures.

Research tasks:

- Validate whether L4 can serve lower-tier requests cost-effectively while A10G/A100 serve high-quality requests.
- Validate which categories should remain beta: transparent garments, complex accessories, unusual poses.
- Tune model distillation/step reduction against real user preference data.

Week-level plan:

| Week | Concrete output |
|---|---|
| 1 | Full regression suite blocking service/model updates |
| 2 | Cost and throughput optimization report with GPU tier routing |
| 3 | Launch candidate burn-in with 2 consecutive clean benchmark runs |
| 4 | Production launch and post-launch monitoring playbook |

Benchmark targets:

Assumption: production warm pool, default 768x1024, 10 steps or validated 6-8 step distilled tier.

- 4 garments p50 <=4.5s and p95 <=6.8s.
- 5 garments p95 <=8.5s or async high-quality fallback.
- Peak VRAM at 768x1024 5 garments <=15.2GB in default path.
- Severe artifact rate <=2% on nightly benchmark.
- Human blind review tie/win vs manual composite >=60%.
- Zero critical regressions for 2 consecutive weeks before launch.

Go / No-Go gate:

- Launch only if latency, memory, quality, OOM recovery, and regression gates pass for 2 consecutive weeks.
- Do not launch transparent/cross-body accessory support broadly unless its class-specific gates pass; keep it beta if needed.

Risk log:

- **Risk:** model updates regress quality silently.
  - **Mitigation:** blocking benchmark suite, canary traffic, and automatic rollback.
- **Risk:** GPU cost exceeds product margin.
  - **Mitigation:** tiered rendering, cache warming, preview/final split, GPU type routing, and per-user quota controls.

Team allocation:

- MLOps: 30%.
- Backend: 25%.
- QA/eval: 20%.
- ML/CV: 15%.
- Mobile: 10%.

## 9. Tradeoff Map

| Decision | Speed gain | Quality gain/loss | Memory gain/loss | Breaks down at | Recommended default | Override when |
|---|---|---|---|---|---|---|
| Single fused diffusion pass instead of N sequential runs | Major: removes N-1 full denoising loops | Gain for interactions; risk of garment mixing | Major gain versus parallel full runs | Very complex 5+ garments with accessories | Default for all multi-garment | Use legacy sequential only as fallback/debug |
| Fixed 5-slot conditioning bank | Stable graph and bounded memory | Slight risk if >5 garments requested | Prevents unbounded per-garment memory | More than 5 garments | Max 5 active slots | Queue separate outfit render for extras |
| Region-gated attention | Keeps per-garment overhead low | Improves assignment; may miss cross-region accessories | Lower attention memory | Accessories crossing many regions | Default | Disable gating for verified cross-body crop only |
| Batched garment encoder | Reduces per-garment encoder overhead | Neutral | Slight temporary VRAM increase | Low-VRAM workers or 1024x1024 5 garments | Micro-batch 2-5 | Stream one garment on L4/high-res fallback |
| Cache garment embeddings | Removes repeat work | Risk stale features after model update | Uses disk/RAM but saves GPU work | Massive catalog without eviction | Versioned cache with LRU | Recompute on version mismatch |
| Object refs instead of base64 JSON | Faster parse/upload; fewer copies | Neutral | Large RAM reduction | Offline-only local testing | Default production path | Accept base64 only as compatibility path |
| 512x768 preview + 768x1024 final | Faster perceived UX | Preview less detailed | Lower preview memory | Premium user expects immediate final only | Default UX | Skip preview for very fast cached render |
| 6-8 step distilled tier | Faster denoising | Possible texture/detail loss | Lower activation duration, same peak mostly | Logos/fine patterns/transparent fabrics | Month 4+ after validation | Use 10-14 steps for premium/high-detail garments |
| Crop seam refiner | Fixes local artifacts cheaply | Can improve seams; risk local inconsistency | Small temporary memory | Many conflict regions covering large area | Run only on failed seam detector | Full rerender only if crop area >35% canvas |
| Laplacian boundary blend | Reduces seams | Can blur fine edges if overused | Low memory | Logos/stripes at boundary | Boundary trimap only | Use hard edge for crisp logos/cuffs |
| CPU offload encoders | Enables low VRAM workers | Slower due transfers | Lower VRAM, higher RAM | Latency-sensitive default path | Emergency fallback only | 1024px 5-garment OOM risk |
| On-device pose/preprocess hints | Reduces server work and upload retries | Depends on device model quality | Saves server RAM/compute | Older devices or bad lighting | Optional hints, not required | Ignore hints if confidence low |

## 10. Failure Mode Analysis

### 10.1 Silent Failures

| Failure | Why dangerous | Detection | Recovery |
|---|---|---|---|
| Garment assigned to wrong body region | Output may look plausible but wrong | Region classifier + mask overlap check | Re-run with corrected category or ask user to relabel |
| Wrong layering order | Jacket under shirt can look stylized, not obviously broken | Pairwise z-order classifier on conflict masks | Rebuild occlusion graph and crop-refine overlap |
| Inner garment disappears | Fused model over-prioritizes outerwear | Expected-visible-region check for collar/cuff/hem | Increase inner garment visibility constraints and rerun crop |
| Product color drift | Render looks realistic but not the purchased item | DeltaE2000 and palette histogram comparison | Color correction in garment-owned regions or rerun high-fidelity tier |
| Logo/pattern hallucination | Bad for e-commerce trust | DINO/CLIP crop similarity, OCR/logo detector | Rerun with stronger product-fidelity conditioning or block result |
| Body pose drift | Garment looks fine but person/mannequin changed | Keypoint drift, outside-mask SSIM/LPIPS | Hard snap outside editable mask; rerun if inside drift affects body |
| Background/shadow drift | Looks subtle, violates product consistency | Background SSIM and white balance check | Restore background from original and restrict shadow region |
| Seam hidden by over-blending | Plausible blurry garment boundary | Boundary sharpness and gradient discontinuity metric | Recompose with narrower trimap or crop refiner |
| Transparent garment rendered opaque | Common plausible failure | Opacity/material classifier + underlayer visibility check | Rerun transparent material path |
| Accessory depth wrong | Bag strap over arm when it should pass behind | Depth-order classifier on accessory/body crossing | Split accessory mask by depth and crop-refine crossing |
| Garment scale wrong | Looks like fashion choice but product fit is wrong | Anchor/landmark ratio checks | Recompute warp with corrected scale prior |
| Category metadata wrong from catalog | Pipeline follows wrong instructions | Category confidence and image classifier mismatch | Override metadata with vision classifier or ask user |
| One garment contaminates another region | Example: shirt texture appears on pants | Per-region product similarity check | Region-gated rerun with stricter masks |
| Lower-res fallback shipped silently | User sees degraded image without explanation | Response metadata and telemetry consistency check | Surface queued HQ render or explicit degraded result flag |

### 10.2 Catastrophic Failures

| Failure | Detection | Recovery |
|---|---|---|
| CUDA OOM | Catch `torch.cuda.OutOfMemoryError`, GPU telemetry | Clear tensors, empty cache, retry lower resolution once, then fallback/queue |
| CPU RAM OOM from payloads | Process memory watchdog, request size guard | Reject oversize base64, require object refs, stream decode |
| FastAPI worker stuck | Request timeout and health probe failure | Kill/restart worker, mark GPU unhealthy, route traffic away |
| Infinite retry loop | Retry counter in request context | Max one degraded retry and one fallback path |
| Cold start exceeds client timeout | Startup telemetry and queue wait time | Warm pool, readiness gate, async job fallback |
| Corrupt image/cache artifact | Decode exceptions, checksum mismatch | Evict cache entry and refetch/reprocess once |
| NaN/invalid latents | Tensor finite checks after denoising steps | Abort render, retry with safe seed/resolution, quarantine request sample |
| Occlusion graph cycle | Topological sort failure | Break tie with deterministic z-prior and log validation error |
| Response image too large | Encoded size check | Use WebP/JPEG where acceptable or lower resolution |
| Third-party CDN timeout | Fetch timeout/error category | Use stored catalog asset mirror; fail fast if unavailable |
| GPU concurrency overload | Queue depth and VRAM admission control | Enforce one high-res request/GPU; autoscale before saturation |
| Model load failure | Health endpoint reports loading/unhealthy | Keep old model warm; rollback deployment |
| Artifact storage outage | Upload/download error rates | Local temporary fallback for current request or degrade to legacy path |
| Security issue via malicious image | MIME sniffing, pixel count limits, decoder sandbox | Reject unsafe files and log abuse signal |

## 11. Benchmark And Evaluation Framework

### 11.1 Test Suite

Build three tiers.

**Tier S0: current mannequin regression**

- Fixed mannequin/front-facing catalog flow currently used by the app.
- Categories: top, pants, outerwear/layer, shoes.
- Combinations: 1, 2, 3, 4 garments.
- Purpose: prevent regressions in the existing production experience.

**Tier S1: real-user mobile captures**

- Body diversity: size, height, skin tone, gender presentation, hair length, sleeves/legs visibility.
- Pose diversity: front neutral, slight side, arms crossed, hand on hip, sitting, walking stance, occluded hands, wide stance.
- Lighting/background: indoor warm, outdoor, cluttered room, mirror selfie, white wall.
- Purpose: validate production mobile VTON beyond mannequin assumptions.

**Tier S2: hard cases**

- Transparent mesh/lace tops.
- Long coats over pants.
- Open jackets exposing shirts.
- Boots under/over pants.
- Belts over tucked and untucked tops.
- Scarves over coats.
- Cross-body bags crossing arms and torso.
- Logos, stripes, plaid, text prints.
- White garments on white background and black garments on dark pants.

Minimum dataset by month 6:

- 300 S0 cases.
- 500 S1 cases.
- 250 S2 hard cases.
- At least 100 cases with 4 garments.
- At least 50 cases with 5 garments or accessory slot.

### 11.2 Metrics And Thresholds

| Metric | Target threshold | Applies to | Why it matters |
|---|---:|---|---|
| End-to-end p50 latency | <=4.5s for 4 garments | Production default | Must beat sequential and feel interactive |
| End-to-end p95 latency | <=6.8s for 4 garments | Production default | Prevents bad mobile UX |
| Peak VRAM | <=15.2GB for 768x1024 5 garments | A10G default | Leaves safety margin on 24GB GPU |
| Peak process RAM | <=900MB for 768x1024 5 garments | API/service | Prevents worker instability |
| Outside-mask SSIM | >=0.985 | Body/background preservation | Prevents identity/background drift |
| Outside-mask LPIPS | <=0.025 | Body/background preservation | Catches perceptual drift |
| Keypoint drift | <=4px at 768x1024 | Person/mannequin geometry | Prevents pose/body movement |
| Region assignment IoU | >=0.90 core categories | Garment placement | Prevents wrong body region |
| Region spillover | <=5% | Garment masks | Prevents contamination across zones |
| Pairwise z-order accuracy | >=95% by launch | Overlap regions | Prevents plausible wrong layering |
| Product DINO similarity | >=0.72 | Garment fidelity | Preserves product identity |
| Product CLIP similarity | >=0.30 absolute or non-regression | Garment fidelity | Catches semantic mismatch |
| Color DeltaE2000 | <=6 for main garment colors | Product fidelity | Prevents color drift |
| Logo/text preservation | >=90% visible-logo pass | Branded garments | E-commerce trust |
| Severe seam artifact rate | <=2% by launch | Boundary quality | Prevents visible composites |
| Human blind review | >=60% tie/win vs manual composite | Overall quality | Final quality gate |
| OOM recovery success | >=98% | Reliability | Prevents crashes |
| Generic failure rate | <=2% | User experience | Keeps feature trustworthy |

### 11.3 Regression Tests For Every Model Update

Every model/service update must run:

1. **PR smoke suite**
   - 40 cases total.
   - Includes 10 single-garment, 15 two/three-garment, 10 four-garment, 5 hard cases.
   - Must finish in <=20 minutes on CI GPU runner or scheduled benchmark worker.

2. **Nightly full suite**
   - Full S0/S1/S2 dataset.
   - Reports latency, memory, quality metrics, cache hit rates, and failure categories.

3. **Canary production suite**
   - 1%-5% production traffic behind feature flag.
   - Automatic rollback if severe artifact detector, OOM, or p95 latency crosses threshold.

Blocking rules:

- Block if outside-mask SSIM drops below 0.985.
- Block if pairwise z-order accuracy drops by more than 2 percentage points.
- Block if p95 latency regresses by more than 15% for any garment count tier.
- Block if peak VRAM regresses by more than 1GB at 768x1024.
- Block if human review severe artifact rate exceeds 3% on the sampled hard set.

## 12. Production API And Service Migration

Recommended migration sequence:

1. Keep `/api/tryon/mobile-vton` stable for the mobile client.
2. Add internal request versioning: `pipeline_version: 'sequential_v1' | 'fused_v2'`.
3. Add a new FastAPI endpoint `/tryon/multi-fused` while keeping `/tryon/multi` as fallback.
4. Change Node strategy to route by feature flag and garment count.
5. Add response diagnostics:
   - `pipelineVersion`
   - `resolutionTier`
   - `renderedGarments`
   - `cacheHits`
   - `peakVramMb`
   - `peakRamMb`
   - `qualityWarnings`
   - `degraded`
6. Move from data URI request payloads to object references.
7. Keep base64 compatibility only for development and small test images.

Default routing at launch:

| Request | Route |
|---|---|
| 1 garment | Fused path if non-regressed; otherwise existing single path |
| 2-4 garments | Fused path default |
| 5 garments without complex accessory | Fused path with seam/depth checks |
| 5 garments with complex accessory | Fused path beta or queued HQ render |
| OOM/high load | Lower resolution retry or async queue |
| Quality detector fail | Crop refiner, then fallback/explicit failure |

## 13. Final Recommendation

The winning architecture is not "run Mobile-VTON N times faster." It is a new multi-garment graph that treats the person, body regions, and output denoising trajectory as shared, while treating garments as compact, region-gated conditioning slots. This is the only approach that satisfies all three hard constraints simultaneously:

- **Speed:** one denoising trajectory instead of N full trajectories.
- **Memory:** fixed model weights plus compact slot bank, not N full activation stacks.
- **Quality:** global garment interactions and occlusion graph, not manual post-hoc compositing.

If I could only do 3 things in Month 1 to maximize long-term success, I would do X, Y, Z — because X = stage-level benchmarking exposes the real latency and memory bottlenecks, Y = building the multi-garment evaluation set turns quality into a repeatable gate, and Z = prototyping the shared person/garment condition cache proves whether the system can beat N sequential single-garment runs without sacrificing product fidelity.
