# Multi-Garment VTON — Implementation Progress

## Month 1-2 Foundation (Implemented)

### 1. Instrumentation & Telemetry
- **File**: `mobile-vton-service/telemetry.py`
- Stage-level wall-clock timers per pipeline phase
- Peak GPU VRAM tracking via `torch.cuda`
- Peak process RAM tracking via `psutil`
- Cache hit/miss counters
- Quality warning and degraded-result flags
- Request-level diagnostics output

### 2. Shared Person + Garment Condition Cache
- **File**: `mobile-vton-service/condition_cache.py`
- `PersonCondition` dataclass: person tensor, VAE latent, prompt embeddings (all 4 CLIP variants)
- `GarmentCondition` dataclass: cloth_pure, cloth_trim, cloth_latent, text embeddings
- LRU in-memory caches with bounded capacity (32 person, 128 garment)
- Image-hash-based cache keys including resolution and dtype

### 3. Occlusion Graph Engine
- **File**: `mobile-vton-service/occlusion_graph.py`
- 10-layer z-priority system (body → inner → top → bottom → dress → shoes → outerwear → accessories)
- Label-to-layer mapping for common garment types
- Body-region assignment (torso, arms, legs, feet, neck, waist)
- DAG construction with edge rules for tuck policy, shoes/pants overlap, transparent materials
- Kahn topological sort with cycle detection and deterministic fallback

### 4. Fused Pipeline Wrapper (Phase 1)
- **File**: `mobile-vton-service/fused_pipeline.py`
- `FusedMobileVTON` class wrapping existing `MobileVTONPipeline`
- Shared person text encoding (cached once per request)
- Per-garment condition extraction with cache hit tracking
- Occlusion-graph-aware render ordering
- Sequential denoising with **pre-computed embeddings** passed to existing pipeline
- Returns full diagnostics: timing, cache, occlusion graph, VRAM

### 5. New FastAPI Endpoint
- **File**: `mobile-vton-service/main.py`
- `POST /tryon/multi-fused` — accepts same payload as `/tryon/multi` plus `pipeline_version`
- Decodes all garment images upfront before passing to fused pipeline
- Returns `pipeline_version`, `diagnostics` in response
- Added `/metrics` (cache stats, CUDA info) and `/admin/clear-cache`
- Instrumented existing `/tryon` and `/tryon/multi` with `reset_cuda_peak_memory()`

### 6. Node API Routing
- **File**: `api/services/mobileVtonClient.js`
- `callMobileVtonMultiFused()` client for new endpoint
- All clients return `pipelineVersion` and `diagnostics`
- **File**: `api/services/strategies/mobileVton.js`
- `pipeline_version` parameter (default `sequential_v1`)
- Routes multi-garment to `fused_v2` when `pipeline_version === 'fused_v2'`
- Includes diagnostics in response

### 7. Mobile Client Update
- **File**: `features/try-on/AITryOnScreen.tsx`
- Sends `pipeline_version: 'fused_v2'` in try-on request body
- Receives and can display diagnostics if wired to UI

### 8. Benchmark Runner
- **File**: `scripts/benchmarkVton.js`
- Compares `sequential_v1` vs `fused_v2` on same outfit
- Reports p50/p95/p99 latency, cache hit/miss, peak VRAM
- Warm-up trials + configurable iteration count

## Month 3: Occlusion Graph, Region Assignment, Batched Warping (Implemented)

### 9. Batched Warp Field Module
- **File**: `mobile-vton-service/warp_fields.py`
- `WarpedGarment` dataclass: warped latent, alpha mask, confidence map, region mask, warp grid
- Category-specific affine transforms with aspect-ratio-aware target boxes
- `F.grid_sample` for differentiable warping of garment latents into body-aligned space
- Confidence map based on garment AR match with target body region
- `batched_warp_garments()` for parallel warp processing
- `assemble_conditioning_bank()` to pack up to 5 garments into fixed-size tensors

### 10. Three-Garment Fixed Benchmark Runner
- **File**: `scripts/runThreeGarmentBenchmark.py`
- Runs sequential vs fused on a fixed 3-garment outfit (top + bottom + outerwear)
- Calls `/evaluate` endpoint for automated SSIM comparison
- Generates markdown report with pass/fail against Month 3 targets:
  - p50 latency <= 4.2s
  - Peak VRAM <= 15GB
  - Outside-mask SSIM >= 0.985
  - Speedup >= 1.82x (55% of 3 sequential)
- Outputs JSON + markdown to configurable output directory

### 11. Occlusion Graph Automated Test Suite
- **File**: `mobile-vton-service/test_occlusion_graph.py`
- `TestBasicLayering`: top over bottom, outerwear over top, shoes under pants
- `TestTuckPolicy`: tucked tops render below pants waistband
- `TestEdgeCases`: belts over pants, scarves over coats, bags top-most
- `TestZOrderAccuracy`: pairwise accuracy test on 10 labeled cases, target >= 92%
- Run with `pytest test_occlusion_graph.py -v`

### 12. Mobile QA Pipeline Toggle
- **File**: `features/try-on/AITryOnScreen.tsx`
- Dev-mode toggle button to switch between `sequential_v1` and `fused_v2`
- Pipeline version sent in every try-on request
- Diagnostics panel shows active pipeline version in results

### 13. Warp Fields Wired Into Fused Pipeline
- **File**: `mobile-vton-service/fused_pipeline.py`
- New `batched_warp` telemetry stage between preprocessing and denoising
- Each garment is warped to align with its semantic body region before denoising
- Warp summary included in diagnostics output

## Architecture Status

| Component | Status | Notes |
|---|---|---|
| Stage instrumentation | ✅ Done | telemetry.py |
| PersonCondition cache | ✅ Done | condition_cache.py |
| GarmentCondition cache | ✅ Done | condition_cache.py |
| Occlusion graph | ✅ Done | occlusion_graph.py |
| Shared encoding (Phase 1) | ✅ Done | fused_pipeline.py |
| Body-region mask generator | ✅ Done | region_masks.py |
| Region mask integration | ✅ Done | Attached to GarmentCondition + conflict detection |
| Evaluation harness | ✅ Done | evaluation.py + /evaluate endpoint |
| Batched warp fields | ✅ Done | warp_fields.py with category-specific affine transforms |
| Conditioning bank assembly | ✅ Done | assemble_conditioning_bank() for 5-slot tensor packing |
| Three-garment benchmark | ✅ Done | runThreeGarmentBenchmark.py with pass/fail report |
| Occlusion graph tests | ✅ Done | test_occlusion_graph.py with >=92% z-order accuracy |
| Mobile QA toggle | ✅ Done | Dev-mode pipeline version switch |
| Fused denoising (Phase 2) | ✅ Done | `multi_garment_denoiser.py` — single trajectory with batched garment UNet |
| Region-gated attention | ✅ Done | `MultiGarmentFeatureAggregator` soft-gates features by body-region masks |
| Crop seam refiner | ✅ Done | `seam_refiner.py` — conflict detection + boundary trimap + Laplacian blend |
| 5-slot accessory support | ✅ Graph ready | Occlusion graph + conditioning bank support 5 slots |
| Fused v3 API routing | ✅ Done | `/tryon/multi-fused` accepts `pipeline_version: fused_v3` |
| Mobile dev toggle | ✅ Done | `AITryOnScreen.tsx` cycles sequential_v1 / fused_v2 / fused_v3 |
| Node API routing | ✅ Done | `mobileVton.js` routes `fused_v3` to single-pass path |

## Month 4+ Deliverables Implemented

### 14. Multi-Garment Denoiser Engine
- **File**: `mobile-vton-service/multi_garment_denoiser.py`
- `MultiGarmentFeatureAggregator`: aggregates batched garment UNet features per block using region-mask-weighted combination
- `BatchedGarmentUNet`: runs `denoiser_garment` on all N garments in one forward pass per timestep
- `FusedDenoisingLoop`: single denoising trajectory with aggregated multi-garment conditioning
- `MultiGarmentTryonPipeline`: high-level wrapper that encodes person, stacks garment tensors, runs fused loop, and decodes output

### 15. Seam Refiner
- **File**: `mobile-vton-service/seam_refiner.py`
- `compute_boundary_trimap`: three-zone trimap (interior / boundary / exterior) for conflict regions
- `compute_seam_score`: gradient-discontinuity-based quality metric at mask edges
- `detect_conflict_regions`: auto-detects poor seams between overlapping garment pairs
- `apply_laplacian_blend`: pyramid blending for soft-boundary composition
- `SeamRefiner`: high-level class integrated into `fused_pipeline.py`

### 16. Fused Pipeline v2 Method
- **File**: `mobile-vton-service/fused_pipeline.py` — added `inference_multi_fused_v2()`
- Phase 2 path: occlusion graph → shared person encoding → garment preprocessing → batched warp → **single-pass fused denoising** → optional seam refiner
- Preserves Phase 1 (`inference_multi_fused`) as fallback path

### 17. API & Client Wiring
- **File**: `mobile-vton-service/main.py` — routes `pipeline_version=fused_v3` to Phase 2 method
- **File**: `api/services/mobileVtonClient.js` — `callMobileVtonMultiFused` accepts `pipelineVersion` parameter
- **File**: `api/services/strategies/mobileVton.js` — routes `fused_v3` through to client
- **File**: `features/try-on/AITryOnScreen.tsx` — dev toggle cycles all three versions with color-coded labels

## Remaining Future Work (Month 5+ Hardening)

1. ~~**Batched garment VAE + text encoder**~~ — **Done** (`condition_cache.py`): cache-aware per-garment check + batch VAE encode + batch text encode for uncached garments.
2. ~~**Transparent/semi-transparent garment handling**~~ — **Done** (`multi_garment_denoiser.py` + `fused_pipeline.py`): per-garment `opacity` scaling in `MultiGarmentFeatureAggregator` lets underlayers show through mesh/lace/transparent garments.
3. ~~**Learned warp model**~~ — **Partially done** (`warp_fields.py`): added `LearnedWarpModel` lightweight CNN architecture (3-layer encoder + flow head), `warp_mode="learned"` integration in `warp_garment_condition` and `batched_warp_garments`, and lazy checkpoint loading. Full training pipeline and dataset collection remain future work.
4. ~~**Crop-level mini-diffusion**~~ — **Partially done** (`seam_refiner.py`): replaced placeholder Gaussian blur with boundary-trimap Laplacian blending. Latent-space mini-diffusion plumbing wired via `final_latents` return; full 2-4 step denoising on crop is Month 5+ stretch.
5. **Production p95 targets**: 4 garments <=4.8s, 5 garments <=5.8s at 768x1024
6. **Validation**: outside-mask SSIM >= 0.985, blind review >= 60% tie/win
7. ~~**Object-ref artifact path**~~ — **Done** (`main.py`): added `/tryon/multi-fused/upload` multipart endpoint. Accepts raw `UploadFile` images (no base64 decode) and can return `StreamingResponse` raw PNG bytes via `return_format=png` (no base64 encode). This eliminates the ~33% base64 overhead on both request and response.
