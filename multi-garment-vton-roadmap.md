# Multi-Garment VTON Roadmap Plan

This plan will produce a production-ready 6-month roadmap for replacing the current sequential single-garment Mobile-VTON loop with a hybrid mobile + cloud GPU multi-garment architecture.

## Grounding From Current System

- **Client path:** `features/try-on/AITryOnScreen.tsx` sends selected slots as `garments` to `/api/tryon/mobile-vton` with a 180s timeout.
- **API path:** `api/services/strategies/mobileVton.js` orders garments by `GARMENT_RENDER_ORDER` and calls `callMobileVtonMulti`.
- **Service path:** `mobile-vton-service/main.py` implements `/tryon/multi` by looping over garments and calling the single-garment pipeline repeatedly.
- **Model path:** `mobile-vton-service/tryon.py` and `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py` recompute prompt encodings, person VAE latents, garment VAE latents, IP-adapter image embeddings, and `denoiser_garment` features per run.
- **Existing useful assets:** garment ordering, deterministic masks/anchors, preprocess cache, FLUX pose-anchored experiments, and Modal A10G deployment config.

## Chosen Baseline Assumption

- **Deployment target:** hybrid mobile + cloud GPU.
- **Why:** this best satisfies the target of diffusion-quality multi-garment realism while beating N sequential runs; mobile should handle capture, segmentation previews, garment preprocessing hints, cache warming, and low-res preview, while server GPU handles final 768×1024/1024×1024 multi-garment diffusion.

## Roadmap Deliverable Structure

1. **Pipeline redesign**
   - Specify a single multi-garment inference graph with shared person encoding, batched garment encoders, garment-slot tokens, region masks, occlusion graph, and one final denoising/composition pass.
   - Explicitly state reusable vs non-reusable computation.
   - Define dynamic handling for 2-5 garments without separate code paths.

2. **Speed architecture**
   - Quantify current bottlenecks from sequential reruns.
   - Provide latency deltas for shared encoders, batched garment conditioning, cached text/image embeddings, mask reuse, and preview/final tiering.
   - Define theoretical minimum latency floors for 2/3/4 garments under A10G/L4-class server GPU assumptions.

3. **Memory architecture**
   - Identify peak memory sources: model weights, latents, garment features, attention KV/cache-like activations, masks, PIL/base64 buffers, and batch tensors.
   - Propose constant/sub-linear memory strategies: streaming garment encodes, fixed-size conditioning bank, reusable person latents, quantized/offloaded encoders, resolution tiers, and explicit cleanup.
   - Include garment count × resolution × technique budget tables.

4. **Occlusion and composition engine**
   - Define metadata + segmentation driven layering rules for tops, bottoms, outerwear, shoes, accessories, transparent garments, and cross-body accessories.
   - Include seam handling, alpha/matte rules, boundary refinement, contact shadows, and conflict resolution.

5. **Six-month week-level roadmap**
   - For each month: engineering focus, research tasks, benchmark targets, go/no-go gate, top risks/mitigations, and team allocation.
   - Include a structured table plus deeper narrative per month.

6. **Tradeoff and risk analysis**
   - Provide explicit speed/quality/memory matrix for major architecture decisions.
   - Include silent/catastrophic failure modes with detection and recovery.

7. **Benchmark and evaluation framework**
   - Define a production regression suite with garment categories, pose diversity, occlusion/edge cases, metric thresholds, and per-model-update checks.

## Important Assumptions To Make Explicit In The Roadmap

- Server final render baseline: warm Modal/GPU container, A10G-class GPU today, L4/A10G optimized target, bf16/fp16 inference.
- Current sequential baseline: one full diffusion run per garment; multi-garment speedup must be measured against this, not against FLUX or third-party APIs.
- Initial output resolution tiers: 512×768 preview, 768×1024 default, 1024×1024 high quality.
- Multi-garment max target for this roadmap: 5 simultaneously rendered garments.
- Memory budgets should include both GPU VRAM and process RAM, including base64/PIL decode overhead in the current API path.

## Decision Needed Before Full Roadmap

Proceed with generating the full roadmap using the hybrid mobile + cloud GPU baseline above.
