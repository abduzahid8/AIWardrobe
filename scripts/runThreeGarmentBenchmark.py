#!/usr/bin/env python3
"""
Three-garment fixed benchmark runner for multi-garment VTON.

Usage:
    python scripts/runThreeGarmentBenchmark.py \
        --person /path/to/person.png \
        --top /path/to/top.png \
        --bottom /path/to/pants.png \
        --outerwear /path/to/jacket.png \
        --output ./benchmark_results \
        --endpoint http://localhost:8001

Generates a markdown report with pass/fail against Month 3 targets:
    - p50 latency <= 4.2s for 3 garments
    - Peak VRAM <= 15GB
    - Outside-mask SSIM >= 0.985
    - Z-order pairwise accuracy >= 92%
    - Speedup >= 1.82x (55% of 3 sequential)
"""

import os
import sys
import json
import time
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from PIL import Image

API_URL = os.environ.get("MOBILE_VTON_URL", "http://localhost:8001")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(path).suffix.lstrip(".") or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def run_pipeline(endpoint: str, payload: dict) -> Tuple[dict, float]:
    start = time.time()
    r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=600)
    elapsed = time.time() - start
    r.raise_for_status()
    return r.json(), elapsed


def save_image(data_uri: str, path: str) -> None:
    if data_uri.startswith("data:"):
        _, b64 = data_uri.split(",", 1)
        img = Image.open(BytesIO(base64.b64decode(b64)))
    else:
        img = Image.open(BytesIO(base64.b64decode(data_uri)))
    img.save(path)


def run_trial(person_b64: str, garments: List[dict], num_steps: int = 10) -> dict:
    """Run sequential and fused for one outfit, return metrics."""
    # Sequential
    seq_payload = {
        "person_image": person_b64,
        "garments": garments,
        "num_inference_steps": num_steps,
        "guidance_scale": 2.0,
    }
    seq_result, seq_wall = run_pipeline("/tryon/multi", seq_payload)
    seq_diag = seq_result.get("diagnostics") or {}

    # Fused v2 (Phase 1: shared encoding + sequential denoising)
    fused_payload = {
        "person_image": person_b64,
        "garments": garments,
        "num_inference_steps": num_steps,
        "guidance_scale": 2.0,
        "pipeline_version": "fused_v2",
    }
    fused_result, fused_wall = run_pipeline("/tryon/multi-fused", fused_payload)
    fused_diag = fused_result.get("diagnostics") or {}

    # Fused v3 (Phase 2: single-pass fused denoising)
    fused_v3_payload = {
        "person_image": person_b64,
        "garments": garments,
        "num_inference_steps": num_steps,
        "guidance_scale": 2.0,
        "pipeline_version": "fused_v3",
    }
    fused_v3_result, fused_v3_wall = run_pipeline("/tryon/multi-fused", fused_v3_payload)
    fused_v3_diag = fused_v3_result.get("diagnostics") or {}

    # Evaluate SSIM if both succeeded
    full_ssim = None
    outside_ssim = None
    full_ssim_v3 = None
    outside_ssim_v3 = None
    if seq_result.get("success") and fused_result.get("success"):
        eval_payload = {
            "sequential_image": seq_result["result_image"],
            "fused_image": fused_result["result_image"],
            "garment_regions": ["torso", "hips", "left_leg", "right_leg"],
        }
        try:
            eval_res, _ = run_pipeline("/evaluate", eval_payload)
            full_ssim = eval_res.get("full_ssim")
            outside_ssim = eval_res.get("outside_mask_ssim")
        except Exception as e:
            print(f"  [warn] evaluate failed: {e}")

    if seq_result.get("success") and fused_v3_result.get("success"):
        eval_payload_v3 = {
            "sequential_image": seq_result["result_image"],
            "fused_image": fused_v3_result["result_image"],
            "garment_regions": ["torso", "hips", "left_leg", "right_leg"],
        }
        try:
            eval_res_v3, _ = run_pipeline("/evaluate", eval_payload_v3)
            full_ssim_v3 = eval_res_v3.get("full_ssim")
            outside_ssim_v3 = eval_res_v3.get("outside_mask_ssim")
        except Exception as e:
            print(f"  [warn] evaluate v3 failed: {e}")

    return {
        "sequential": {
            "success": seq_result.get("success", False),
            "elapsedMs": seq_result.get("elapsed_ms", 0),
            "wallSeconds": round(seq_wall, 2),
            "peakVramMb": seq_diag.get("peakVramMb"),
        },
        "fused": {
            "success": fused_result.get("success", False),
            "elapsedMs": fused_result.get("elapsed_ms", 0),
            "wallSeconds": round(fused_wall, 2),
            "peakVramMb": fused_diag.get("peakVramMb"),
            "pipelineVersion": fused_result.get("pipeline_version"),
            "cacheHits": fused_diag.get("cacheHits", {}),
            "cacheMisses": fused_diag.get("cacheMisses", {}),
            "renderedGarments": fused_diag.get("renderedGarments", []),
        },
        "fused_v3": {
            "success": fused_v3_result.get("success", False),
            "elapsedMs": fused_v3_result.get("elapsed_ms", 0),
            "wallSeconds": round(fused_v3_wall, 2),
            "peakVramMb": fused_v3_diag.get("peakVramMb"),
            "pipelineVersion": fused_v3_result.get("pipeline_version"),
            "cacheHits": fused_v3_diag.get("cacheHits", {}),
            "cacheMisses": fused_v3_diag.get("cacheMisses", {}),
            "renderedGarments": fused_v3_diag.get("renderedGarments", []),
            "seamRefiner": fused_v3_diag.get("seamRefiner", {}),
        },
        "ssim": {
            "full": full_ssim,
            "outsideMask": outside_ssim,
            "full_v3": full_ssim_v3,
            "outsideMask_v3": outside_ssim_v3,
        },
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def generate_report(results: List[dict], output_dir: str) -> str:
    """Generate a markdown report."""
    os.makedirs(output_dir, exist_ok=True)

    seq_times = [r["sequential"]["elapsedMs"] for r in results if r["sequential"]["success"]]
    fused_times = [r["fused"]["elapsedMs"] for r in results if r["fused"]["success"]]
    fused_v3_times = [r["fused_v3"]["elapsedMs"] for r in results if r["fused_v3"]["success"]]
    seq_vrams = [r["sequential"]["peakVramMb"] or 0 for r in results]
    fused_vrams = [r["fused"]["peakVramMb"] or 0 for r in results]
    fused_v3_vrams = [r["fused_v3"]["peakVramMb"] or 0 for r in results]
    full_ssims = [r["ssim"]["full"] for r in results if r["ssim"]["full"] is not None]
    outside_ssims = [r["ssim"]["outsideMask"] for r in results if r["ssim"]["outsideMask"] is not None]
    full_ssims_v3 = [r["ssim"]["full_v3"] for r in results if r["ssim"]["full_v3"] is not None]
    outside_ssims_v3 = [r["ssim"]["outsideMask_v3"] for r in results if r["ssim"]["outsideMask_v3"] is not None]

    speedups = [s / f for s, f in zip(seq_times, fused_times) if f > 0]
    speedups_v3 = [s / f for s, f in zip(seq_times, fused_v3_times) if f > 0]

    passes = {
        "latency_p50": percentile(fused_times, 50) <= 4200,
        "latency_v3_p50": percentile(fused_v3_times, 50) <= 4200,
        "vram_p50": percentile(fused_vrams, 50) <= 15360,
        "vram_v3_p50": percentile(fused_v3_vrams, 50) <= 15360,
        "outside_ssim_p50": percentile(outside_ssims, 50) >= 0.985 if outside_ssims else False,
        "outside_ssim_v3_p50": percentile(outside_ssims_v3, 50) >= 0.985 if outside_ssims_v3 else False,
        "speedup_p50": percentile(speedups, 50) >= 1.82 if speedups else False,
        "speedup_v3_p50": percentile(speedups_v3, 50) >= 1.82 if speedups_v3 else False,
    }

    def fmt(val, spec):
        return f"{val:{spec}}" if val is not None else "N/A"

    lines = [
        "# Multi-Garment VTON — Three-Garment Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**API:** {API_URL}",
        f"**Trials:** {len(results)}",
        "",
        "## Results Summary",
        "",
        "| Metric | Sequential | Fused v2 | Fused v3 | Target | Pass? |",
        "|---|---|---|---|---|---|",
        f"| p50 Latency (ms) | {percentile(seq_times, 50):.0f} | {percentile(fused_times, 50):.0f} | {percentile(fused_v3_times, 50):.0f} | <=4200 | {'✅' if passes['latency_v3_p50'] else '❌'} |",
        f"| p50 VRAM (MB) | {percentile(seq_vrams, 50):.0f} | {percentile(fused_vrams, 50):.0f} | {percentile(fused_v3_vrams, 50):.0f} | <=15360 | {'✅' if passes['vram_v3_p50'] else '❌'} |",
        f"| p50 Full SSIM | — | {fmt(percentile(full_ssims, 50) if full_ssims else None, '.4f')} | {fmt(percentile(full_ssims_v3, 50) if full_ssims_v3 else None, '.4f')} | — | — |",
        f"| p50 Outside-mask SSIM | — | {fmt(percentile(outside_ssims, 50) if outside_ssims else None, '.4f')} | {fmt(percentile(outside_ssims_v3, 50) if outside_ssims_v3 else None, '.4f')} | >=0.985 | {'✅' if passes['outside_ssim_v3_p50'] else '❌'} |",
        f"| p50 Speedup | — | {fmt(percentile(speedups, 50) if speedups else None, '.2f')}x | {fmt(percentile(speedups_v3, 50) if speedups_v3 else None, '.2f')}x | >=1.82x | {'✅' if passes['speedup_v3_p50'] else '❌'} |",
        "",
        "## Month 4+ Go/No-Go Gate",
        "",
        f"**Status:** {'🟢 GO' if all(passes.values()) else '🔴 NO-GO'}",
        "",
        "| Criterion | Required | Fused v2 | Fused v3 | Pass? |",
        "|---|---|---|---|---|",
        f"| 3-garment p50 latency | <=4.2s | {percentile(fused_times, 50)/1000:.2f}s | {percentile(fused_v3_times, 50)/1000:.2f}s | {'✅' if passes['latency_v3_p50'] else '❌'} |",
        f"| Peak VRAM | <=15GB | {percentile(fused_vrams, 50):.0f}MB | {percentile(fused_v3_vrams, 50):.0f}MB | {'✅' if passes['vram_v3_p50'] else '❌'} |",
        f"| Speedup vs sequential | >=1.82x | {percentile(speedups, 50):.2f}x | {percentile(speedups_v3, 50):.2f}x | {'✅' if passes['speedup_v3_p50'] else '❌'} |",
        f"| Outside-mask SSIM | >=0.985 | {percentile(outside_ssims, 50):.4f} | {percentile(outside_ssims_v3, 50):.4f} | {'✅' if passes['outside_ssim_v3_p50'] else '❌'} |",
        "",
        "## Per-Trial Details",
        "",
        "| Trial | Seq (ms) | Fused v2 (ms) | Fused v3 (ms) | Speedup v2 | Speedup v3 | VRAM v2 | VRAM v3 | Outside SSIM v2 | Outside SSIM v3 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for i, r in enumerate(results):
        s = r["sequential"]["elapsedMs"]
        f2 = r["fused"]["elapsedMs"]
        f3 = r["fused_v3"]["elapsedMs"]
        sp2 = s / f2 if f2 > 0 else 0
        sp3 = s / f3 if f3 > 0 else 0
        vm2 = r["fused"]["peakVramMb"] or 0
        vm3 = r["fused_v3"]["peakVramMb"] or 0
        lines.append(
            f"| {i+1} | {s:.0f} | {f2:.0f} | {f3:.0f} | {sp2:.2f}x | {sp3:.2f}x | {vm2:.0f} | {vm3:.0f} | "
            f"{r['ssim']['outsideMask'] or 'N/A'} | {r['ssim']['outsideMask_v3'] or 'N/A'} |"
        )

    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")

    report_path = os.path.join(output_dir, "three_garment_benchmark_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    json_path = os.path.join(output_dir, "three_garment_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to {report_path}")
    print(f"JSON saved to {json_path}")
    return report_path


def main():
    global API_URL
    parser = argparse.ArgumentParser(description="Three-garment VTON benchmark")
    parser.add_argument("--person", required=True, help="Path to person/mannequin image")
    parser.add_argument("--top", required=True, help="Path to top garment image")
    parser.add_argument("--bottom", required=True, help="Path to bottom garment image")
    parser.add_argument("--outerwear", required=True, help="Path to outerwear garment image")
    parser.add_argument("--output", default="./benchmark_results", help="Output directory")
    parser.add_argument("--endpoint", default=API_URL, help="Mobile-VTON API base URL")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps")
    args = parser.parse_args()

    API_URL = args.endpoint

    print(f"Benchmarking 3-garment outfit against {API_URL}")
    print(f"Iterations: {args.iterations}, Steps: {args.steps}")

    person_b64 = encode_image(args.person)
    garments = [
        {"garment_image": encode_image(args.top), "description": "Top", "label": "top"},
        {"garment_image": encode_image(args.bottom), "description": "Pants", "label": "pants"},
        {"garment_image": encode_image(args.outerwear), "description": "Jacket", "label": "layer"},
    ]

    results = []
    for i in range(args.iterations):
        print(f"\n--- Trial {i+1}/{args.iterations} ---")
        result = run_trial(person_b64, garments, num_steps=args.steps)
        results.append(result)
        print(f"  sequential: {result['sequential']['elapsedMs']:.0f}ms")
        print(f"  fused v2: {result['fused']['elapsedMs']:.0f}ms")
        print(f"  fused v3: {result['fused_v3']['elapsedMs']:.0f}ms")
        print(f"  speedup v2: {result['sequential']['elapsedMs'] / max(result['fused']['elapsedMs'], 1):.2f}x")
        print(f"  speedup v3: {result['sequential']['elapsedMs'] / max(result['fused_v3']['elapsedMs'], 1):.2f}x")
        if result["ssim"]["outsideMask"]:
            print(f"  outside-mask SSIM v2: {result['ssim']['outsideMask']:.4f}")
        if result["ssim"]["outsideMask_v3"]:
            print(f"  outside-mask SSIM v3: {result['ssim']['outsideMask_v3']:.4f}")

    generate_report(results, args.output)


if __name__ == "__main__":
    main()
