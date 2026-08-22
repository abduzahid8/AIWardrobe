#!/usr/bin/env bash
# Deploy the v17 mobile-vton-service to Modal and run the standard 3-garment test.
#
# Prerequisites:
#   1. Modal CLI installed:        pip install modal
#   2. Auth set with NEW account:  modal token set --token-id <ID> --token-secret <SECRET>
#      (token ID starts with ak-, secret starts with as-)
#   3. .env at project root has:   HF_TOKEN=hf_...
#
# Usage:
#   cd backend/mobile-vton-service   (from the repo root)
#   bash deploy_and_test.sh
#
# What it does:
#   1. cd into mobile-vton-service
#   2. modal deploy modal_app.py           (builds image, downloads SD1.5 weights ~4 GB)
#   3. parses the deployed URL from output
#   4. curl /health to verify GPU
#   5. python tests/test_new_code_live.py  (3-garment test: top + pants + shoes)
#
# First deploy takes 5-10 minutes (model download). Subsequent deploys ~30s.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> [1/4] Checking Modal auth..."
if ! modal token list 2>/dev/null | grep -q .; then
  echo "ERROR: Modal auth not set. Run:"
  echo "  modal token set --token-id <YOUR_ID> --token-secret <YOUR_SECRET>"
  exit 1
fi

echo "==> [2/4] Deploying to Modal..."
DEPLOY_OUTPUT=$(modal deploy modal_app.py 2>&1)
echo "$DEPLOY_OUTPUT"

# Extract the deployed URL from the deploy output.
# Modal prints: "App deployed at https://<workspace>--aiwardrobe-mobile-vton-fastapi-app.modal.run"
URL=$(echo "$DEPLOY_OUTPUT" | grep -oE 'https://[^ ]+aiwardrobe-mobile-vton-fastapi-app[^ ]*' | head -1)

if [ -z "$URL" ]; then
  echo "ERROR: Could not find deployed URL in Modal output."
  echo "Look for the URL above and update ENDPOINT in tests/test_new_code_live.py manually."
  exit 1
fi

echo ""
echo "==> Deployed URL: $URL"
echo ""

echo "==> [3/4] Health check..."
HEALTH=$(curl -sS "${URL}/health" || echo "FAILED")
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

# Verify pipeline_version is v17
if echo "$HEALTH" | grep -q "v17_soft_color_match"; then
  echo "==> Pipeline version: v17_soft_color_match ✓"
else
  echo "WARNING: pipeline_version is not v17. Check health response above."
fi

echo ""
echo "==> [4/4] Running 3-garment test (seed=7, 100)..."
# Patch the ENDPOINT in tests/test_new_code_live.py for this run.
ENDPOINT="${URL}/tryon/multi-fused" python3 -c "
import os, sys
import ssl, urllib.request, json, base64, time
from pathlib import Path

# CWD is already backend/mobile-vton-service/ (see the `cd` above).
ROOT = Path(os.getcwd()).resolve().parent.parent
ASSETS = ROOT / 'assets' / 'images'
OUT = Path(os.getcwd())
ENDPOINT = os.environ['ENDPOINT']
ctx = ssl._create_unverified_context()

def b64(p): return base64.b64encode(p.read_bytes()).decode()

def call(payload, timeout=600):
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        return json.loads(r.read())

def save_result(resp, name):
    img_b64 = resp['result_image']
    if img_b64.startswith('data:'): img_b64 = img_b64.split(',', 1)[1]
    (OUT / name).write_bytes(base64.b64decode(img_b64))
    print(f'  saved {name} ({len(img_b64)} chars b64)')

for seed in (7, 100):
    print(f'\n=== seed {seed} ===')
    payload = {
        'person_image': b64(ASSETS / 'mannequin_front.png'),
        'garments': [
            {'label': 'top',   'garment_image': b64(ASSETS / 'basic_white_tshirt.png')},
            {'label': 'pants', 'garment_image': b64(ASSETS / 'basic_brown_pants.png')},
            {'label': 'shoes', 'garment_image': b64(ASSETS / 'basic_brown_loafers.png')},
        ],
        'seed': seed,
        'guidance_scale': 7.5,
        'num_inference_steps': 25,
        'pipeline_version': 'fused_v17',
    }
    t0 = time.time()
    resp = call(payload)
    dt = time.time() - t0
    print(f'  success={resp.get(\"success\")}  method={resp.get(\"method_used\")}  '
          f'elapsed={resp.get(\"elapsed_ms\")}ms  wall={dt:.1f}s')
    if resp.get('diagnostics'):
        diag = resp['diagnostics']
        print(f'  rendered={diag.get(\"renderedGarments\")}  order={diag.get(\"dressingOrder\")}')
        for cd in diag.get('colorDiagnostics', []):
            tgt = cd['targetRgb']; rnd = cd['renderedRgb']; d = cd['delta']
            print(f'  {cd[\"label\"]:5s}: target={tgt}  rendered={rnd}  delta=({d[\"r\"]:+d},{d[\"g\"]:+d},{d[\"b\"]:+d})')
    save_result(resp, f'test_v17_seed{seed}.png')

print()
print('All done. Compare:')
print(f'  ls -la {OUT}/test_v17_seed*.png')
"

echo ""
echo "==> Done. Test images saved to mobile-vton-service/test_v17_seed*.png"
