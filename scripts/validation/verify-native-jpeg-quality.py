"""Verify collected native JPEG previews against same-codec PNG baselines.
Usage: python3 verify-native-jpeg-quality.py host-report.json output.json
Requires Pillow; input paths must refer to locally collected synthetic fixtures.
"""
import hashlib
import json
import math
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

root = Path(__file__).resolve().parents[2]
report = json.loads(Path(sys.argv[1]).read_text())
results = []
for number in (1, 2):
    case = next(c for c in report['cases'] if c['name'] == f'Opaque photo {number}')
    original = Path(case['originalPath'])
    thumbnail = Path(case['thumbnailPath'])
    baseline = Path(case['baselinePngPath'])
    actual, expected = Image.open(thumbnail), Image.open(baseline)
    dimensions_match = actual.size == expected.size and max(actual.size) <= 512
    jpeg = actual.format == 'JPEG'
    difference = ImageStat.Stat(ImageChops.difference(actual.convert('RGB'), expected.convert('RGB')))
    mae = sum(difference.mean) / 3
    mse = sum(value ** 2 for value in difference.rms) / 3
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else 100
    fixture = root / f'scripts/validation/fixtures/opaque-photo/photo-{number}.jpg'
    unchanged = hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(fixture.read_bytes()).digest()
    results.append({'name': case['name'], 'jpegBytes': thumbnail.stat().st_size,
                    'pngBytes': baseline.stat().st_size,
                    'reductionPercent': 100 * (1 - thumbnail.stat().st_size / baseline.stat().st_size),
                    'meanAbsoluteChannelError': mae, 'psnrDb': psnr,
                    'originalUnchanged': unchanged,
                    'passed': jpeg and dimensions_match and unchanged and mae < 3 and psnr >= 35})
output = {'scope': 'Native JPEG vs same-native PNG; repository-derived photos, not camera captures',
          'thresholds': {'meanAbsoluteChannelErrorBelow': 3, 'psnrDbAtLeast': 35},
          'cases': results, 'passed': all(case['passed'] for case in results)}
Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output))
sys.exit(0 if output['passed'] else 1)
