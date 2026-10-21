"""Compare native thumbnails against independent Pillow EXIF transforms.
Usage: python3 verify-native-orientation.py /path/to/thumbnail-audit.json
Only synthetic EXIF fixtures are inspected; no image content is modified.
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote
from PIL import Image, ImageOps, ImageChops, ImageStat

root = Path(__file__).resolve().parents[2]
report = json.loads(Path(sys.argv[1]).read_text())
platform = report.get('platform', 'ios')
if platform not in ('ios', 'android'):
    raise ValueError('Unexpected native platform')
results = []
for orientation in range(1, 9):
    entry = next((case for case in report['cases'] if case['name'] == f'EXIF orientation {orientation}'), None)
    if entry is None:
        results.append({'orientation': orientation, 'passed': False, 'missingNativeResult': True})
        continue
    fixture = root / 'scripts/validation/fixtures/photo-orientation' / f'orientation-{orientation}.jpg'
    original = Path(unquote(urlparse(entry['originalPath']).path))
    thumbnail = Path(unquote(urlparse(entry['thumbnailPath']).path))
    source = Image.open(fixture)
    assert source.getexif()[274] == orientation
    expected = ImageOps.exif_transpose(source).convert('RGB')
    actual = Image.open(thumbnail).convert('RGB')
    dimensions_match = max(actual.size) <= 512 and abs(actual.width / actual.height - expected.width / expected.height) < 0.01
    expected = expected.resize(actual.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(actual, expected)
    mean_error = sum(ImageStat.Stat(difference).mean) / 3
    corner_error = max(max(abs(a-b) for a,b in zip(actual.getpixel((int(actual.width*x),int(actual.height*y))),expected.getpixel((int(actual.width*x),int(actual.height*y))))) for x,y in [(0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9)])
    unchanged = hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(fixture.read_bytes()).digest()
    results.append({'orientation': orientation, 'dimensions': actual.size, 'meanAbsoluteChannelError': mean_error, 'maxCornerChannelError': corner_error, 'originalBytesUnchanged': unchanged, 'passed': dimensions_match and unchanged and mean_error < 5 and corner_error < 15})
output = {'scope': f'Native {platform} output compared to independent Pillow EXIF transpose; eight synthetic JPEG fixtures', 'thresholds': {'meanAbsoluteChannelErrorBelow': 5, 'maxCornerChannelErrorBelow': 15}, 'cases': results, 'passed': len(results) == 8 and all(case['passed'] for case in results)}
destination = Path(sys.argv[2]) if len(sys.argv) > 2 else root / ('docs/reports/readiness-evidence/native-android-jpeg-orientation.json' if platform == 'android' else 'docs/reports/readiness-evidence/native-jpeg-orientation.json')
destination.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output))
sys.exit(0 if output['passed'] else 1)
