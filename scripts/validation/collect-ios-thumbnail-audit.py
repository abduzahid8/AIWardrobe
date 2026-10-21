"""Collect a completed codec report from an explicitly selected simulator app container.
Usage: python3 collect-ios-thumbnail-audit.py CONTAINER OUTPUT_DIRECTORY
Copies only local files under that container; never launches a simulator.
"""
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

container = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=True)
report_file = container / 'Documents/thumbnail-audit.json'
report = json.loads(report_file.read_text())
if report.get('platform') != 'ios' or report.get('passed') is not True or len(report.get('cases', [])) != 12:
    raise ValueError('Native iOS audit did not complete all twelve cases successfully')
shutil.copyfile(report_file, output / 'device-report.json')
for index, case in enumerate(report['cases']):
    for key in ('originalPath', 'thumbnailPath', 'baselinePngPath'):
        if key not in case:
            continue
        uri = urlparse(case[key])
        if uri.scheme != 'file' or uri.netloc or uri.query or uri.fragment:
            raise ValueError('Expected a local app-container file')
        source = Path(unquote(uri.path)).resolve(strict=True)
        relative = source.relative_to(container)
        if relative.parts[0] not in ('Documents', 'Library') or not source.is_file():
            raise ValueError('Unexpected codec artifact path')
        destination = output / f'{index:02d}-{key}{source.suffix}'
        shutil.copyfile(source, destination)
        case[key] = str(destination)
(output / 'host-report.json').write_text(json.dumps(report, indent=2) + '\n')
print('Collected twelve native iOS codec cases')
