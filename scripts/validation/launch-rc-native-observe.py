#!/usr/bin/env python3
"""Observe only the isolated com.aiwardrobe.rclocal simulator harness, never the normal app.

Build scripts/validation/launch-rc-native-entry.tsx as a Release ENTRY_FILE first.
This measures an isolated fixture, not AIWardrobe Home/auth/provider startup.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

BUNDLE_ID = 'com.aiwardrobe.rclocal'

def simctl(simulator, *args, check=True):
    return subprocess.run(['xcrun', 'simctl', args[0], simulator, *args[1:]], check=check,
                          text=True, capture_output=True).stdout.strip()

def fixture_file(simulator):
    container = Path(simctl(simulator, 'get_app_container', BUNDLE_ID, 'data'))
    if simulator not in str(container) or 'CoreSimulator/Devices' not in str(container):
        raise RuntimeError('Refusing a non-simulator data container')
    return container / 'Documents/rc-local-native.json'

def write(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['benchmark', 'memory'])
    parser.add_argument('simulator')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result_path = fixture_file(args.simulator)
    common = {'schemaVersion': 1, 'bundleId': BUNDLE_ID, 'simulatorId': args.simulator,
              'scope': 'isolated Release native harness, not normal client RC', 'productionClaim': False}
    if args.mode == 'benchmark':
        if not 1 <= args.runs <= 20:
            raise ValueError('runs must be 1..20')
        samples = []
        for index in range(args.runs):
            simctl(args.simulator, 'terminate', BUNDLE_ID, check=False)
            result_path.unlink(missing_ok=True)  # Only this disposable harness receipt.
            start = time.monotonic()
            launch = simctl(args.simulator, 'launch', BUNDLE_ID)
            pid = int(re.search(r':\s*(\d+)$', launch).group(1))
            while not result_path.exists() and time.monotonic() - start < 30:
                time.sleep(0.05)
            if not result_path.exists():
                raise RuntimeError('No first-layout receipt in 30 seconds')
            receipt = json.loads(result_path.read_text())
            if not receipt.get('releaseBuild') or receipt.get('networkAttempts') != 0:
                raise RuntimeError('Expected offline Release fixture receipt')
            samples.append({'run': index + 1, 'processToReceiptMs': (time.monotonic() - start) * 1000,
                            'moduleToFirstLayoutMs': receipt['moduleToFirstLayoutMs'], 'pid': pid})
            time.sleep(0.5)
        durations = [row['processToReceiptMs'] for row in samples]
        write(args.output, {**common, 'method': 'simctl process launch to first-layout JSON receipt; includes launch tool and receipt IO; excludes full app root; OS caches retained',
                            'samples': samples, 'p50Ms': statistics.median(durations), 'rangeMs': [min(durations), max(durations)],
                            'moduleLayoutP50Ms': statistics.median(row['moduleToFirstLayoutMs'] for row in samples)})
    else:
        if not args.pid or not 1 <= args.seconds <= 180:
            raise ValueError('memory mode requires --pid and seconds 1..180')
        command = subprocess.check_output(['ps', '-p', str(args.pid), '-o', 'command='], text=True).strip()
        if '/com.apple.CoreSimulator.SimRuntime.' in command:
            raise RuntimeError('Expected the harness process, not simulator service')
        if '/AIWardrobe.app/AIWardrobe' not in command:
            raise RuntimeError('Refusing to observe an unrelated process')
        start = time.monotonic()
        samples = []
        while time.monotonic() - start < args.seconds:
            rss = subprocess.run(['ps', '-p', str(args.pid), '-o', 'rss='], text=True, capture_output=True)
            if rss.returncode != 0 or not rss.stdout.strip():
                raise RuntimeError('Harness process ended during memory observation')
            try:
                status = json.loads(result_path.read_text()).get('status', 'unknown')
            except (FileNotFoundError, json.JSONDecodeError):
                status = 'unknown'
            samples.append({'elapsedSeconds': time.monotonic() - start, 'rssKiB': int(rss.stdout.strip()), 'harnessStatus': status})
            time.sleep(1)
        write(args.output, {**common, 'method': 'host ps RSS of iOS Simulator app process at 1s intervals; includes native caches; not physical-device RAM and not JS retained heap',
                            'pid': args.pid, 'samples': samples, 'baselineKiB': samples[0]['rssKiB'],
                            'peakKiB': max(row['rssKiB'] for row in samples), 'finalKiB': samples[-1]['rssKiB'],
                            'leakConclusion': 'not established; no controlled GC or repeat-cycle baseline'})
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
