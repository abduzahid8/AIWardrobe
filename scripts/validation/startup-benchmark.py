#!/usr/bin/env python3
"""Measure process-cold Release launches; build with EXPO_PUBLIC_MEASURE_STARTUP=true.

Usage: python3 scripts/validation/startup-benchmark.py SIMULATOR_UDID --runs 5
The app must already be installed. Existing app data is preserved.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def simctl(*args, check=True):
    return subprocess.run(['xcrun', 'simctl', *args], check=check,
                          capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulator')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    container = Path(simctl('get_app_container', args.simulator, 'com.aiwardrobe', 'data'))
    metric = container / 'Documents/startup-timing.json'
    results = []
    for index in range(args.runs):
        simctl('terminate', args.simulator, 'com.aiwardrobe', check=False)
        metric.unlink(missing_ok=True)
        launched = time.monotonic()
        simctl('launch', args.simulator, 'com.aiwardrobe')
        deadline = launched + 30
        while not metric.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not metric.exists():
            raise RuntimeError('No startup measurement after 30s; check Release instrumentation and launch errors')
        result = json.loads(metric.read_text())
        result['run'] = index + 1
        results.append(result)
        print(json.dumps(result), flush=True)
        # Allow deferred services to settle before the next process-cold launch.
        time.sleep(3)
    report = {'scenario': 'process-cold; current persisted session/language', 'runs': results,
              'maximumMs': max(row['durationMs'] for row in results),
              'passed': all(row['passed'] for row in results)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
