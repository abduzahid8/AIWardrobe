#!/usr/bin/env python3
"""Benchmark only the separate com.aiwardrobe.rcoffline QA app (actual App root, signed out).

No hosted endpoint, store purchase, account fixture or production app is targeted.
The entry disables billing/notification registration/analytics sending/prewarm.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

BUNDLE_ID = 'com.aiwardrobe.rcoffline'

def simctl(simulator, *args, check=True):
    return subprocess.run(['xcrun','simctl',args[0],simulator,*args[1:]],capture_output=True,text=True,check=check).stdout.strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulator')
    parser.add_argument('--runs',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.runs<=20:raise ValueError('runs must be 1..20')
    root=Path(simctl(args.simulator,'get_app_container',BUNDLE_ID,'data'))
    if args.simulator not in str(root) or 'CoreSimulator/Devices' not in str(root):raise RuntimeError('Not a simulator QA container')
    metric=root/'Documents/startup-timing.json';security=root/'Documents/rc-offline-app.json'
    samples=[]
    for index in range(args.runs):
        simctl(args.simulator,'terminate',BUNDLE_ID,check=False)
        metric.unlink(missing_ok=True);security.unlink(missing_ok=True)
        start=time.monotonic();simctl(args.simulator,'launch',BUNDLE_ID)
        while (not metric.exists() or not security.exists()) and time.monotonic()-start<30:time.sleep(0.05)
        if not metric.exists() or not security.exists():raise RuntimeError('No startup/security receipts in 30s')
        elapsed=(time.monotonic()-start)*1000
        timing=json.loads(metric.read_text());time.sleep(2)
        state=json.loads(security.read_text())
        if not state['releaseBuild'] or not state['isInitialized'] or state['authenticated'] or state['config']['analyticsSending']:
            raise RuntimeError('Offline QA state invariant failed')
        samples.append({'run':index+1,'processToReceiptMs':elapsed,'nativeVisibleMs':timing['durationMs'],'nativeBudgetPassed':timing['passed'],'nativeOrigin':timing['origin'],'route':state['currentRoute'],'blockedRequestCount':len(state['blockedRequests']),'localAssetRequests':state.get('localAssetRequests',0),'disabledServiceCalls':state['disabledServiceCalls']})
    native=[s['nativeVisibleMs'] for s in samples];wall=[s['processToReceiptMs'] for s in samples]
    report={'schemaVersion':1,'scope':'Actual App root, signed-out offline QA; not authenticated startup, real backend, billing or production configuration','bundleId':BUNDLE_ID,'simulatorId':args.simulator,'method':'simctl process-cold launches preserving local language; existing RN native-start-to-revealed-screen metric plus process-to-file receipt including CLI/IO','buildType':'Release simulator arm64','productionClaim':False,'samples':samples,'nativeVisibleP50Ms':statistics.median(native),'nativeVisibleRangeMs':[min(native),max(native)],'nativeBudgetPassedAll':all(s['nativeBudgetPassed'] for s in samples),'processToReceiptP50Ms':statistics.median(wall),'processToReceiptRangeMs':[min(wall),max(wall)]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
