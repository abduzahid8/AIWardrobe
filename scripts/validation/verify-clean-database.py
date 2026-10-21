"""Create a new isolated local DB and verify a clean installation. No drops or remote hosts."""
from pathlib import Path
import subprocess
import time

name = 'aiwardrobe_readiness_' + str(time.time_ns())
base = ['psql', '-h', '/tmp', '-p', '55439', '-v', 'ON_ERROR_STOP=1']
log_path = Path('/tmp/aiwardrobe-clean-replay.log')
with log_path.open('w') as log:
    subprocess.run(base + ['-d', 'postgres', '-c', f'CREATE DATABASE {name};'], stdout=log, stderr=log, check=True)
    files = [Path('scripts/validation/bootstrap-local.sql'), *sorted(Path('backend/supabase/migrations').glob('[0-9]*.sql')), Path('scripts/validation/seed-capacity.sql'), Path('scripts/validation/security.sql'), Path('scripts/validation/revenuecat.sql'), Path('scripts/validation/usage-retention.sql')]
    for file in files:
        result = subprocess.run(base + ['-d', name, '--single-transaction', '-f', str(file)], stdout=log, stderr=log)
        print(file.name, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    for scenario in ['quota', 'revenuecat']:
        fixture = f'scripts/validation/{scenario}-concurrency.pgbench'
        subprocess.run(['pgbench', '-n', '-h', '/tmp', '-p', '55439', '-c', '40', '-j', '4', '-t', '20', '-f', fixture, name], stdout=log, stderr=log, check=True)
        subprocess.run(base + ['-d', name, '-f', f'scripts/validation/{scenario}-concurrency-assert.sql'], stdout=log, stderr=log, check=True)
        print(scenario + ' concurrency PASS', flush=True)
print('Clean replay and security assertions passed. Database:', name)
