"""Replay numbered migrations in the isolated readiness DB, never production.
Use --from NNN to resume after inspecting/fixing a failed migration.
"""
from pathlib import Path
import subprocess
import sys
start = sys.argv[sys.argv.index('--from') + 1] if '--from' in sys.argv else '001'
for migration in sorted(Path('backend/supabase/migrations').glob('[0-9]*.sql')):
    if (migration.name if '_' in start else migration.name[:3]) < start:
        continue
    result = subprocess.run(['psql', '-h', '/tmp', '-p', '55439', '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '--single-transaction', '-f', str(migration)], capture_output=True, text=True)
    with open('/tmp/aiwardrobe-migrations.log', 'a') as log:
        log.write('\n' + migration.name + '\n' + result.stdout + result.stderr)
    print(migration.name, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
    if result.returncode:
        print(result.stderr[-1500:])
        sys.exit(result.returncode)
