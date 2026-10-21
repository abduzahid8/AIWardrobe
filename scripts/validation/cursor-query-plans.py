"""Measure authenticated owner/cursor plans on the isolated readiness DB only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

owner = str(uuid.UUID(hashlib.md5(b'load-user-1').hexdigest()))
phase = sys.argv[1]
assert phase in ('before', 'after')
base = ['psql', '-X', '-qAt', '-h', '/tmp', '-p', '55440', '-d', 'wear_full_replay', '-v', 'ON_ERROR_STOP=1']
results = {}
for table in ['clothing_items', 'wear_logs']:
    for page, cursor in [('first', ''), ('later', "and id < '80000000-0000-4000-8000-000000000000'::uuid")]:
        query = f"""begin; set local role authenticated;
          set local request.jwt.claim.sub='{owner}';
          explain (analyze,buffers,format json)
          select * from public.{table} where user_id='{owner}' {cursor} order by id desc limit 250;
          rollback;"""
        response = subprocess.run(base, input=query, text=True, capture_output=True, check=True, timeout=60)
        plan = json.loads(response.stdout)[0]
        results[f'{table}_{page}'] = plan
        def indexes(node):
            return ([node['Index Name']] if 'Index Name' in node else []) + [name for child in node.get('Plans', []) for name in indexes(child)]
        print(table, page, plan['Execution Time'], 'ms', indexes(plan['Plan']))
        assert plan['Plan']['Actual Rows'] == 250
        if phase == 'after':
            expected = f'{table}_owner_cursor_idx'
            assert expected in indexes(plan['Plan']), plan
            assert not any(node.get('Node Type') == 'Sort' for node in plan['Plan'].get('Plans', [])), plan
Path(f'docs/reports/readiness-evidence/cursor-query-plans-{phase}.json').write_text(json.dumps(results, indent=2)+'\n')
