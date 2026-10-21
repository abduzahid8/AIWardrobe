"""Local-only concurrent wear logging regression. Requires the test schema/migration."""
import concurrent.futures
import json
import subprocess
import uuid

BASE = ['psql', '-h', '/tmp', '-p', '55440', '-d', 'wear_counter_test', '-X', '-qAt', '-v', 'ON_ERROR_STOP=1']
owner, first, second = [str(uuid.uuid4()) for _ in range(3)]


def sql(query):
    result = subprocess.run(BASE, input=query, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def operation(index):
    ids = [first, second] if index % 2 else [second, first]
    log_id = str(uuid.uuid4())
    values = f"'{log_id}','{owner}','{json.dumps(ids)}','2026-10-16','2026-10-16T12:00:00Z'"
    sql(f"""begin; set local role authenticated;
      set local request.jwt.claim.sub='{owner}';
      insert into public.wear_logs(id,user_id,item_ids,date,created_at) values ({values});
      commit;""")
    return log_id


try:
    sql(f"""insert into auth.users(id,email,raw_user_meta_data)
      values('{owner}','{owner}@example.invalid','{{"username":"{owner}"}}');
      insert into public.clothing_items(id,user_id) values('{first}','{owner}'),('{second}','{owner}');""")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        logs = list(pool.map(operation, range(120)))
    counts = sql(f"select string_agg(wear_count::text,',' order by id) from public.clothing_items where user_id='{owner}';")
    assert counts == '120,120', counts

    def delete(log_id):
        sql(f"begin; set local role authenticated; set local request.jwt.claim.sub='{owner}'; delete from public.wear_logs where id='{log_id}'; commit;")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(delete, logs))
    counts = sql(f"select string_agg(wear_count::text,',' order by id) from public.clothing_items where user_id='{owner}';")
    timestamps = sql(f"select count(*) from public.clothing_items where user_id='{owner}' and last_worn_at is not null;")
    assert counts == '0,0', counts
    assert timestamps == '0', timestamps
    print(json.dumps({'workers': 12, 'insertTransactions': 120, 'deleteTransactions': 120,
                      'insertCounts': [120, 120], 'finalCounts': [0, 0], 'remainingTimestamps': 0}))
finally:
    sql(f"delete from auth.users where id='{owner}';")
