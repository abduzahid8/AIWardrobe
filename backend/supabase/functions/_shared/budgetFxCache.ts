import { BudgetFxError, type BudgetFxSnapshot, parseCbuBudgetRates, uzbekDateKey, validateBudgetFxSnapshot } from './budgetFx.ts';

// One refresh per isolate; the persistent shared row also survives cold starts.
let refresh: Promise<BudgetFxSnapshot> | null = null;
export async function currentBudgetRates(admin: any, now = Date.now(), request: typeof fetch = fetch): Promise<BudgetFxSnapshot> {
  const readController = new AbortController();
  let readTimer: ReturnType<typeof setTimeout> | undefined;
  let cached;
  try {
    const query = admin.from('budget_exchange_rates').select('snapshot').order('fetched_at', {ascending: false}).limit(1).abortSignal(readController.signal).maybeSingle();
    cached = await Promise.race([query, new Promise<never>((_, reject) => { readTimer = setTimeout(() => { readController.abort(); reject(new BudgetFxError()); }, 5000); })]);
  } finally { if (readTimer) clearTimeout(readTimer); }
  const {data, error} = cached;
  if (error) throw new BudgetFxError();
  if (data?.snapshot) {
    try { return validateBudgetFxSnapshot(data.snapshot, now); } catch { /* Refresh an expired or invalid snapshot. */ }
  }
  if (refresh) return refresh;
  const pending = (async () => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        (async () => {
          const url = `https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/${uzbekDateKey(now)}/`;
          const response = await request(url, {signal: controller.signal, redirect: 'error', headers: {Accept: 'application/json'}});
          if (!response.ok || Number(response.headers.get('content-length') || 0) > 262144) throw new BudgetFxError();
          const reader = response.body?.getReader();
          if (!reader) throw new BudgetFxError();
          const chunks: Uint8Array[] = []; let size = 0;
          try {
            for (;;) {
              const {done, value} = await reader.read(); if (done) break;
              size += value.byteLength;
              if (size > 262144) { await reader.cancel(); throw new BudgetFxError(); }
              chunks.push(value);
            }
          } finally { reader.releaseLock(); }
          const bytes = new Uint8Array(size); let offset = 0;
          for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
          const snapshot = parseCbuBudgetRates(JSON.parse(new TextDecoder().decode(bytes)), now);
          if (controller.signal.aborted) throw new BudgetFxError();
          const saved = await admin.from('budget_exchange_rates').upsert({id: snapshot.id, as_of: snapshot.asOf, fetched_at: snapshot.fetchedAt, snapshot}, {onConflict: 'id'}).abortSignal(controller.signal);
          if (saved.error || controller.signal.aborted) throw new BudgetFxError();
          return snapshot;
        })(),
        new Promise<never>((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new BudgetFxError()); }, 8000); }),
      ]);
    } catch { throw new BudgetFxError(); }
    finally { if (timer) clearTimeout(timer); }
  })();
  refresh = pending;
  try { return await pending; } finally { if (refresh === pending) refresh = null; }
}

/** A client may name a trusted stored snapshot, never supply its conversion values. */
export async function budgetRatesById(admin: any, id: string, now = Date.now()): Promise<BudgetFxSnapshot> {
  if (typeof id !== 'string' || !id.startsWith('CBU:') || id.length > 200) throw new BudgetFxError();
  const controller = new AbortController(); let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const query = admin.from('budget_exchange_rates').select('snapshot').eq('id', id).abortSignal(controller.signal).maybeSingle();
    const result: any = await Promise.race([query, new Promise<never>((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new BudgetFxError()); }, 5000); })]);
    if (result.error || !result.data?.snapshot) throw new BudgetFxError();
    const snapshot = validateBudgetFxSnapshot(result.data.snapshot, now);
    if (snapshot.id !== id) throw new BudgetFxError();
    return snapshot;
  } finally { if (timer) clearTimeout(timer); }
}
