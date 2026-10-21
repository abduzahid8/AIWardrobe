interface StorageBackend {
  getItem(name: string): Promise<string | null>;
  setItem(name: string, value: string): Promise<unknown>;
  removeItem(name: string): Promise<unknown>;
}
interface PendingWrite {
  value: string;
  timer: ReturnType<typeof setTimeout>;
  waiters: Array<{ resolve: () => void; reject: (error: unknown) => void }>;
}

/** Coalesced writes with settled callers and ordered removal. */
export function createDebouncedStorage(backend: StorageBackend, delay = 500, onError: (error: unknown) => void = () => undefined) {
  const pending = new Map<string, PendingWrite>();
  const chains = new Map<string, Promise<void>>();
  const versions = new Map<string, number>();
  const failures = new Map<string, unknown>();
  function run(name: string, action: () => Promise<unknown>): Promise<void> {
    const operation = (chains.get(name) ?? Promise.resolve()).then(action).then(() => {
      failures.delete(name);
    }, error => {
      failures.set(name, error);
      throw error;
    });
    const settled = operation.catch(() => undefined);
    chains.set(name, settled);
    void settled.then(() => { if (chains.get(name) === settled) chains.delete(name); });
    return operation;
  }
  function commit(name: string): Promise<void> {
    const write = pending.get(name);
    if (!write) return chains.get(name) ?? Promise.resolve();
    clearTimeout(write.timer);
    pending.delete(name);
    const operation = run(name, () => backend.setItem(name, write.value));
    void operation.then(() => write.waiters.forEach(waiter => waiter.resolve()), error => write.waiters.forEach(waiter => waiter.reject(error)));
    return operation;
  }
  return {
    async getItem(name: string): Promise<string | null> {
      const version = versions.get(name);
      // Capture the read at invocation. An extra await before choosing the
      // value could hydrate an older intermediate optimistic snapshot later.
      const queued = pending.get(name);
      const inFlight = chains.get(name);
      const value = await (queued ? Promise.resolve(queued.value)
        : inFlight ? inFlight.then(() => backend.getItem(name)) : backend.getItem(name));
      return version === versions.get(name) ? value : null;
    },
    setItem(name: string, value: string): Promise<void> {
      const previous = pending.get(name);
      if (previous) clearTimeout(previous.timer);
      const promise = new Promise<void>((resolve, reject) => {
        const waiters = previous?.waiters ?? [];
        waiters.push({ resolve, reject });
        const timer = setTimeout(() => { void commit(name).catch(() => undefined); }, delay);
        pending.set(name, { value, timer, waiters });
      });
      // Background Zustand callers may not await; explicit callers still see rejection.
      void promise.catch(onError);
      return promise;
    },
    removeItem(name: string): Promise<void> {
      versions.set(name, (versions.get(name) ?? 0) + 1);
      const write = pending.get(name);
      if (write) {
        clearTimeout(write.timer); pending.delete(name);
        write.waiters.forEach(waiter => waiter.resolve());
      }
      return run(name, () => backend.removeItem(name));
    },
    async flush(): Promise<void> {
      // Drain all work even if one key fails. Internal continuation promises
      // intentionally swallow errors, so retain failures separately until a
      // successful write/removal for that key establishes its storage state.
      await Promise.allSettled([...pending.keys()].map(commit));
      await Promise.allSettled(chains.values());
      if (failures.size) throw failures.values().next().value;
    },
  };
}
