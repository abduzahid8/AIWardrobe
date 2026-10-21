/**
 * Serializes complete authentication operations, including their cleanup.
 * Callers must await every identity-changing side effect inside the operation.
 * A UI deadline may stop waiting, but must never release this queue early.
 */
export function createAuthOperationQueue() {
    let tail: Promise<void> = Promise.resolve();
    return {
        run<T>(operation: (retain: <R>(work: Promise<R>) => Promise<R>) => Promise<T>): Promise<T> {
            const retained: Promise<unknown>[] = [];
            const retain = <R>(work: Promise<R>): Promise<R> => {
                retained.push(work.then(() => undefined, () => undefined));
                return work;
            };
            const result = tail.then(() => operation(retain));
            // A failed operation must not poison subsequent authentication.
            tail = result.then(() => undefined, () => undefined).then(async () => {
                // UI deadlines can resolve the action while native/SDK cleanup
                // is still running. Keep later identities behind those writes.
                await Promise.all(retained);
            });
            return result;
        },
    };
}
