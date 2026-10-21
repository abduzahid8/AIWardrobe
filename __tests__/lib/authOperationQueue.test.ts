import { createAuthOperationQueue } from '../../src/lib/authOperationQueue';

function deferred() {
    let resolve!: () => void;
    const promise = new Promise<void>(done => { resolve = done; });
    return { promise, resolve };
}

it('keeps later authentication behind outstanding cleanup, even after the caller stops waiting', async () => {
    const queue = createAuthOperationQueue(), cleanup = deferred(), started = deferred();
    const events: string[] = [];
    const old = queue.run(async () => { events.push('old'); started.resolve(); await cleanup.promise; events.push('cleanup'); });
    await started.promise;
    await Promise.race([old, Promise.resolve('caller deadline')]);
    const newer = queue.run(async () => { events.push('new'); });
    await Promise.resolve();
    expect(events).toEqual(['old']);
    cleanup.resolve();
    await Promise.all([old, newer]);
    expect(events).toEqual(['old', 'cleanup', 'new']);
});

it('continues after rejection while preserving the error for its caller', async () => {
    const queue = createAuthOperationQueue();
    const failure = new Error('synthetic operation failure');
    const failed = queue.run(async () => { throw failure; });
    const next = queue.run(async () => 'next account');
    await expect(failed).rejects.toBe(failure);
    await expect(next).resolves.toBe('next account');
});

it('retains separately registered native work after the action resolves', async () => {
    const queue = createAuthOperationQueue(), native = deferred();
    await queue.run(async retain => { retain(native.promise); });
    const started = jest.fn();
    const next = queue.run(async () => { started(); });
    await Promise.resolve();
    expect(started).not.toHaveBeenCalled();
    native.resolve(); await next;
    expect(started).toHaveBeenCalledTimes(1);
});
