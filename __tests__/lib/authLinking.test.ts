import { createAuthLinking } from '../../src/lib/authLinking';

function deferred() {
    let resolve!: () => void;
    const promise = new Promise<void>(done => { resolve = done; });
    return { promise, resolve };
}

function setup(processLink = jest.fn(async (_url: string): Promise<void> => undefined)) {
    let receive!: (event: { url: string }) => void;
    const remove = jest.fn();
    const source = {
        getInitialURL: jest.fn(async (): Promise<string | null> => 'com.aiwardrobe://reset-password?code=synthetic'),
        addEventListener: jest.fn((_type: 'url', callback: typeof receive) => {
            receive = callback;
            return { remove };
        }),
    };
    return { linking: createAuthLinking(source, processLink), source, processLink, remove,
        emit: (url: string) => receive({ url }) };
}

it('holds cold-start navigation until the recovery identity is published', async () => {
    const work = deferred();
    const test = setup(jest.fn((_url: string) => work.promise));
    let delivered = false;
    const result = test.linking.getInitialURL().then(url => { delivered = true; return url; });
    await Promise.resolve();
    expect(test.processLink).toHaveBeenCalledTimes(1);
    expect(delivered).toBe(false);
    work.resolve();
    await expect(result).resolves.toBe('com.aiwardrobe://reset-password?code=synthetic');
});

it('does not process a missing startup link', async () => {
    const test = setup();
    test.source.getInitialURL.mockResolvedValue(null);
    await expect(test.linking.getInitialURL()).resolves.toBeNull();
    expect(test.processLink).not.toHaveBeenCalled();
});

it('holds warm navigation until processing completes and preserves event order', async () => {
    const work = deferred();
    const test = setup(jest.fn().mockImplementationOnce(() => work.promise).mockResolvedValue(undefined));
    const listener = jest.fn();
    test.linking.subscribe(listener);
    test.emit('first'); test.emit('second');
    await Promise.resolve();
    expect(test.processLink.mock.calls).toEqual([['first']]);
    expect(listener).not.toHaveBeenCalled();
    work.resolve();
    for (let i = 0; i < 12; i++) await Promise.resolve();
    expect(test.processLink.mock.calls).toEqual([['first'], ['second']]);
    expect(listener.mock.calls).toEqual([['first'], ['second']]);
});

it('does not navigate or start queued work after the subscription is removed', async () => {
    const work = deferred();
    const test = setup(jest.fn((_url: string) => work.promise));
    const listener = jest.fn();
    const unsubscribe = test.linking.subscribe(listener);
    test.emit('first'); test.emit('second');
    await Promise.resolve();
    unsubscribe(); work.resolve();
    for (let i = 0; i < 12; i++) await Promise.resolve();
    expect(test.remove).toHaveBeenCalledTimes(1);
    expect(test.processLink).toHaveBeenCalledTimes(1);
    expect(listener).not.toHaveBeenCalled();
});

it('omits failed startup processing instead of returning its route', async () => {
    const test = setup(jest.fn().mockRejectedValue(new Error('synthetic')));
    await expect(test.linking.getInitialURL()).resolves.toBeNull();
});

it('does not route rejected warm work and admits the next event', async () => {
    const test = setup(jest.fn().mockRejectedValueOnce(new Error('synthetic')).mockResolvedValue(undefined));
    const listener = jest.fn();
    test.linking.subscribe(listener);
    test.emit('failed'); test.emit('valid');
    for (let i = 0; i < 12; i++) await Promise.resolve();
    expect(listener.mock.calls).toEqual([['valid']]);
});
