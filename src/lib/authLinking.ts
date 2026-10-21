type LinkSource = {
    getInitialURL: () => Promise<string | null>;
    addEventListener: (type: 'url', listener: (event: { url: string }) => void) => { remove: () => void };
};

/** Publish routes only after the link's authentication operation has settled. */
export function createAuthLinking(source: LinkSource, processLink: (url: string) => Promise<void>) {
    return {
        async getInitialURL(): Promise<string | null> {
            try {
                const url = await source.getInitialURL();
                if (!url) return null;
                await processLink(url);
                return url;
            } catch {
                // Keep startup usable when URL retrieval or identity work
                // fails; navigation must not consume the unprocessed URL.
                return null;
            }
        },
        subscribe(listener: (url: string) => void) {
            let active = true;
            let tail = Promise.resolve();
            const subscription = source.addEventListener('url', ({ url }) => {
                tail = tail.then(async () => {
                    if (!active) return;
                    await processLink(url);
                    if (active) listener(url);
                }).catch(() => {
                    // Never navigate with a partially processed identity or
                    // log a URL containing one-use codes or session tokens.
                });
            });
            return () => {
                active = false;
                subscription.remove();
            };
        },
    };
}
