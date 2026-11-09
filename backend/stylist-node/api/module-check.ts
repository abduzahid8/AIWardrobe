/**
 * Temporary PREVIEW diagnostic that verifies both runtime ESM transport and
 * the generated original Stylist kernel are loadable in Vercel. Never call
 * the handler, read secrets, or process user data here. Remove before release.
 */
export default {
  async fetch(): Promise<Response> {
    try {
      const transport = await import('../transport/signature.mjs');
      const stylist = await import('../dist/stylist-bundle.mjs');
      const glob = globalThis as typeof globalThis & {
        __AIW_STYLIST_HANDLER__?: (request: Request) => Promise<Response>;
      };
      void stylist;
      return Response.json({
        status: 'ok',
        signatureLoaded: typeof transport.verifyStylistTransport === 'function',
        originalStylistHandlerLoaded: typeof glob.__AIW_STYLIST_HANDLER__ === 'function',
      }, { headers: { 'Cache-Control': 'no-store' } });
    } catch (error: unknown) {
      const e = error instanceof Error ? error : new Error('unknown');
      return Response.json({
        status: 'module_load_failed',
        errorName: e.name,
        errorCode: 'code' in e ? String(e.code) : undefined,
        detail: e.message.slice(0, 180),
      }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
    }
  },
};
