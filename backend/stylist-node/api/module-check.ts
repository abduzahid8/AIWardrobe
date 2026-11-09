/**
 * Temporary PREVIEW-ONLY runtime module diagnostic. No tokens, secrets, user
 * data, product identifiers, or request input are read. Remove before release.
 */
export default {
  async fetch(): Promise<Response> {
    try {
      const mod = await import('../transport/signature.ts');
      return Response.json({
        status: 'ok',
        available: typeof mod.signStylistTransport === 'function',
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
