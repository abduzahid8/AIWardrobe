/**
 * Liveness is intentionally separate from readiness. A Vercel preview can
 * show READY while the original Stylist is unable to authenticate to
 * Supabase or call NVIDIA. Do not expose individual secret names or values.
 */
const REQUIRED_SERVER_ENV = [
  'STYLIST_NODE_SHARED_SECRET',
  'SUPABASE_URL',
  'SUPABASE_ANON_KEY',
  'SUPABASE_SERVICE_ROLE_KEY',
  'NVIDIA_API_KEY',
] as const;

export default {
  fetch(): Response {
    const configured = REQUIRED_SERVER_ENV.every((key) => {
      const value = process.env[key];
      return typeof value === 'string' && value.trim().length >= (key === 'STYLIST_NODE_SHARED_SECRET' ? 32 : 1);
    });
    return Response.json(
      { status: 'ok', computeConfigured: configured },
      { status: 200, headers: { 'Cache-Control': 'no-store' } }
    );
  },
};
