// Historical Preview diagnostic. Paid judging now runs in the isolated local
// pilot, with hash-bound media and a single-use launch claim.
export default {
  fetch(request: Request): Response {
    return Response.json({ ok: false, code: 'vision_diagnostic_retired' }, {
      status: request.method === 'POST' ? 410 : 405,
      headers: { 'Cache-Control': 'no-store' },
    });
  },
};
