/** Static diagnostic route. No credentials, request data, or user metadata. */
export default {
  fetch(): Response {
    return Response.json({ status: 'ok' }, { status: 200, headers: { 'Cache-Control': 'no-store' } });
  },
};
