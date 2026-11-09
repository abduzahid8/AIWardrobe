export async function POST(_request: Request) {
  return Response.json({ status: 'ok', handler: 'POST' });
}
export function GET() {
  return Response.json({ status: 'ok', handler: 'GET' });
}
export default {
  fetch(request: Request) {
    if (request.method === 'POST') return POST(request);
    return GET();
  },
};
