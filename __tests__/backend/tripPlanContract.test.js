/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/create-trip-plan/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const day = date => Date.parse(`${date}T12:00:00Z`) / 1000;
function load({ key = 'weather-key', fetchImpl, wardrobe = [] } = {}) {
  let handler;
  const mockFetch = jest.fn(fetchImpl);
  const client = { auth: { getUser: async () => ({ data: { user: { id: 'owner' } } }) }, from: () => ({ select: () => ({ eq: async () => ({ data: wardrobe, error: null }) }) }) };
  new Function('require', 'exports', 'Deno', 'fetch', compiled)(name => {
    if (name.includes('rateLimit')) return { enforceRateLimit: async () => null };
    if (name.includes('server.ts')) return { serve: fn => { handler = fn; } };
    return { createClient: () => client };
  }, {}, { env: { get: name => name === 'OPENWEATHER_API_KEY' ? key : 'configuration' } }, mockFetch);
  return { handler, mockFetch };
}
const trip = { destination: 'Paris & units=imperial', startDate: '2026-10-15', endDate: '2026-10-16', occasions: ['casual', 'travel'] };
const request = body => new Request('https://example.test', { method: 'POST', body: JSON.stringify(body) });
const row = (date, temp = 20) => ({ dt: day(date), main: { temp }, weather: [{ main: 'Clear', description: 'clear sky', icon: '01d' }] });
const weatherFetch = dates => async url => new Response(JSON.stringify(url.pathname.endsWith('/weather') ? { coord: { lat: 48.85, lon: 2.35 }, name: 'Paris', sys: { country: 'FR' } } : { city: { timezone: 7200 }, list: dates.map(date => row(date)) }));

test('missing weather credentials return unavailable, never invented clear skies', async () => {
  const { handler, mockFetch } = load({ key: '' });
  const result = await handler(request(trip));
  expect(result.status).toBe(503);
  expect((await result.json()).weather).toBeUndefined();
  expect(mockFetch).not.toHaveBeenCalled();
});
test.each([
  { startDate: '2026-02-30' }, { endDate: '2026-10-14' }, { occasions: 'casual' }, { occasions: [] }, { destination: {} },
])('rejects invalid trip input %j before weather work', async overrides => {
  const { handler, mockFetch } = load();
  expect((await handler(request({ ...trip, ...overrides }))).status).toBe(400);
  expect(mockFetch).not.toHaveBeenCalled();
});
test('weather failures do not produce fake successful plans', async () => {
  const { handler } = load({ fetchImpl: async () => new Response('{}', { status: 503 }) });
  expect((await handler(request(trip))).status).toBe(503);
});
test('rejects dates outside returned forecast coverage', async () => {
  const { handler } = load({ fetchImpl: weatherFetch(['2026-10-15']) });
  expect((await handler(request(trip))).status).toBe(422);
});
test('uses provider coordinates, encodes city text, filters dates and counts actual outfits', async () => {
  const wardrobe = ['top', 'bottom', 'shoes'].map((type, i) => ({ id: String(i), type, image_url: `https://example.test/${i}.jpg` }));
  const { handler, mockFetch } = load({ wardrobe, fetchImpl: weatherFetch(['2026-10-14', '2026-10-15', '2026-10-16', '2026-10-17']) });
  const result = await handler(request(trip));
  expect(result.status).toBe(200);
  const body = await result.json();
  expect(body.weather.map(row => row.date)).toEqual(['2026-10-15', '2026-10-16']);
  expect(body.stats).toEqual({ totalItems: 3, totalOutfits: 4, daysPlanned: 2 });
  expect(mockFetch.mock.calls[0][0].searchParams.get('q')).toBe(trip.destination);
  expect(mockFetch.mock.calls[1][0].searchParams.get('lat')).toBe('48.85');
  expect(body.packingList.every(item => item.image)).toBe(true);
});
test('does not label a single shirt as a complete outfit', async () => {
  const { handler } = load({ wardrobe: [{ id: 'shirt', type: 'top' }], fetchImpl: weatherFetch(['2026-10-15', '2026-10-16']) });
  const body = await (await handler(request(trip))).json();
  expect(body.stats.totalOutfits).toBe(0);
  expect(body.missingItems).toHaveLength(4);
});

test('a dress and shoes form a complete one-piece outfit without invented trousers', async () => {
  const { handler } = load({ wardrobe: [{ id: 'dress', type: 'maxi dress', category: 'dress' }, { id: 'shoes', type: 'shoes' }], fetchImpl: weatherFetch(['2026-10-15', '2026-10-16']) });
  const body = await (await handler(request(trip))).json();
  expect(body.stats.totalOutfits).toBe(4);
  expect(body.packingList.map(item => item.id).sort()).toEqual(['dress', 'shoes']);
});
