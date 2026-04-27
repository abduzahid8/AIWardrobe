// Weather Edge Function
// Replaces the /weather/coords endpoint on the deprecated Express server.
// Reads OPENWEATHER_API_KEY from Supabase secrets.

// deno-lint-ignore-file no-explicit-any
import 'jsr:@supabase/functions-js/edge-runtime.d.ts';

const OPENWEATHER_BASE = 'https://api.openweathermap.org/data/2.5';

interface WeatherBody {
  lat?: number;
  lon?: number;
  units?: 'metric' | 'imperial';
}

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'method_not_allowed' }), {
      status: 405,
      headers: { 'content-type': 'application/json' },
    });
  }

  let body: WeatherBody;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: 'invalid_json' }), {
      status: 400,
      headers: { 'content-type': 'application/json' },
    });
  }

  const { lat, lon, units = 'metric' } = body;
  if (typeof lat !== 'number' || typeof lon !== 'number') {
    return new Response(JSON.stringify({ error: 'lat_and_lon_required' }), {
      status: 400,
      headers: { 'content-type': 'application/json' },
    });
  }

  const key = Deno.env.get('OPENWEATHER_API_KEY');
  if (!key) {
    return new Response(JSON.stringify({ error: 'weather_not_configured' }), {
      status: 503,
      headers: { 'content-type': 'application/json' },
    });
  }

  const url = `${OPENWEATHER_BASE}/weather?lat=${lat}&lon=${lon}&units=${units}&appid=${key}`;
  const res = await fetch(url);
  if (!res.ok) {
    return new Response(
      JSON.stringify({ error: 'upstream_failed', status: res.status }),
      { status: 502, headers: { 'content-type': 'application/json' } },
    );
  }
  const data = await res.json() as any;

  return new Response(
    JSON.stringify({
      temp: Math.round(data.main?.temp ?? 0),
      temp_min: data.main?.temp_min,
      temp_max: data.main?.temp_max,
      condition: data.weather?.[0]?.main ?? 'Unknown',
      description: data.weather?.[0]?.description ?? '',
      icon: data.weather?.[0]?.icon,
      city: data.name,
    }),
    { headers: { 'content-type': 'application/json' } },
  );
});
