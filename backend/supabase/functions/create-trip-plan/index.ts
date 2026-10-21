import { enforceRateLimit } from '../_shared/rateLimit.ts'

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const WEATHER_API_KEY = Deno.env.get('OPENWEATHER_API_KEY') || ''
const WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5'

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    const rateLimitResponse = await enforceRateLimit(req, { feature: 'create-trip-plan', limit: 5, headers: corsHeaders })
    if (rateLimitResponse) return rateLimitResponse


    try {
        const supabaseClient = createClient(
            Deno.env.get('SUPABASE_URL') ?? '',
            Deno.env.get('SUPABASE_ANON_KEY') ?? '',
            { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
        )

        const { destination, startDate, endDate, occasions } = await req.json()

        // Get User ID from Auth
        const {
            data: { user },
        } = await supabaseClient.auth.getUser()

        if (!user) {
            throw new Error("User not authenticated")
        }

        // Admission control has already verified server-owned entitlements.
        if (typeof destination !== 'string' || !destination.trim() || destination.length > 200 ||
            !validTripDate(startDate) || !validTripDate(endDate) || endDate < startDate ||
            (Date.parse(endDate) - Date.parse(startDate)) / 86400000 > 30 ||
            (occasions !== undefined && (!Array.isArray(occasions) || !occasions.length || occasions.length > 8 ||
              occasions.some((value: unknown) => typeof value !== 'string' || !value.trim() || value.length > 60)))) {
            return tripError('Invalid destination, trip dates or occasions', 400)
        }

        const weatherData = await fetchWeatherForTrip(destination.trim(), startDate, endDate);

        // 2. Get user's wardrobe from Supabase
        // Note: 'clothing_items' table is expected to have 'type', 'category', 'item_type' etc. 
        // Adapting to whatever schema exists.
        const { data: wardrobe, error: dbError } = await supabaseClient
            .from('clothing_items')
            .select('*')
            .eq('user_id', user.id);

        if (dbError) throw dbError;

        if (!wardrobe || wardrobe.length === 0) {
            // Return error or maybe success with empty lists?
            // Legacy returned 400.
            return new Response(
                JSON.stringify({ error: 'Empty wardrobe', message: 'Please add clothing items to your wardrobe first' }),
                { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
            )
        }

        // 3. Generate packing list and outfits
        // Map Supabase snake_case to logic's expected format if needed
        // Logic below assumes camelCase or specific property access. 
        // Let's normalize wardrobe items
        const normalizedWardrobe = wardrobe.map(item => ({
            _id: item.id,
            itemType: item.type || item.category || 'unknown',
            material: item.material || '',
            style: item.style || '',
            image: item.image_url || item.image,
            color: item.color || 'unknown',
            ...item
        }));

        const tripPlan = generateTripPlan({
            wardrobe: normalizedWardrobe,
            weather: weatherData.forecasts,
            startDate,
            endDate,
            occasions: occasions || ['casual']
        });

        return new Response(
            JSON.stringify({
                success: true,
                tripId: Date.now().toString(),
                destination: weatherData.city,
                weather: weatherData.forecasts,
                packingList: tripPlan.packingList,
                outfitsByDay: tripPlan.outfitsByDay,
                missingItems: tripPlan.missingItems,
                stats: {
                    totalItems: tripPlan.packingList.length,
                    totalOutfits: tripPlan.outfitsByDay.reduce((total: number, day: any) => total + day.outfits.length, 0),
                    daysPlanned: tripPlan.outfitsByDay.length
                }
            }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
        )

    } catch (error: any) {
        return new Response(
            JSON.stringify({ error: error instanceof TripError ? error.message : error instanceof SyntaxError ? 'Invalid JSON request' : 'Trip planning failed' }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: error instanceof TripError ? error.status : error instanceof SyntaxError ? 400 : 500 }
        )
    }
})

// --- Logic Helpers ---

class TripError extends Error {
    constructor(message: string, public status: number) { super(message) }
}
function tripError(error: string, status: number) {
    return new Response(JSON.stringify({ error }), { status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })
}
function validTripDate(value: unknown): value is string {
    if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
    const parsed = new Date(`${value}T00:00:00Z`)
    return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
}

async function fetchWeatherForTrip(city: string, startDate: string, endDate: string) {
    if (!WEATHER_API_KEY) throw new TripError('Weather service is not configured', 503)
    try {
        const geoUrl = new URL(`${WEATHER_API_URL}/weather`)
        geoUrl.search = new URLSearchParams({ q: city, appid: WEATHER_API_KEY, units: 'metric' }).toString()
        const geoRes = await fetch(geoUrl, { signal: AbortSignal.timeout(10000) })
        if (!geoRes.ok) throw new TripError('Destination weather lookup failed', geoRes.status === 404 ? 422 : 503)
        const geo = await geoRes.json()
        const { lat, lon } = geo.coord || {}
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) throw new TripError('Invalid weather location response', 503)
        const forecastUrl = new URL(`${WEATHER_API_URL}/forecast`)
        forecastUrl.search = new URLSearchParams({ lat: String(lat), lon: String(lon), appid: WEATHER_API_KEY, units: 'metric', cnt: '40' }).toString()
        const forecastRes = await fetch(forecastUrl, { signal: AbortSignal.timeout(10000) })
        if (!forecastRes.ok) throw new TripError('Weather forecast is unavailable', 503)
        const forecast = await forecastRes.json()
        if (!Array.isArray(forecast.list) || !forecast.list.length) throw new TripError('Weather forecast is unavailable', 503)
        if (!Number.isFinite(forecast.city?.timezone)) throw new TripError('Forecast timezone is unavailable', 503)
        const forecasts: any[] = parseDailyForecasts(forecast.list, forecast.city.timezone)
            .filter((day: any) => day.date >= startDate && day.date <= endDate)
        const expectedDays = (Date.parse(endDate) - Date.parse(startDate)) / 86400000 + 1
        if (forecasts.length !== expectedDays) throw new TripError('Forecast does not cover the requested trip dates', 422)
        return { city: geo.name || city, country: geo.sys?.country || '', forecasts }
    } catch (error) {
        if (error instanceof TripError) throw error
        throw new TripError('Weather service is unavailable', 503)
    }
}

function parseDailyForecasts(forecastList: any[], timezoneOffsetSec = 0) {
    const dailyData: any = {};

    forecastList.forEach(item => {
        if (!Number.isFinite(item.dt) || !Number.isFinite(item.main?.temp) || !item.weather?.[0]?.main) throw new TripError('Invalid forecast response', 503)
        const date = new Date((item.dt + timezoneOffsetSec) * 1000).toISOString().slice(0, 10)
        if (!dailyData[date]) {
            dailyData[date] = { date, temps: [], conditions: [], descriptions: [], icons: [] };
        }
        dailyData[date].temps.push(item.main.temp);
        dailyData[date].conditions.push(item.weather[0].main);
        dailyData[date].descriptions.push(item.weather[0].description);
        dailyData[date].icons.push(item.weather[0].icon);
    });

    return Object.values(dailyData).map((day: any) => ({
        date: day.date,
        tempHigh: Math.round(Math.max(...day.temps)),
        tempLow: Math.round(Math.min(...day.temps)),
        tempAvg: Math.round(day.temps.reduce((a: number, b: number) => a + b) / day.temps.length),
        condition: mostCommon(day.conditions),
        description: mostCommon(day.descriptions),
        icon: mostCommon(day.icons)
    }));
}

function mostCommon(arr: any[]) {
    const counts: any = {};
    arr.forEach(item => counts[item] = (counts[item] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

function generateTripPlan({ wardrobe, weather, startDate, endDate, occasions }: any) {
    const outfitsByDay: any[] = [];
    const usedItems: any = {};
    const packingList = new Set();
    const missingItems: any[] = [];

    weather.forEach((dayWeather: any, index: number) => {
        const dayOutfits: any[] = [];
        occasions.forEach((occasion: string) => {
            const outfit = allocateOutfit({
                wardrobe,
                weather: dayWeather,
                occasion,
                usedItems,
                dayIndex: index
            });

            if (outfit.items.length > 0) {
                dayOutfits.push({ occasion, items: outfit.items });
                outfit.items.forEach((item: any) => {
                    packingList.add(item._id.toString());
                    usedItems[item._id.toString()] = (usedItems[item._id.toString()] || 0) + 1;
                });
            } else {
                missingItems.push({
                    occasion,
                    date: dayWeather.date,
                    reason: `No suitable items for ${occasion} in ${dayWeather.condition} weather`
                });
            }
        });
        outfitsByDay.push({ date: dayWeather.date, weather: dayWeather, outfits: dayOutfits });
    });

    const packingListArray = Array.from(packingList).map(itemId => {
        const item = wardrobe.find((w: any) => w._id.toString() === itemId);
        return {
            ...item,
            uses: usedItems[itemId as string]
        };
    });

    return { packingList: packingListArray, outfitsByDay, missingItems };
}

function allocateOutfit({ wardrobe, weather, occasion, usedItems, dayIndex }: any) {
    const selectedItems = [];
    const suitable = wardrobe.filter((item: any) => isWeatherAppropriate(item, weather) && isOccasionAppropriate(item, occasion));

    if (suitable.length === 0) return { items: [] };

    const dress = suitable.filter((item: any) => {
        const type = String(item.itemType || '').toLowerCase();
        const category = String(item.category || '').toLowerCase();
        return /^(dress|dresses|jumpsuit|jumpsuits|romper|rompers)$/.test(category) ||
          (/\b(dress|jumpsuit|romper)\b/.test(type) && !/\bdress[ -]+(shirt|shoes?|pants|trousers)\b/.test(type));
    }).sort((a: any, b: any) => (usedItems[String(a._id)] || 0) - (usedItems[String(b._id)] || 0))[0];
    if (dress) selectedItems.push(dress);
    const top = selectBestItem(suitable, ['shirt', 'tshirt', 'blouse', 'sweater', 'hoodie', 'top'], usedItems, dayIndex);
    if (top && !dress) selectedItems.push(top);

    const bottom = selectBestItem(suitable, ['pants', 'jeans', 'shorts', 'skirt', 'trousers', 'bottom'], usedItems, dayIndex);
    if (bottom && !dress) selectedItems.push(bottom);

    const shoes = selectBestItem(suitable, ['shoe', 'sneaker', 'boot', 'sandal', 'heel'], usedItems, dayIndex);
    if (shoes) selectedItems.push(shoes);

    let hasLayer = false;
    if (weather.tempHigh < 15) {
        const jacket = selectBestItem(suitable, ['jacket', 'coat', 'blazer'], usedItems, dayIndex);
        if (jacket) { selectedItems.push(jacket); hasLayer = true; }
    }

    if ((!dress && (!top || !bottom)) || !shoes || (weather.tempHigh < 15 && !hasLayer)) return { items: [] };
    return { items: selectedItems };
}

function isWeatherAppropriate(item: any, weather: any) {
    const temp = weather.tempAvg;
    const type = item.itemType?.toLowerCase() || '';
    const material = item.material?.toLowerCase() || '';
    const evidence = `${type} ${item.name || ''} ${item.sub_category || ''}`.toLowerCase();
    if (temp < 10 && /\b(shorts?|sandals?|tank|swimwear|flip[ -]?flops?)\b/.test(evidence)) return false;
    if (temp > 25 && /\b(overcoat|parka|puffer|heavy wool|thermal)\b/.test(`${evidence} ${material}`)) return false;

    if (temp > 25) {
        if (['shorts', 'tank', 'sandals', 'tshirt', 'top'].some(t => type.includes(t))) return true;
        if (material.includes('linen') || material.includes('cotton')) return true;
    }
    if (temp < 10) {
        if (['sweater', 'jacket', 'coat', 'boots', 'jeans', 'hoodie'].some(t => type.includes(t))) return true;
        if (material.includes('wool') || material.includes('fleece')) return true;
    }
    return true; // Moderate
}

function isOccasionAppropriate(item: any, occasion: string) {
    const type = item.itemType?.toLowerCase() || '';
    const style = item.style?.toLowerCase() || '';
    if (occasion === 'formal' || occasion === 'business') {
        return ['blazer', 'suit', 'dress', 'button', 'formal', 'trousers'].some(k => type.includes(k) || style.includes(k));
    }
    if (occasion === 'beach') {
        return ['swimwear', 'shorts', 'sandals', 'hat', 'sunglasses'].some(k => type.includes(k));
    }
    return true; // Casual default
}

function selectBestItem(wardrobe: any[], types: string[], usedItems: any, dayIndex: number) {
    const candidates = wardrobe.filter(item => types.some(type => item.itemType?.toLowerCase().includes(type)));
    if (candidates.length === 0) return null;

    // Sort by least worn
    candidates.sort((a, b) => {
        const aUses = usedItems[a._id.toString()] || 0;
        const bUses = usedItems[b._id.toString()] || 0;
        return aUses - bUses;
    });
    return candidates[0];
}
