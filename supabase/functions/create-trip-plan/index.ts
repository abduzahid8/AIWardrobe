
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

        console.log(`Creating trip plan for user ${user.id} to ${destination}`);

        // 1. Fetch weather forecast (or dummy)
        const weatherData = await fetchWeatherForTrip(destination, startDate, endDate);

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
            image: item.image,
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
                    totalOutfits: tripPlan.outfitsByDay.length,
                    daysPlanned: tripPlan.outfitsByDay.length
                }
            }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
        )

    } catch (error: any) {
        return new Response(
            JSON.stringify({ error: error.message }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
        )
    }
})

// --- Logic Helpers ---

async function fetchWeatherForTrip(city: string, startDate: string, endDate: string) {
    if (!WEATHER_API_KEY) {
        return {
            city,
            country: 'Unknown',
            forecasts: generateDummyWeather(startDate, endDate)
        };
    }

    try {
        // Get coordinates
        const geoRes = await fetch(`${WEATHER_API_URL}/weather?q=${city}&appid=${WEATHER_API_KEY}&units=metric`);
        if (!geoRes.ok) throw new Error('Weather API Error');
        const geoData = await geoRes.json();
        const { lat, lon, name, sys } = geoData;

        // Get forecast
        const forecastRes = await fetch(`${WEATHER_API_URL}/forecast?lat=${lat}&lon=${lon}&appid=${WEATHER_API_KEY}&units=metric&cnt=40`);
        if (!forecastRes.ok) throw new Error('Forecast API Error');
        const forecastData = await forecastRes.json();

        return {
            city: name,
            country: sys.country,
            forecasts: parseDailyForecasts(forecastData.list)
        };
    } catch (error) {
        console.error('Weather fetch error:', error);
        return {
            city,
            country: 'Unknown',
            forecasts: generateDummyWeather(startDate, endDate)
        };
    }
}

function parseDailyForecasts(forecastList: any[]) {
    const dailyData: any = {};

    forecastList.forEach(item => {
        const date = item.dt_txt.split(' ')[0]; // YYYY-MM-DD
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

function generateDummyWeather(startDate: string, endDate: string) {
    const forecasts = [];
    const start = new Date(startDate);
    const end = new Date(endDate);
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
        forecasts.push({
            date: d.toISOString().split('T')[0],
            tempHigh: 25,
            tempLow: 18,
            tempAvg: 22,
            condition: 'Clear',
            description: 'clear sky',
            icon: '01d'
        });
    }
    return forecasts;
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

    // Select items (Logic ported)
    const top = selectBestItem(suitable, ['shirt', 'tshirt', 'blouse', 'sweater', 'hoodie', 'top'], usedItems, dayIndex);
    if (top) selectedItems.push(top);

    const bottom = selectBestItem(suitable, ['pants', 'jeans', 'shorts', 'skirt', 'trousers'], usedItems, dayIndex);
    if (bottom) selectedItems.push(bottom);

    const shoes = selectBestItem(suitable, ['shoes', 'sneakers', 'boots', 'sandals', 'heels'], usedItems, dayIndex);
    if (shoes) selectedItems.push(shoes);

    if (weather.tempHigh < 15) {
        const jacket = selectBestItem(suitable, ['jacket', 'coat', 'blazer'], usedItems, dayIndex);
        if (jacket) selectedItems.push(jacket);
    }

    return { items: selectedItems };
}

function isWeatherAppropriate(item: any, weather: any) {
    const temp = weather.tempAvg;
    const type = item.itemType?.toLowerCase() || '';
    const material = item.material?.toLowerCase() || '';

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
