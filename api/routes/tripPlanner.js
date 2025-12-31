/**
 * Trip Planner Routes
 * Handles trip creation, weather forecasting, and outfit allocation
 */

import express from 'express';
import axios from 'axios';
import ClothingItem from '../models/ClothingItem.js';
import User from '../models/user.js';

const router = express.Router();

// Weather API Configuration (using OpenWeatherMap)
const WEATHER_API_KEY = process.env.OPENWEATHER_API_KEY || '';
const WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5';

/**
 * GET /api/trip-planner/weather
 * Fetch weather forecast for destination
 */
router.get('/weather', async (req, res) => {
    const { city, startDate, endDate } = req.query;

    try {
        // Get coordinates for city
        const geoResponse = await axios.get(`${WEATHER_API_URL}/weather`, {
            params: {
                q: city,
                appid: WEATHER_API_KEY,
                units: 'metric'
            }
        });

        const { lat, lon } = geoResponse.data.coord;

        // Get 7-day forecast
        const forecastResponse = await axios.get(`${WEATHER_API_URL}/forecast`, {
            params: {
                lat,
                lon,
                appid: WEATHER_API_KEY,
                units: 'metric',
                cnt: 40 // 5 days, 3-hour intervals
            }
        });

        // Parse forecast by date
        const dailyForecasts = parseDailyForecasts(forecastResponse.data.list);

        res.json({
            success: true,
            city: geoResponse.data.name,
            country: geoResponse.data.sys.country,
            forecasts: dailyForecasts
        });

    } catch (error) {
        console.error('Weather API error:', error.message);
        res.status(500).json({
            error: 'Failed to fetch weather',
            message: error.message
        });
    }
});

/**
 * POST /api/trip-planner/create
 * Create trip plan with packing list and daily outfits
 */
router.post('/create', async (req, res) => {
    const { userId, destination, startDate, endDate, occasions } = req.body;

    try {
        console.log(`Creating trip plan for user ${userId} to ${destination}`);

        // 1. Fetch weather forecast
        const weatherData = await fetchWeatherForTrip(destination, startDate, endDate);

        // 2. Get user's wardrobe
        const wardrobe = await ClothingItem.find({ userId });

        if (wardrobe.length === 0) {
            return res.status(400).json({
                error: 'Empty wardrobe',
                message: 'Please add clothing items to your wardrobe first'
            });
        }

        // 3. Generate packing list and outfits
        const tripPlan = generateTripPlan({
            wardrobe,
            weather: weatherData.forecasts,
            startDate,
            endDate,
            occasions: occasions || ['casual']
        });

        res.json({
            success: true,
            tripId: Date.now().toString(), // Simple ID for demo
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
        });

    } catch (error) {
        console.error('Trip creation error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * Helper: Fetch weather for trip dates
 */
async function fetchWeatherForTrip(city, startDate, endDate) {
    try {
        // Get coordinates
        const geoResponse = await axios.get(`${WEATHER_API_URL}/weather`, {
            params: {
                q: city,
                appid: WEATHER_API_KEY,
                units: 'metric'
            }
        });

        const { lat, lon, name, sys } = geoResponse.data;

        // Get forecast
        const forecastResponse = await axios.get(`${WEATHER_API_URL}/forecast`, {
            params: {
                lat,
                lon,
                appid: WEATHER_API_KEY,
                units: 'metric',
                cnt: 40
            }
        });

        const dailyForecasts = parseDailyForecasts(forecastResponse.data.list);

        return {
            city: name,
            country: sys.country,
            forecasts: dailyForecasts
        };
    } catch (error) {
        console.error('Weather fetch error:', error.message);
        // Return dummy data if API fails
        return {
            city,
            country: 'Unknown',
            forecasts: generateDummyWeather(startDate, endDate)
        };
    }
}

/**
 * Helper: Parse 3-hour forecast into daily summaries
 */
function parseDailyForecasts(forecastList) {
    const dailyData = {};

    forecastList.forEach(item => {
        const date = item.dt_txt.split(' ')[0]; // YYYY-MM-DD

        if (!dailyData[date]) {
            dailyData[date] = {
                date,
                temps: [],
                conditions: [],
                descriptions: [],
                icons: []
            };
        }

        dailyData[date].temps.push(item.main.temp);
        dailyData[date].conditions.push(item.weather[0].main);
        dailyData[date].descriptions.push(item.weather[0].description);
        dailyData[date].icons.push(item.weather[0].icon);
    });

    // Aggregate into daily summaries
    return Object.values(dailyData).map(day => ({
        date: day.date,
        tempHigh: Math.round(Math.max(...day.temps)),
        tempLow: Math.round(Math.min(...day.temps)),
        tempAvg: Math.round(day.temps.reduce((a, b) => a + b) / day.temps.length),
        condition: mostCommon(day.conditions),
        description: mostCommon(day.descriptions),
        icon: mostCommon(day.icons)
    }));
}

/**
 * Helper: Get most common item in array
 */
function mostCommon(arr) {
    const counts = {};
    arr.forEach(item => counts[item] = (counts[item] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

/**
 * Helper: Generate dummy weather (fallback)
 */
function generateDummyWeather(startDate, endDate) {
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

/**
 * Core Algorithm: Generate Trip Plan
 */
function generateTripPlan({ wardrobe, weather, startDate, endDate, occasions }) {
    const outfitsByDay = [];
    const usedItems = {};  // Track item usage count
    const packingList = new Set();
    const missingItems = [];

    // For each day of the trip
    weather.forEach((dayWeather, index) => {
        const dayOutfits = [];

        // For each occasion on this day
        occasions.forEach(occasion => {
            const outfit = allocateOutfit({
                wardrobe,
                weather: dayWeather,
                occasion,
                usedItems,
                dayIndex: index
            });

            if (outfit.items.length > 0) {
                dayOutfits.push({
                    occasion,
                    items: outfit.items
                });

                // Add items to packing list
                outfit.items.forEach(item => {
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

        outfitsByDay.push({
            date: dayWeather.date,
            weather: dayWeather,
            outfits: dayOutfits
        });
    });

    // Convert packing list to array with usage counts
    const packingListArray = Array.from(packingList).map(itemId => {
        const item = wardrobe.find(w => w._id.toString() === itemId);
        return {
            ...item.toObject(),
            uses: usedItems[itemId]
        };
    });

    return {
        packingList: packingListArray,
        outfitsByDay,
        missingItems: Array.from(new Set(missingItems.map(m => JSON.stringify(m)))).map(m => JSON.parse(m))
    };
}

/**
 * Helper: Allocate outfit for specific day/occasion/weather
 */
function allocateOutfit({ wardrobe, weather, occasion, usedItems, dayIndex }) {
    const selectedItems = [];

    // Filter wardrobe by weather appropriateness
    const suitable = wardrobe.filter(item => {
        return isWeatherAppropriate(item, weather) && isOccasionAppropriate(item, occasion);
    });

    if (suitable.length === 0) {
        return { items: [] };
    }

    // Select top (shirt/blouse)
    const top = selectBestItem(suitable, ['shirt', 'tshirt', 'blouse', 'sweater', 'hoodie'], usedItems, dayIndex);
    if (top) selectedItems.push(top);

    // Select bottom (pants/skirt/shorts)
    const bottom = selectBestItem(suitable, ['pants', 'jeans', 'shorts', 'skirt'], usedItems, dayIndex);
    if (bottom) selectedItems.push(bottom);

    // Select shoes
    const shoes = selectBestItem(suitable, ['shoes', 'sneakers', 'boots', 'sandals'], usedItems, dayIndex);
    if (shoes) selectedItems.push(shoes);

    // Optional: jacket if cold
    if (weather.tempHigh < 15) {
        const jacket = selectBestItem(suitable, ['jacket', 'coat', 'blazer'], usedItems, dayIndex);
        if (jacket) selectedItems.push(jacket);
    }

    return { items: selectedItems };
}

/**
 * Helper: Check if item is appropriate for weather
 */
function isWeatherAppropriate(item, weather) {
    const temp = weather.tempAvg;
    const type = item.itemType?.toLowerCase() || '';
    const material = item.material?.toLowerCase() || '';

    // Hot weather (>25°C)
    if (temp > 25) {
        if (['shorts', 'tank', 'sandals', 'tshirt'].some(t => type.includes(t))) return true;
        if (material.includes('linen') || material.includes('cotton')) return true;
    }

    // Cold weather (<10°C)
    if (temp < 10) {
        if (['sweater', 'jacket', 'coat', 'boots', 'jeans'].some(t => type.includes(t))) return true;
        if (material.includes('wool') || material.includes('fleece')) return true;
    }

    // Moderate weather (default: most items work)
    return true;
}

/**
 * Helper: Check if item matches occasion
 */
function isOccasionAppropriate(item, occasion) {
    const type = item.itemType?.toLowerCase() || '';
    const style = item.style?.toLowerCase() || '';

    if (occasion === 'formal' || occasion === 'business') {
        return ['blazer', 'suit', 'dress', 'button', 'formal'].some(keyword =>
            type.includes(keyword) || style.includes(keyword)
        );
    }

    if (occasion === 'beach') {
        return ['swimwear', 'shorts', 'sandals', 'hat', 'sunglasses'].some(keyword =>
            type.includes(keyword)
        );
    }

    // Default: casual works for most occasions
    return true;
}

/**
 * Helper: Select best item from candidates
 */
function selectBestItem(wardrobe, types, usedItems, dayIndex) {
    const candidates = wardrobe.filter(item =>
        types.some(type => item.itemType?.toLowerCase().includes(type))
    );

    if (candidates.length === 0) return null;

    // Sort by least worn (maximize variety)
    candidates.sort((a, b) => {
        const aUses = usedItems[a._id.toString()] || 0;
        const bUses = usedItems[b._id.toString()] || 0;
        return aUses - bUses;
    });

    return candidates[0];
}

export default router;
