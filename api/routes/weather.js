import express from "express";

const router = express.Router();

const WEATHER_API_KEY = "0b0b523ea5bec9aef3883b17a3dbec98";

/**
 * GET /weather
 * Get weather by city name or coordinates
 */
router.get("/", async (req, res) => {
    const { city, lat, lon } = req.query;

    try {
        let query;

        // If coordinates are provided
        if (lat && lon) {
            query = `${lat},${lon}`;
        }
        // If city is provided
        else if (city) {
            query = city;
        }
        // Default to Tashkent
        else {
            query = "Tashkent";
        }

        const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${query}&aqi=no`;

        console.log("🌤️ Fetching weather for:", query);

        const response = await fetch(url);

        if (!response.ok) {
            const errorData = await response.json();
            console.error("❌ Weather API Error:", errorData);
            return res.status(response.status).json({
                error: errorData.error?.message || "Failed to fetch weather"
            });
        }

        const data = await response.json();

        console.log("✅ Weather fetched:", data.location.name, data.current.temp_c + "°C");

        res.json({
            temp: Math.round(data.current.temp_c),
            feels_like: Math.round(data.current.feelslike_c),
            description: data.current.condition.text,
            icon: data.current.condition.icon,
            city: data.location.name,
            humidity: data.current.humidity,
            wind_speed: data.current.wind_kph
        });

    } catch (error) {
        console.error("❌ Weather fetch error:", error.message);
        res.status(500).json({
            error: "Failed to fetch weather data",
            details: error.message
        });
    }
});

/**
 * POST /weather/coords
 * Get weather by latitude and longitude
 */
router.post("/coords", async (req, res) => {
    const { latitude, longitude } = req.body;

    if (!latitude || !longitude) {
        return res.status(400).json({
            error: "Latitude and longitude are required"
        });
    }

    try {
        const query = `${latitude},${longitude}`;
        const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${query}&aqi=no`;

        console.log("🌤️ Fetching weather by coords:", query);

        const response = await fetch(url);

        if (!response.ok) {
            const errorData = await response.json();
            console.error("❌ Weather API Error:", errorData);
            return res.status(response.status).json({
                error: errorData.error?.message || "Failed to fetch weather"
            });
        }

        const data = await response.json();

        res.json({
            temp: Math.round(data.current.temp_c),
            feels_like: Math.round(data.current.feelslike_c),
            description: data.current.condition.text,
            icon: data.current.condition.icon,
            city: data.location.name,
            humidity: data.current.humidity,
            wind_speed: data.current.wind_kph
        });

    } catch (error) {
        console.error("❌ Weather fetch error:", error.message);
        res.status(500).json({
            error: "Failed to fetch weather data",
            details: error.message
        });
    }
});

export default router;
