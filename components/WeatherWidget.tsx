import React, { useEffect, useState } from 'react';
import { View, StyleSheet,  } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import Animated, { FadeIn } from 'react-native-reanimated';
import { supabase } from '../lib/supabase';
import { createLogger } from '../src/utils/logger';

const logger = createLogger('WeatherWidget');

interface WeatherData {
    temp: number;
    tempHigh?: number;
    tempLow?: number;
    condition: string;
    location?: string;
}

/**
 * WeatherWidget - Alta-style context-aware header widget
 * 
 * Shows current weather to provide context for outfit recommendations:
 * "18° H:20° L:12°"
 * 
 * Features:
 * - Auto-fetches based on location
 * - Graceful fallback on error
 * - Minimal, non-intrusive design
 */
const WeatherWidget: React.FC = () => {
    const { t } = useTranslation();
    const [weather, setWeather] = useState<WeatherData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchWeather();
    }, []);

    const fetchWeather = async () => {
        try {
            const { status } = await Location.requestForegroundPermissionsAsync();
            if (status !== 'granted') {
                setLoading(false);
                return;
            }

            const location = await Location.getCurrentPositionAsync({
                accuracy: Location.Accuracy.Lowest,
            });

            const { data, error } = await supabase.functions.invoke('weather', {
                body: {
                    lat: location.coords.latitude,
                    lon: location.coords.longitude,
                    units: 'metric',
                },
            });

            if (!error && data) {
                setWeather({
                    temp: Math.round(data.temp ?? 20),
                    tempHigh: data.temp_max ? Math.round(data.temp_max) : undefined,
                    tempLow: data.temp_min ? Math.round(data.temp_min) : undefined,
                    condition: data.condition || data.description || 'clear',
                    location: data.city,
                });
            }
        } catch (error) {
            logger.error('Weather fetch failed', error);
        } finally {
            setLoading(false);
        }
    };

    const getWeatherIcon = (condition: string): keyof typeof Ionicons.glyphMap => {
        const lower = condition.toLowerCase();
        if (lower.includes('rain')) return 'rainy-outline';
        if (lower.includes('cloud')) return 'cloudy-outline';
        if (lower.includes('snow')) return 'snow-outline';
        if (lower.includes('thunder')) return 'thunderstorm-outline';
        if (lower.includes('clear') || lower.includes('sunny')) return 'sunny-outline';
        return 'partly-sunny-outline';
    };

    if (loading || !weather) return null;

    return (
        <Animated.View entering={FadeIn.duration(300)} style={styles.container}>
            <Ionicons
                name={getWeatherIcon(weather.condition)}
                size={16}
                color="#8E8E8E"
            />
            <ScaledText style={styles.temp}>{weather.temp}°</ScaledText>
            {weather.tempHigh && weather.tempLow && (
                <ScaledText style={styles.range}>
                    {t('weather.high')}:{weather.tempHigh}° {t('weather.low')}:{weather.tempLow}°
                </ScaledText>
            )}
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#F5F5F5',
        paddingHorizontal: 10,
        paddingVertical: 6,
        borderRadius: 16,
        gap: 4,
    },
    temp: {
        fontSize: 13,
        fontWeight: '600',
        color: '#0A1931',
    },
    range: {
        fontSize: 11,
        color: '#8E8E8E',
    },
});

export default WeatherWidget;
