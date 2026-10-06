import React, { createContext, useContext, useState, useEffect, useMemo, useCallback, ReactNode } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { lightColors, getThemeColors } from './index';
import { createLogger } from '../utils/logger';

const logger = createLogger('Theme');

type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeContextType {
    isDark: boolean;
    themeMode: ThemeMode;
    colors: typeof lightColors;
    setThemeMode: (mode: ThemeMode) => void;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_STORAGE_KEY = '@aiwardrobe_theme_mode';

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const systemColorScheme = useColorScheme();
    const [themeMode, setThemeModeState] = useState<ThemeMode>('system');

    // Load saved theme preference in background — do NOT block first render.
    // Previously this gated the whole app behind AsyncStorage + ActivityIndicator,
    // adding a second splash gate on top of RootNavigator's gate (freeze on cold start).
    // Now we render immediately with system default and hydrate silently.
    useEffect(() => {
        let cancelled = false;
        const loadTheme = async () => {
            try {
                const savedMode = await AsyncStorage.getItem(THEME_STORAGE_KEY);
                if (!cancelled && savedMode && ['light', 'dark', 'system'].includes(savedMode)) {
                    setThemeModeState(savedMode as ThemeMode);
                }
            } catch (error) {
                logger.error('Error loading theme', error);
            }
        };
        loadTheme();
        return () => {
            cancelled = true;
        };
    }, []);

    // Calculate actual dark mode state
    const isDark = themeMode === 'system'
        ? systemColorScheme === 'dark'
        : themeMode === 'dark';

    // getThemeColors returns a stable palette reference (light/dark object),
    // so no per-render allocation here.
    const colors = getThemeColors(isDark);

    // Stable callbacks — previously re-created every render, invalidating
    // the provider value and re-rendering every useTheme() consumer.
    const setThemeMode = useCallback(async (mode: ThemeMode) => {
        setThemeModeState(mode);
        try {
            await AsyncStorage.setItem(THEME_STORAGE_KEY, mode);
        } catch (error) {
            logger.error('Error saving theme', error);
        }
    }, []);

    // Toggle between light and dark (skips system)
    const toggleTheme = useCallback(() => {
        setThemeModeState((prev) => {
            const currentlyDark = prev === 'system'
                ? systemColorScheme === 'dark'
                : prev === 'dark';
            const newMode = currentlyDark ? 'light' : 'dark';
            AsyncStorage.setItem(THEME_STORAGE_KEY, newMode).catch((error) =>
                logger.error('Error saving theme', error),
            );
            return newMode;
        });
    }, [systemColorScheme]);

    // No Appearance listener needed — useColorScheme() already re-renders
    // on system changes. The old listener forced an extra setState cycle.

    const value = useMemo(
        () => ({ isDark, themeMode, colors, setThemeMode, toggleTheme }),
        [isDark, themeMode, colors, setThemeMode, toggleTheme],
    );

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    );
};

// Custom hook to use theme
export const useTheme = (): ThemeContextType => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};

// Hook that returns just the colors (for backwards compatibility)
export const useThemeColors = () => {
    const { colors } = useTheme();
    return colors;
};

export default ThemeContext;
