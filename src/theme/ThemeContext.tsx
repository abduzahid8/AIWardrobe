import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Appearance, ColorSchemeName, useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { lightColors, darkColors, getThemeColors } from './index';

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
    const [isInitialized, setIsInitialized] = useState(false);

    // Load saved theme preference
    useEffect(() => {
        const loadTheme = async () => {
            try {
                const savedMode = await AsyncStorage.getItem(THEME_STORAGE_KEY);
                if (savedMode && ['light', 'dark', 'system'].includes(savedMode)) {
                    setThemeModeState(savedMode as ThemeMode);
                }
            } catch (error) {
                console.log('Error loading theme:', error);
            } finally {
                setIsInitialized(true);
            }
        };
        loadTheme();
    }, []);

    // Calculate actual dark mode state
    const isDark = themeMode === 'system'
        ? systemColorScheme === 'dark'
        : themeMode === 'dark';

    // Get colors based on current theme
    const colors = getThemeColors(isDark);

    // Set theme mode and persist
    const setThemeMode = async (mode: ThemeMode) => {
        setThemeModeState(mode);
        try {
            await AsyncStorage.setItem(THEME_STORAGE_KEY, mode);
        } catch (error) {
            console.log('Error saving theme:', error);
        }
    };

    // Toggle between light and dark (skips system)
    const toggleTheme = () => {
        const newMode = isDark ? 'light' : 'dark';
        setThemeMode(newMode);
    };

    // Listen to system theme changes
    useEffect(() => {
        const subscription = Appearance.addChangeListener(({ colorScheme }) => {
            // Only update if using system theme
            if (themeMode === 'system') {
                // Force re-render
                setThemeModeState('system');
            }
        });

        return () => subscription.remove();
    }, [themeMode]);

    if (!isInitialized) {
        return null; // Or a loading spinner
    }

    return (
        <ThemeContext.Provider
            value={{
                isDark,
                themeMode,
                colors,
                setThemeMode,
                toggleTheme,
            }}
        >
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
