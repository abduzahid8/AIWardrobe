// useTheme Hook - Access theme colors and utilities
// Provides theme context with dark/light mode support

import { useState, useEffect } from 'react';
import { Appearance, ColorSchemeName } from 'react-native';
import {
    lightColors,
    darkColors,
    typography,
    spacing,
    borderRadius,
    shadows,
    animations,
    haptics,
} from '../theme';

export interface Theme {
    colors: typeof lightColors;
    typography: typeof typography;
    spacing: typeof spacing;
    borderRadius: typeof borderRadius;
    shadows: typeof shadows;
    animations: typeof animations;
    haptics: typeof haptics;
    isDark: boolean;
}

export const useTheme = (): Theme => {
    const [colorScheme, setColorScheme] = useState<ColorSchemeName>(
        Appearance.getColorScheme()
    );

    useEffect(() => {
        const subscription = Appearance.addChangeListener(({ colorScheme }) => {
            setColorScheme(colorScheme);
        });

        return () => subscription.remove();
    }, []);

    const isDark = colorScheme === 'dark';
    const colors = isDark ? darkColors : lightColors;

    return {
        colors,
        typography,
        spacing,
        borderRadius,
        shadows,
        animations,
        haptics,
        isDark,
    };
};
