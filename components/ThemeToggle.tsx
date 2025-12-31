import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useTheme } from '../src/theme/ThemeContext';
import { spacing, borderRadius, shadows } from '../src/theme';

interface ThemeToggleProps {
    showLabel?: boolean;
    compact?: boolean;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({
    showLabel = true,
    compact = false
}) => {
    const { isDark, themeMode, setThemeMode, toggleTheme, colors } = useTheme();

    const handleToggle = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        toggleTheme();
    };

    if (compact) {
        return (
            <TouchableOpacity
                onPress={handleToggle}
                style={[styles.compactButton, { backgroundColor: colors.surfaceHighlight }]}
            >
                <Ionicons
                    name={isDark ? 'moon' : 'sunny'}
                    size={20}
                    color={colors.text.primary}
                />
            </TouchableOpacity>
        );
    }

    return (
        <View style={[styles.container, { backgroundColor: colors.surface }]}>
            <View style={styles.labelContainer}>
                <Ionicons
                    name={isDark ? 'moon' : 'sunny'}
                    size={22}
                    color={colors.text.primary}
                />
                {showLabel && (
                    <Text style={[styles.label, { color: colors.text.primary }]}>
                        Dark Mode
                    </Text>
                )}
            </View>
            <Switch
                value={isDark}
                onValueChange={handleToggle}
                trackColor={{
                    false: colors.border,
                    true: colors.text.accent
                }}
                thumbColor={isDark ? colors.surface : colors.surface}
                ios_backgroundColor={colors.border}
            />
        </View>
    );
};

// Full theme selector with system option
export const ThemeSelector: React.FC = () => {
    const { themeMode, setThemeMode, colors } = useTheme();

    const options: { mode: 'light' | 'dark' | 'system'; icon: any; label: string }[] = [
        { mode: 'light', icon: 'sunny', label: 'Light' },
        { mode: 'dark', icon: 'moon', label: 'Dark' },
        { mode: 'system', icon: 'phone-portrait-outline', label: 'System' },
    ];

    const handleSelect = (mode: 'light' | 'dark' | 'system') => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setThemeMode(mode);
    };

    return (
        <View style={[styles.selectorContainer, { backgroundColor: colors.surface }]}>
            <Text style={[styles.selectorTitle, { color: colors.text.primary }]}>
                Appearance
            </Text>
            <View style={styles.optionsRow}>
                {options.map((option) => {
                    const isActive = themeMode === option.mode;
                    return (
                        <TouchableOpacity
                            key={option.mode}
                            style={[
                                styles.option,
                                {
                                    backgroundColor: isActive
                                        ? colors.text.primary
                                        : colors.surfaceHighlight,
                                },
                            ]}
                            onPress={() => handleSelect(option.mode)}
                            activeOpacity={0.7}
                        >
                            <Ionicons
                                name={option.icon}
                                size={20}
                                color={isActive ? colors.text.inverse : colors.text.secondary}
                            />
                            <Text
                                style={[
                                    styles.optionLabel,
                                    {
                                        color: isActive
                                            ? colors.text.inverse
                                            : colors.text.secondary,
                                    },
                                ]}
                            >
                                {option.label}
                            </Text>
                        </TouchableOpacity>
                    );
                })}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.m,
        borderRadius: borderRadius.m,
        ...shadows.soft,
    },
    labelContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.m,
    },
    label: {
        fontSize: 16,
        fontWeight: '500',
    },
    compactButton: {
        padding: spacing.s,
        borderRadius: borderRadius.full,
    },
    selectorContainer: {
        padding: spacing.m,
        borderRadius: borderRadius.l,
        ...shadows.soft,
    },
    selectorTitle: {
        fontSize: 14,
        fontWeight: '600',
        marginBottom: spacing.m,
    },
    optionsRow: {
        flexDirection: 'row',
        gap: spacing.s,
    },
    option: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: spacing.xs,
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: borderRadius.m,
    },
    optionLabel: {
        fontSize: 13,
        fontWeight: '600',
    },
});

export default ThemeToggle;
