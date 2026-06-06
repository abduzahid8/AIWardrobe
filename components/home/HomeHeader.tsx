import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native'
import { ScaledText } from '../ui/ScaledText';;
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import StreakBadge from '../StreakBadge';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { colors, spacing, typography } = LiquidGlass2026Theme;

interface Props {
    greeting: string;
    userName: string;
    onCalendarPress: () => void;
}

export default function HomeHeader({ greeting, userName, onCalendarPress }: Props) {
    const { t } = useTranslation();
    return (
        <>
            <View style={styles.titleSection}>
                <ScaledText style={styles.appTitle} accessibilityRole="header">AIWardrobe</ScaledText>
            </View>

            <View style={[styles.greetingSection, { marginBottom: spacing.md }]}>
                <ScaledText style={styles.greetingText} numberOfLines={2}>
                    {greeting}, {userName}
                </ScaledText>
                <View style={{ flexDirection: 'row', alignItems: 'center', gap: spacing.sm }}>
                    <StreakBadge variant="inline" />
                    <TouchableOpacity
                        style={styles.calendarButton}
                        onPress={onCalendarPress}
                        accessibilityLabel={t('homeHeader.openCalendar')}
                    >
                        <Ionicons name="calendar-outline" size={24} color={colors.text.primary} />
                        <View style={styles.calendarDot} />
                    </TouchableOpacity>
                </View>
            </View>
        </>
    );
}

const styles = StyleSheet.create({
    titleSection: { paddingTop: 0, marginBottom: 0 },
    appTitle: {
        ...typography.scale.headlineMedium,
        color: colors.text.primary,
        fontWeight: '700',
        textAlign: 'center',
        marginBottom: spacing.xl,
    },
    greetingSection: {
        paddingHorizontal: spacing.screenPadding,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    greetingText: {
        ...typography.scale.titleMedium,
        color: colors.text.secondary,
        fontWeight: '500',
        flex: 1,
    },
    calendarButton: {
        width: 40, height: 40, borderRadius: 20,
        backgroundColor: colors.background.secondary,
        justifyContent: 'center', alignItems: 'center',
        position: 'relative',
    },
    calendarDot: {
        position: 'absolute', top: 8, right: 8,
        width: 8, height: 8, borderRadius: 4,
        backgroundColor: colors.text.primary,
        borderWidth: 1.5,
        borderColor: colors.background.secondary,
    },
});
