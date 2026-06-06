/**
 * PrivacyPolicyScreen — Required for App Store submission
 */

import React from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { useTranslation } from 'react-i18next';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

const SECTIONS = [
    {
        title: 'privacyPolicy.informationWeCollect.title',
        body: 'privacyPolicy.informationWeCollect.body',
    },
    {
        title: 'privacyPolicy.howWeUse.title',
        body: 'privacyPolicy.howWeUse.body',
    },
    {
        title: 'privacyPolicy.dataStorage.title',
        body: 'privacyPolicy.dataStorage.body',
    },
    {
        title: 'privacyPolicy.thirdParty.title',
        body: 'privacyPolicy.thirdParty.body',
    },
    {
        title: 'privacyPolicy.yourRights.title',
        body: 'privacyPolicy.yourRights.body',
    },
    {
        title: 'privacyPolicy.childrensPrivacy.title',
        body: 'privacyPolicy.childrensPrivacy.body',
    },
    {
        title: 'privacyPolicy.changes.title',
        body: 'privacyPolicy.changes.body',
    },
    {
        title: 'privacyPolicy.contact.title',
        body: 'privacyPolicy.contact.body',
    },
];

export default function PrivacyPolicyScreen({ navigation }: any) {
    const { t } = useTranslation();
    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
                </TouchableOpacity>
                <ScaledText style={styles.headerTitle}>{t('privacyPolicy.title')}</ScaledText>
                <View style={{ width: 32 }} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                <ScaledText style={styles.lastUpdated}>{t('privacyPolicy.lastUpdated')}</ScaledText>

                <ScaledText style={styles.intro}>
                    {t('privacyPolicy.intro')}
                </ScaledText>

                {SECTIONS.map((section, idx) => (
                    <View key={idx} style={styles.section}>
                        <ScaledText style={styles.sectionTitle}>{t(section.title)}</ScaledText>
                        <ScaledText style={styles.sectionBody}>{t(section.body)}</ScaledText>
                    </View>
                ))}
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.lg,
        paddingVertical: spacing.md,
    },
    backButton: {
        width: 32,
        height: 32,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    scrollContent: {
        paddingHorizontal: spacing.lg,
        paddingBottom: spacing.xxxl,
    },
    lastUpdated: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        marginBottom: spacing.md,
    },
    intro: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        lineHeight: 22,
        marginBottom: spacing.xl,
    },
    section: {
        marginBottom: spacing.xl,
    },
    sectionTitle: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '700',
        marginBottom: spacing.sm,
    },
    sectionBody: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        lineHeight: 22,
    },
});
