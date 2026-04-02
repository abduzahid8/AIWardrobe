/**
 * TermsOfServiceScreen — Required for App Store submission
 */

import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

const SECTIONS = [
    {
        title: '1. Acceptance of Terms',
        body: 'By downloading, installing, or using AIWardrobe, you agree to be bound by these Terms of Service. If you do not agree, do not use the application.',
    },
    {
        title: '2. Description of Service',
        body: `AIWardrobe is an AI-powered wardrobe management application that provides:

• Clothing item digitization via camera/photo scanning
• AI-generated outfit suggestions
• Virtual try-on functionality
• Wardrobe analytics and style insights
• Cloud synchronization across devices

Some features require an active internet connection and may use third-party AI services.`,
    },
    {
        title: '3. User Accounts',
        body: `• You must provide accurate information when creating an account.
• You are responsible for maintaining the security of your account credentials.
• You must be at least 13 years old to create an account.
• One person may not maintain more than one account.
• We reserve the right to suspend or terminate accounts that violate these terms.`,
    },
    {
        title: '4. User Content',
        body: `• You retain ownership of all photos and content you upload.
• By uploading content, you grant us a license to process, store, and display it within the app.
• You agree not to upload content that is illegal, offensive, or infringes on others' rights.
• We may use anonymized, aggregated data to improve our AI models.`,
    },
    {
        title: '5. Subscriptions & Payments',
        body: `• AIWardrobe offers free and premium subscription tiers.
• Premium subscriptions are billed through the App Store.
• Subscriptions auto-renew unless cancelled at least 24 hours before the end of the current period.
• Refunds are handled according to the App Store's refund policy.`,
    },
    {
        title: '6. AI-Generated Content',
        body: `• AI suggestions are generated algorithmically and may not always be accurate.
• Virtual try-on results are approximations and may not perfectly represent real-world appearance.
• We do not guarantee the accuracy of AI color, pattern, or material detection.
• AI-generated outfit suggestions are for informational purposes only.`,
    },
    {
        title: '7. Prohibited Uses',
        body: `You agree not to:

• Reverse engineer, decompile, or attempt to extract source code from the app.
• Use the app for commercial purposes without authorization.
• Scrape, harvest, or collect data from other users.
• Attempt to bypass subscription or authentication mechanisms.
• Use the app to transmit malware or malicious content.`,
    },
    {
        title: '8. Intellectual Property',
        body: '• AIWardrobe and its original content, features, and functionality are owned by us and protected by intellectual property laws.\n• The AIWardrobe name, logo, and branding are our trademarks.',
    },
    {
        title: '9. Limitation of Liability',
        body: 'AIWardrobe is provided "as is" without warranty of any kind. We are not liable for any indirect, incidental, special, or consequential damages resulting from your use of the app.',
    },
    {
        title: '10. Changes to Terms',
        body: 'We reserve the right to modify these terms at any time. Continued use after changes constitutes acceptance. We will notify you of significant changes through the app.',
    },
    {
        title: '11. Contact',
        body: 'For questions about these Terms, contact us at:\n\n📧 support@aiwardrobe.app',
    },
];

export default function TermsOfServiceScreen({ navigation }: any) {
    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Terms of Service</Text>
                <View style={{ width: 32 }} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                <Text style={styles.lastUpdated}>Last updated: March 2026</Text>

                <Text style={styles.intro}>
                    Please read these Terms of Service carefully before using AIWardrobe.
                </Text>

                {SECTIONS.map((section, idx) => (
                    <View key={idx} style={styles.section}>
                        <Text style={styles.sectionTitle}>{section.title}</Text>
                        <Text style={styles.sectionBody}>{section.body}</Text>
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
