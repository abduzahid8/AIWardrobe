/**
 * PrivacyPolicyScreen — Required for App Store submission
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
        title: 'Information We Collect',
        body: `When you use AIWardrobe, we collect the following types of information:

• Account Information: Email address, username, and optional profile details you provide during registration.
• Wardrobe Data: Photos of clothing items you upload, along with AI-detected attributes (category, color, pattern, material).
• Usage Data: Wear logs, outfit history, and feature usage patterns to improve AI suggestions.
• Device Information: Device type, OS version, and app version for bug fixing and optimization.
• Location Data: Optional — used only for weather-based outfit recommendations when you grant permission.`,
    },
    {
        title: 'How We Use Your Information',
        body: `We use your information to:

• Provide and improve AI-powered outfit suggestions and wardrobe management.
• Sync your wardrobe across devices via secure cloud storage.
• Generate personalized analytics (most worn items, style diversity, etc.).
• Send relevant notifications (daily outfit suggestions, streak reminders).
• Improve our AI models and app experience.

We do NOT sell your personal data to third parties.`,
    },
    {
        title: 'Data Storage & Security',
        body: `• Your data is stored securely using Supabase (PostgreSQL) with row-level security.
• All data transmission is encrypted via TLS/HTTPS.
• Photos are stored in secure cloud storage with access restricted to your account.
• We retain your data as long as your account is active. You can delete your account and all associated data at any time.`,
    },
    {
        title: 'Third-Party Services',
        body: `We use the following third-party services:

• Supabase: Authentication and data storage.
• Google Gemini: AI clothing analysis and outfit suggestions.
• OpenAI: Natural language processing for AI assistant.
• Replicate: Virtual try-on image generation.

Each service has its own privacy policy that governs their handling of data.`,
    },
    {
        title: 'Your Rights',
        body: `You have the right to:

• Access: View all data associated with your account.
• Correction: Update or correct your personal information.
• Deletion: Delete your account and all associated data via Profile > Delete Account.
• Data Portability: Request a copy of your data in a machine-readable format.
• Opt-Out: Disable location services and notifications at any time.`,
    },
    {
        title: 'Children\'s Privacy',
        body: 'AIWardrobe is not intended for children under 13. We do not knowingly collect personal information from children under 13. If you believe a child has provided us with personal data, please contact us.',
    },
    {
        title: 'Changes to This Policy',
        body: 'We may update this Privacy Policy from time to time. We will notify you of significant changes through the app or via email. Continued use of AIWardrobe after changes constitutes acceptance of the updated policy.',
    },
    {
        title: 'Contact Us',
        body: 'If you have questions about this Privacy Policy, please contact us at:\n\n📧 support@aiwardrobe.app',
    },
];

export default function PrivacyPolicyScreen({ navigation }: any) {
    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Privacy Policy</Text>
                <View style={{ width: 32 }} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                <Text style={styles.lastUpdated}>Last updated: March 2026</Text>

                <Text style={styles.intro}>
                    AIWardrobe ("we", "our", "us") is committed to protecting your privacy.
                    This Privacy Policy explains how we collect, use, and safeguard your information
                    when you use our mobile application.
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
