/**
 * src/components/NavigationMenu.tsx — Universal Navigation Menu
 *
 * A slide-up modal that lists ALL screens in the app, organized by category.
 * Accessible from any main screen via a hamburger menu button.
 */

import React, { useCallback, useMemo } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, SlideInUp } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// ============================================
// SCREEN REGISTRY — All app screens organized
// ============================================

interface ScreenItem {
    name: string;
    route: string;
    icon: keyof typeof Ionicons.glyphMap;
    description?: string;
}

interface ScreenCategory {
    title: string;
    icon: keyof typeof Ionicons.glyphMap;
    screens: ScreenItem[];
}

const SCREEN_REGISTRY: ScreenCategory[] = [
    {
        title: 'Main Tabs',
        icon: 'grid-outline',
        screens: [
            { name: 'Home', route: 'Home', icon: 'home-outline', description: 'Daily outfit suggestions' },
            { name: 'My Closet', route: 'Closet', icon: 'shirt-outline', description: 'Your wardrobe items' },
            { name: 'Inspiration', route: 'Inspo', icon: 'sparkles-outline', description: 'Style guides & shopping' },
            { name: 'Profile', route: 'Profile', icon: 'person-outline', description: 'Settings & analytics' },
        ],
    },
    {
        title: 'AI Features',
        icon: 'sparkles',
        screens: [
            { name: 'AI Stylist Chat', route: 'StylistChat', icon: 'chatbubble-ellipses', description: 'Ask AI for outfit help' },
            { name: 'AI Try-On', route: 'AITryOn', icon: 'body', description: 'Virtual outfit preview' },
            { name: 'AI Outfit Creator', route: 'AIOutfit', icon: 'color-wand', description: 'Generate new outfits' },
        ],
    },
    {
        title: 'Outfits & Planning',
        icon: 'calendar',
        screens: [
            { name: 'Outfit Calendar', route: 'OutfitCalendar', icon: 'calendar-number', description: 'Plan outfits by date' },
        ],
    },
    {
        title: 'Wardrobe & Items',
        icon: 'shirt',
        screens: [
            { name: 'Wardrobe Video', route: 'WardrobeVideo', icon: 'videocam', description: 'Video catalog' },
            { name: 'Wardrobe Analytics', route: 'WardrobeAnalytics', icon: 'stats-chart', description: 'Usage insights' },
        ],
    },
    {
        title: 'Style & Profile',
        icon: 'person',
        screens: [
            { name: 'Create Avatar', route: 'CreateAvatar', icon: 'person-add', description: 'Digital you' },
        ],
    },
    {
        title: 'Camera & Scan',
        icon: 'camera',
        screens: [
            { name: 'Camera', route: 'Camera', icon: 'camera', description: 'Scan clothing' },
            { name: 'Add by Photo', route: 'Camera', icon: 'image', description: 'Upload from gallery' },
        ],
    },
    {
        title: 'Account & Settings',
        icon: 'settings',
        screens: [
            { name: 'Sign In', route: 'SignIn', icon: 'log-in', description: 'Login' },
            { name: 'Sign Up', route: 'SignUp', icon: 'person-add', description: 'Create account' },
            { name: 'Forgot Password', route: 'ForgotPassword', icon: 'key', description: 'Reset password' },
            { name: 'Reset Password', route: 'ResetPassword', icon: 'lock-open', description: 'New password' },
            { name: 'Privacy Policy', route: 'PrivacyPolicy', icon: 'shield', description: 'Data privacy' },
            { name: 'Terms of Service', route: 'TermsOfService', icon: 'document-text', description: 'Legal terms' },
        ],
    },
    {
        title: 'Other',
        icon: 'ellipsis-horizontal',
        screens: [
            { name: 'Paywall', route: 'Paywall', icon: 'card', description: 'Upgrade to Pro' },
            { name: 'Reviews', route: 'Review', icon: 'star', description: 'Rate the app' },
        ],
    },
];

// ============================================
// COMPONENT
// ============================================

interface NavigationMenuProps {
    visible: boolean;
    onClose: () => void;
}

export const NavigationMenu: React.FC<NavigationMenuProps> = ({ visible, onClose }) => {
    const navigation = useNavigation();

    const handleNavigate = useCallback((route: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onClose();
        // Small delay to let modal close first
        setTimeout(() => {
            (navigation as any).navigate(route);
        }, 200);
    }, [navigation, onClose]);

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <Animated.View entering={FadeIn} style={styles.overlay}>
                <BlurView intensity={30} tint="dark" style={styles.blur}>
                    <TouchableOpacity style={styles.backdrop} onPress={onClose} />
                </BlurView>

                <Animated.View entering={SlideInUp.springify()} style={styles.sheet}>
                    {/* Header */}
                    <View style={styles.header}>
                        <Text style={styles.title}>All Screens</Text>
                        <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                            <Ionicons name="close" size={24} color={colors.text.primary} />
                        </TouchableOpacity>
                    </View>

                    {/* Screen List */}
                    <ScrollView
                        style={styles.scrollView}
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                    >
                        {SCREEN_REGISTRY.map((category, catIndex) => (
                            <View key={category.title} style={styles.category}>
                                {/* Category Header */}
                                <View style={styles.categoryHeader}>
                                    <Ionicons name={category.icon} size={18} color={colors.text.secondary} />
                                    <Text style={styles.categoryTitle}>{category.title}</Text>
                                    <View style={styles.badge}>
                                        <Text style={styles.badgeText}>{category.screens.length}</Text>
                                    </View>
                                </View>

                                {/* Screens in Category */}
                                <View style={styles.screensGrid}>
                                    {category.screens.map((screen) => (
                                        <TouchableOpacity
                                            key={screen.route + screen.name}
                                            style={styles.screenButton}
                                            onPress={() => handleNavigate(screen.route)}
                                            activeOpacity={0.7}
                                        >
                                            <View style={styles.iconContainer}>
                                                <Ionicons name={screen.icon} size={22} color={colors.text.primary} />
                                            </View>
                                            <View style={styles.screenInfo}>
                                                <Text style={styles.screenName} numberOfLines={1}>
                                                    {screen.name}
                                                </Text>
                                                {screen.description && (
                                                    <Text style={styles.screenDesc} numberOfLines={1}>
                                                        {screen.description}
                                                    </Text>
                                                )}
                                            </View>
                                            <Ionicons name="chevron-forward" size={18} color={colors.text.tertiary} />
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                {catIndex < SCREEN_REGISTRY.length - 1 && (
                                    <View style={styles.divider} />
                                )}
                            </View>
                        ))}

                        {/* Bottom spacing */}
                        <View style={{ height: 40 }} />
                    </ScrollView>
                </Animated.View>
            </Animated.View>
        </Modal>
    );
};


const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        justifyContent: 'flex-end',
    },
    blur: {
        ...StyleSheet.absoluteFillObject,
    },
    backdrop: {
        flex: 1,
    },
    sheet: {
        backgroundColor: colors.background.primary,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        maxHeight: SCREEN_HEIGHT * 0.85,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.15,
        shadowRadius: 20,
        elevation: 20,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
        borderBottomWidth: 1,
        borderBottomColor: colors.border.glass,
    },
    title: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
    },
    closeButton: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    scrollView: {
        maxHeight: SCREEN_HEIGHT * 0.75,
    },
    scrollContent: {
        paddingTop: 8,
    },
    category: {
        paddingHorizontal: 16,
        paddingVertical: 12,
    },
    categoryHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        marginBottom: 12,
    },
    categoryTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.secondary,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    badge: {
        backgroundColor: colors.background.tertiary,
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 10,
    },
    badgeText: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    screensGrid: {
        gap: 8,
    },
    screenButton: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
        backgroundColor: colors.background.secondary,
        paddingHorizontal: 12,
        paddingVertical: 12,
        borderRadius: 12,
    },
    iconContainer: {
        width: 40,
        height: 40,
        borderRadius: 10,
        backgroundColor: colors.background.tertiary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    screenInfo: {
        flex: 1,
    },
    screenName: {
        fontSize: 15,
        fontWeight: '600',
        color: colors.text.primary,
    },
    screenDesc: {
        fontSize: 12,
        color: colors.text.tertiary,
        marginTop: 2,
    },
    divider: {
        height: 1,
        backgroundColor: colors.border.glass,
        marginVertical: 16,
        marginHorizontal: 16,
    },
});

export default NavigationMenu;
