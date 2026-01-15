/**
 * ALTA DAILY - PIXEL PERFECT PROFILE SCREEN
 * Based on exact design specification from screenshots
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    TouchableOpacity,
    StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { jwtDecode } from 'jwt-decode';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS from design spec
const ALTA = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#000000',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
};

// PressableScale props type
interface PressableScaleProps {
    children: React.ReactNode;
    onPress: () => void;
    style?: object;
}

// Saved look type
interface SavedLookType {
    id?: string;
    image: string;
    name?: string;
}

// Pressable with scale animation
const PressableScale = ({ children, onPress, style }: PressableScaleProps) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <TouchableOpacity
            activeOpacity={1}
            onPressIn={() => {
                scale.value = withSpring(0.97);
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            }}
            onPressOut={() => scale.value = withSpring(1)}
            onPress={onPress}
            style={style}
        >
            <Animated.View style={animatedStyle}>{children}</Animated.View>
        </TouchableOpacity>
    );
};

// Tab Button - Exact Alta style
const TabButton = ({ title, isActive, onPress }: { title: string; isActive: boolean; onPress: () => void }) => (
    <TouchableOpacity
        style={[styles.tabButton, isActive && styles.tabButtonActive]}
        onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
        }}
    >
        <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{title}</Text>
    </TouchableOpacity>
);

const AltaProfileScreen = () => {
    const navigation = useNavigation();
    const [activeTab, setActiveTab] = useState<'looks' | 'trips'>('looks');
    const [userName, setUserName] = useState('Username 14');
    const [location, setLocation] = useState('City, Country');
    const [savedLooks, setSavedLooks] = useState<SavedLookType[]>([]);

    useFocusEffect(useCallback(() => {
        loadUserData();
        loadSavedLooks();
    }, []));

    const loadUserData = async () => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                const decoded = jwtDecode<{ name?: string; username?: string }>(token);
                setUserName(decoded.name || decoded.username || 'Username 14');
            }
        } catch (e) { }
    };

    const loadSavedLooks = async () => {
        try {
            const saved = await AsyncStorage.getItem('savedLooks');
            if (saved) setSavedLooks(JSON.parse(saved));
        } catch (e) { }
    };

    const handleLogout = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        await AsyncStorage.removeItem('userToken');
        (navigation as any).reset({ index: 0, routes: [{ name: 'Auth' }] });
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={ALTA.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header - Exact Alta layout */}
                <View style={styles.header}>
                    {/* Left: Share icon */}
                    <TouchableOpacity style={styles.headerIcon}>
                        <Ionicons name="share-outline" size={22} color={ALTA.text} />
                    </TouchableOpacity>

                    {/* Right: Avatar pill + Settings */}
                    <View style={styles.headerRight}>
                        <TouchableOpacity style={styles.avatarPill}>
                            <Ionicons name="person-outline" size={14} color={ALTA.text} />
                            <Text style={styles.avatarPillText}>Your avatar</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.headerIcon}>
                            <Ionicons name="settings-outline" size={22} color={ALTA.text} />
                        </TouchableOpacity>
                    </View>
                </View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {/* Profile Section */}
                    <View style={styles.profileSection}>
                        <View style={styles.avatarCircle}>
                            <Ionicons name="shirt" size={28} color={ALTA.textMuted} />
                        </View>
                        <Text style={styles.username}>{userName}</Text>
                        <View style={styles.locationRow}>
                            <Text style={styles.location}>{location}</Text>
                            <Ionicons name="location" size={12} color={ALTA.textMuted} />
                        </View>
                    </View>

                    {/* Friends Card - Exact Alta style */}
                    <View style={styles.friendsCard}>
                        <Text style={styles.friendsTitle}>Alta is better with friends</Text>
                        <Text style={styles.friendsSubtitle}>
                            Share your style and try on your friends' looks!
                        </Text>

                        {/* Overlapping avatars - 48px size, -12px overlap */}
                        <View style={styles.avatarsRow}>
                            <Image
                                source={{ uri: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100' }}
                                style={styles.friendAvatar}
                            />
                            <Image
                                source={{ uri: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100' }}
                                style={[styles.friendAvatar, styles.avatarOverlap]}
                            />
                            <Image
                                source={{ uri: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=100' }}
                                style={[styles.friendAvatar, styles.avatarOverlap]}
                            />
                        </View>

                        {/* Add friends button - Black, 14px vertical, 28px radius */}
                        <PressableScale onPress={() => { }}>
                            <View style={styles.addFriendsButton}>
                                <Ionicons name="add" size={18} color={ALTA.background} />
                                <Text style={styles.addFriendsText}>Add friends</Text>
                            </View>
                        </PressableScale>
                    </View>

                    {/* Tabs - 24px gap, 2px underline */}
                    <View style={styles.tabsContainer}>
                        <TabButton
                            title="Looks"
                            isActive={activeTab === 'looks'}
                            onPress={() => setActiveTab('looks')}
                        />
                        <TabButton
                            title="Trips"
                            isActive={activeTab === 'trips'}
                            onPress={() => setActiveTab('trips')}
                        />
                    </View>

                    {/* Tab Content */}
                    {activeTab === 'looks' ? (
                        savedLooks.length === 0 ? (
                            <View style={styles.emptyState}>
                                <Ionicons name="images-outline" size={48} color={ALTA.textMuted} />
                                <Text style={styles.emptyTitle}>No saved looks yet</Text>
                                <Text style={styles.emptySubtitle}>Your favorite outfits will appear here</Text>
                            </View>
                        ) : (
                            <View style={styles.looksGrid}>
                                {savedLooks.map((look, i) => (
                                    <View key={i} style={styles.lookCard}>
                                        <Image source={{ uri: look.image }} style={styles.lookImage} />
                                    </View>
                                ))}
                            </View>
                        )
                    ) : (
                        <View style={styles.emptyState}>
                            <Ionicons name="airplane-outline" size={48} color={ALTA.textMuted} />
                            <Text style={styles.emptyTitle}>No trips planned</Text>
                            <Text style={styles.emptySubtitle}>Plan a trip to get packing suggestions</Text>
                        </View>
                    )}

                    {/* Logout */}
                    <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
                        <Ionicons name="log-out-outline" size={20} color={ALTA.textMuted} />
                        <Text style={styles.logoutText}>Log Out</Text>
                    </TouchableOpacity>

                    <View style={{ height: 100 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: ALTA.background,
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 10,
    },
    headerIcon: {
        padding: 8,
    },
    headerRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    // Avatar pill: bg #F5F5F5, radius 20, padding 8v 12h
    avatarPill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: ALTA.surface,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 20,
    },
    avatarPillText: {
        fontSize: 12,
        fontWeight: '500',
        color: ALTA.text,
    },

    // Content
    scrollContent: {
        paddingTop: 16,
    },

    // Profile
    profileSection: {
        alignItems: 'center',
        marginBottom: 24,
    },
    avatarCircle: {
        width: 72,
        height: 72,
        borderRadius: 36,
        backgroundColor: ALTA.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    username: {
        fontSize: 18, // From spec
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 4,
    },
    locationRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    location: {
        fontSize: 13,
        color: ALTA.textMuted,
    },

    // Friends Card: border 1px #E5E5E5, radius 16, padding 20
    friendsCard: {
        marginHorizontal: 20,
        backgroundColor: ALTA.background,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: ALTA.border,
        padding: 20,
        marginBottom: 32,
    },
    friendsTitle: {
        fontSize: 16, // From spec
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 4,
    },
    friendsSubtitle: {
        fontSize: 13, // From spec
        fontWeight: '400',
        color: ALTA.textSecondary,
        marginBottom: 16,
    },
    // Avatars: 48px, -12px overlap
    avatarsRow: {
        flexDirection: 'row',
        marginBottom: 16,
    },
    friendAvatar: {
        width: 48,
        height: 48,
        borderRadius: 24,
        borderWidth: 2,
        borderColor: ALTA.background,
    },
    avatarOverlap: {
        marginLeft: -12,
    },
    // Button: 14px vertical, 28px radius
    addFriendsButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        backgroundColor: ALTA.text,
        paddingVertical: 14,
        borderRadius: 28,
    },
    addFriendsText: {
        fontSize: 15, // From spec
        fontWeight: '600',
        color: ALTA.background,
    },

    // Tabs: 24px gap, 2px underline
    tabsContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 24,
        marginBottom: 32,
    },
    tabButton: {
        paddingVertical: 8,
        paddingHorizontal: 4,
    },
    tabButtonActive: {
        borderBottomWidth: 2,
        borderBottomColor: ALTA.text,
    },
    tabText: {
        fontSize: 15, // From spec
        fontWeight: '500',
        color: ALTA.textMuted, // #999 inactive
    },
    tabTextActive: {
        color: ALTA.text, // #000 active
    },

    // Empty State
    emptyState: {
        alignItems: 'center',
        paddingVertical: 48,
    },
    emptyTitle: {
        fontSize: 17,
        fontWeight: '600',
        color: ALTA.text,
        marginTop: 16,
    },
    emptySubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        marginTop: 4,
    },

    // Looks Grid
    looksGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: 16,
        gap: 8,
    },
    lookCard: {
        width: (width - 40) / 3,
        aspectRatio: 0.7,
        borderRadius: 8,
        overflow: 'hidden',
        backgroundColor: ALTA.surface,
    },
    lookImage: {
        width: '100%',
        height: '100%',
    },

    // Logout
    logoutButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        marginTop: 32,
        paddingVertical: 16,
    },
    logoutText: {
        fontSize: 15,
        color: ALTA.textMuted,
    },
});

export default AltaProfileScreen;
