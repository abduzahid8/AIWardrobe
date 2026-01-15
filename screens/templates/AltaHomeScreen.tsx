/**
 * ALTA DAILY - PIXEL PERFECT HOME SCREEN
 * Based on exact design specification from screenshots
 */

import React, { useState, useCallback, ReactNode } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    TouchableOpacity,
    ActivityIndicator,
    StatusBar,
    ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as Haptics from 'expo-haptics';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS from design spec
const ALTA = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#000000',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
    toastBg: '#1C1C1E',
    toastProgress: '#FFFFFF',
};

// Grid: 4 columns, 8px gap
const GRID_GAP = 8;
const GRID_COLUMNS = 4;
const ITEM_WIDTH = (width - 32 - (GRID_GAP * (GRID_COLUMNS - 1))) / GRID_COLUMNS;

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://aiwardrobe-ivh4.onrender.com';

// ============================================
// TYPE DEFINITIONS
// ============================================

type RootStackParamList = {
    Home: undefined;
    WardrobeVideo: undefined;
    AITryOn: { selectedItem?: WardrobeItemType };
};

interface WardrobeItemType {
    _id?: string;
    id?: string;
    imageUrl?: string;
    image?: string;
    type?: string;
    itemType?: string;
    name?: string;
    color?: string;
    style?: string;
    category?: string;
}

interface PressableScaleProps {
    children: ReactNode;
    onPress?: () => void;
    style?: ViewStyle;
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

// Wardrobe Grid Item - Exact Alta style
const WardrobeItem = ({ item, onPress }: { item: WardrobeItemType; onPress: () => void }) => {
    const imageUrl = item.imageUrl || item.image;
    const itemName = item.type || item.itemType || item.name || 'Item';

    return (
        <PressableScale style={styles.gridItem} onPress={onPress}>
            <View style={styles.itemImageBox}>
                {imageUrl ? (
                    <Image
                        source={{ uri: imageUrl }}
                        style={styles.itemImage}
                        resizeMode="contain"
                    />
                ) : (
                    <Ionicons name="shirt-outline" size={24} color={ALTA.textMuted} />
                )}
            </View>
            <Text style={styles.itemName} numberOfLines={2}>{itemName}</Text>
        </PressableScale>
    );
};

// Progress Toast - Exact Alta style (dark toast at bottom)
const ProgressToast = ({ current, target }: { current: number; target: number }) => {
    const progress = Math.min(current / target, 1);
    const remaining = Math.max(target - current, 0);

    if (current >= target) return null;

    return (
        <View style={styles.progressToast}>
            <Text style={styles.progressText}>
                Add {remaining} item{remaining !== 1 ? 's' : ''} to unlock personalized daily looks
            </Text>
            <View style={styles.progressBarBg}>
                <View style={[styles.progressBar, { width: `${progress * 100}%` }]} />
            </View>
            <Text style={styles.progressCount}>{current}/{target} items</Text>
        </View>
    );
};

const AltaHomeScreen = () => {
    const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
    const [items, setItems] = useState<WardrobeItemType[]>([]);
    const [loading, setLoading] = useState(true);

    const loadItems = useCallback(async () => {
        try {
            setLoading(true);
            const token = await AsyncStorage.getItem('userToken');

            if (token) {
                const response = await axios.get(`${API_URL}/clothing-items`, {
                    headers: { Authorization: `Bearer ${token}` },
                    timeout: 10000,
                });
                const data = Array.isArray(response.data) ? response.data : response.data?.items || [];
                setItems(data);
            } else {
                const local = await AsyncStorage.getItem('wardrobeItems');
                if (local) setItems(JSON.parse(local));
            }
        } catch (e) {
            console.log('Load error, using local:', e);
            const local = await AsyncStorage.getItem('wardrobeItems');
            if (local) setItems(JSON.parse(local));
        } finally {
            setLoading(false);
        }
    }, []);

    useFocusEffect(useCallback(() => { loadItems(); }, [loadItems]));

    const handleItemPress = (item: WardrobeItemType) => {
        navigation.navigate('AITryOn', { selectedItem: item });
    };

    const handleScanPress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        navigation.navigate('WardrobeVideo');
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={ALTA.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header - Exact Alta layout */}
                <View style={styles.header}>
                    {/* Left: Weather */}
                    <View style={styles.weatherWidget}>
                        <Ionicons name="partly-sunny" size={14} color={ALTA.textMuted} />
                        <Text style={styles.weatherTemp}>18°</Text>
                    </View>

                    {/* Center: ALTA logo */}
                    <Text style={styles.logo}>ALTA</Text>

                    {/* Right: Icons */}
                    <View style={styles.headerRight}>
                        <TouchableOpacity style={styles.headerIcon}>
                            <Ionicons name="search-outline" size={20} color={ALTA.text} />
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.headerIcon}>
                            <Ionicons name="person-circle-outline" size={22} color={ALTA.text} />
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Content */}
                <ScrollView
                    style={styles.scrollView}
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {loading ? (
                        <View style={styles.centerContent}>
                            <ActivityIndicator size="large" color={ALTA.text} />
                        </View>
                    ) : items.length === 0 ? (
                        <View style={styles.centerContent}>
                            <Ionicons name="shirt-outline" size={64} color={ALTA.textMuted} />
                            <Text style={styles.emptyTitle}>Your closet is empty</Text>
                            <Text style={styles.emptySubtitle}>Scan your wardrobe to get started</Text>
                            <TouchableOpacity style={styles.scanButton} onPress={handleScanPress}>
                                <Text style={styles.scanButtonText}>Scan Wardrobe</Text>
                            </TouchableOpacity>
                        </View>
                    ) : (
                        <View style={styles.grid}>
                            {items.map((item, index) => (
                                <WardrobeItem
                                    key={item._id || item.id || index}
                                    item={item}
                                    onPress={() => handleItemPress(item)}
                                />
                            ))}
                        </View>
                    )}

                    <View style={{ height: 140 }} />
                </ScrollView>

                {/* Progress Toast */}
                <ProgressToast current={items.length} target={5} />

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

    // Header - Exact specs: 16px horizontal, 10px vertical
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 10,
    },
    weatherWidget: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    weatherTemp: {
        fontSize: 12,
        fontWeight: '500',
        color: ALTA.text,
    },
    logo: {
        fontSize: 16,
        fontWeight: '800', // Extra bold
        letterSpacing: 3,
        color: ALTA.text,
    },
    headerRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    headerIcon: {
        padding: 4,
    },

    // Content
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: 16,
        paddingTop: 8,
    },

    // 4-column grid with 8px gap
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: GRID_GAP,
    },
    gridItem: {
        width: ITEM_WIDTH,
        marginBottom: 8,
    },
    itemImageBox: {
        width: '100%',
        aspectRatio: 1,
        backgroundColor: ALTA.surface, // #F5F5F5
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    itemImage: {
        width: '90%',
        height: '90%',
    },
    itemName: {
        fontSize: 11, // Exact from spec
        fontWeight: '400',
        color: ALTA.text,
        textAlign: 'center',
        marginTop: 4,
        lineHeight: 13,
    },

    // Empty state
    centerContent: {
        paddingVertical: 100,
        alignItems: 'center',
    },
    emptyTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: ALTA.text,
        marginTop: 16,
    },
    emptySubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        marginTop: 4,
    },
    scanButton: {
        marginTop: 24,
        backgroundColor: ALTA.text,
        paddingHorizontal: 32,
        paddingVertical: 14,
        borderRadius: 28,
    },
    scanButtonText: {
        fontSize: 15,
        fontWeight: '600',
        color: ALTA.background,
    },

    // Progress Toast - Dark toast, exact specs
    progressToast: {
        position: 'absolute',
        bottom: 16,
        left: 16,
        right: 16,
        backgroundColor: ALTA.toastBg, // #1C1C1E
        borderRadius: 16,
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    progressText: {
        fontSize: 13,
        fontWeight: '500',
        color: ALTA.toastProgress,
        marginBottom: 12,
    },
    progressBarBg: {
        height: 4,
        backgroundColor: '#333333',
        borderRadius: 2,
        overflow: 'hidden',
    },
    progressBar: {
        height: '100%',
        backgroundColor: ALTA.toastProgress,
        borderRadius: 2,
    },
    progressCount: {
        fontSize: 11,
        color: ALTA.textMuted,
        textAlign: 'center',
        marginTop: 8,
    },
});

export default AltaHomeScreen;
