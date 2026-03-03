/**
 * ALTA DAILY - PIXEL PERFECT FAVORITES SCREEN
 * Based on exact design specification
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
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS
const ALTA = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#0A1931',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
    heart: '#FF3B30',
};

const GRID_GAP = 12;
const ITEM_WIDTH = (width - 32 - GRID_GAP) / 2;

// Type for favorite item
interface FavoriteItemType {
    _id?: string;
    id?: string;
    name?: string;
    type?: string;
    brand?: string;
    category?: string;
    imageUrl?: string;
    image?: string;
}

// Type for favorite item props
interface FavoriteItemProps {
    item: FavoriteItemType;
    onPress: () => void;
    onRemove: (item: FavoriteItemType) => void;
}

// Favorite Item Card with scale animation
const FavoriteItem = ({ item, onPress, onRemove }: FavoriteItemProps) => {
    const scale = useSharedValue(1);
    const imageUrl = item.imageUrl || item.image;

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <Animated.View entering={FadeInUp.springify()}>
            <TouchableOpacity
                activeOpacity={1}
                onPressIn={() => {
                    scale.value = withSpring(0.97);
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                }}
                onPressOut={() => scale.value = withSpring(1)}
                onPress={onPress}
            >
                <Animated.View style={[styles.favoriteCard, animatedStyle]}>
                    <View style={styles.favoriteImageBox}>
                        {imageUrl ? (
                            <Image source={{ uri: imageUrl }} style={styles.favoriteImage} resizeMode="cover" />
                        ) : (
                            <Ionicons name="shirt-outline" size={32} color={ALTA.textMuted} />
                        )}
                        <TouchableOpacity
                            style={styles.heartButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                onRemove(item);
                            }}
                        >
                            <Ionicons name="heart" size={18} color={ALTA.heart} />
                        </TouchableOpacity>
                    </View>
                    <Text style={styles.favoriteTitle} numberOfLines={1}>
                        {item.name || item.type || 'Saved Look'}
                    </Text>
                    <Text style={styles.favoriteSubtitle} numberOfLines={1}>
                        {item.brand || item.category || 'From your closet'}
                    </Text>
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

const AltaFavoritesScreen = () => {
    const navigation = useNavigation();
    const [favorites, setFavorites] = useState<FavoriteItemType[]>([]);

    const loadFavorites = useCallback(async () => {
        try {
            const saved = await AsyncStorage.getItem('favoriteItems');
            if (saved) setFavorites(JSON.parse(saved));
        } catch (e) { }
    }, []);

    useFocusEffect(useCallback(() => { loadFavorites(); }, [loadFavorites]));

    const removeFavorite = (item: FavoriteItemType) => {
        Alert.alert(
            'Remove Favorite',
            'Are you sure you want to remove this item from favorites?',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Remove',
                    style: 'destructive',
                    onPress: async () => {
                        const updated = favorites.filter(f => (f._id || f.id) !== (item._id || item.id));
                        setFavorites(updated);
                        await AsyncStorage.setItem('favoriteItems', JSON.stringify(updated));
                    },
                },
            ]
        );
    };

    const handleItemPress = (item: FavoriteItemType) => {
        (navigation as any).navigate('AITryOn', { selectedItem: item });
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={ALTA.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header */}
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>Favorites</Text>
                </View>

                <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                    {favorites.length === 0 ? (
                        <View style={styles.emptyContainer}>
                            <Ionicons name="heart-outline" size={64} color={ALTA.textMuted} />
                            <Text style={styles.emptyTitle}>No favorites yet</Text>
                            <Text style={styles.emptySubtitle}>Heart items to save them here</Text>
                            <TouchableOpacity
                                style={styles.browseButton}
                                onPress={() => (navigation as any).navigate('Home')}
                            >
                                <Text style={styles.browseButtonText}>Browse Closet</Text>
                            </TouchableOpacity>
                        </View>
                    ) : (
                        <View style={styles.grid}>
                            {favorites.map((item, index) => (
                                <FavoriteItem
                                    key={item._id || item.id || index}
                                    item={item}
                                    onPress={() => handleItemPress(item)}
                                    onRemove={removeFavorite}
                                />
                            ))}
                        </View>
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: ALTA.background },
    safeArea: { flex: 1 },

    // Header
    header: {
        paddingHorizontal: 20,
        paddingVertical: 16,
        borderBottomWidth: 0.5,
        borderBottomColor: ALTA.border,
    },
    headerTitle: { fontSize: 28, fontWeight: '700', color: ALTA.text },

    // Content
    scrollContent: { padding: 16 },

    // Grid
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: GRID_GAP,
    },

    // Favorite Card
    favoriteCard: {
        width: ITEM_WIDTH,
        marginBottom: 8,
    },
    favoriteImageBox: {
        width: '100%',
        aspectRatio: 0.8,
        backgroundColor: ALTA.surface,
        borderRadius: 12,
        overflow: 'hidden',
        marginBottom: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    favoriteImage: { width: '100%', height: '100%' },
    heartButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: ALTA.background,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    favoriteTitle: { fontSize: 14, fontWeight: '600', color: ALTA.text },
    favoriteSubtitle: { fontSize: 12, color: ALTA.textSecondary, marginTop: 2 },

    // Empty
    emptyContainer: { paddingVertical: 100, alignItems: 'center' },
    emptyTitle: { fontSize: 20, fontWeight: '600', color: ALTA.text, marginTop: 16 },
    emptySubtitle: { fontSize: 14, color: ALTA.textSecondary, marginTop: 4 },
    browseButton: {
        marginTop: 24,
        backgroundColor: ALTA.text,
        paddingHorizontal: 32,
        paddingVertical: 14,
        borderRadius: 28,
    },
    browseButtonText: { fontSize: 15, fontWeight: '600', color: ALTA.background },
});

export default AltaFavoritesScreen;
