import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    FlatList,
    Image,
    ActivityIndicator,
    Alert,
    TextInput,
    StatusBar,
    Platform,
    Linking,
} from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useIsFocused } from '@react-navigation/native';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    FadeOut,
    FadeOutDown,
    ZoomIn,
    ZoomOut,
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
    Easing
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import Config from '../src/config/env';
import useAuthStore from '../store/auth';
import { ExternalAIService } from '../src/services/externalAIService';
import { BASIC_CLOTHING_ITEMS } from '../data/basicClothingItems';
import { createLogger } from '../src/utils/logger';
import { useTranslation } from 'react-i18next';

const logger = createLogger('MyCloset');

const GLASS = {
    bg: 'rgba(255, 255, 255, 0.55)',
    bgLight: 'rgba(255, 255, 255, 0.35)',
    border: 'rgba(255, 255, 255, 0.7)',
    accent: '#007AFF',
    accentGlow: 'rgba(0, 122, 255, 0.25)',
    textPrimary: '#1C1C1E',
    textSecondary: 'rgba(60, 60, 67, 0.6)',
};

const LiquidGlassSpinner = () => {
    const rotation = useSharedValue(0);
    const pulse = useSharedValue(1);
    const innerPulse = useSharedValue(0.6);

    useEffect(() => {
        rotation.value = withRepeat(withTiming(360, { duration: 2400, easing: Easing.linear }), -1, false);
        pulse.value = withRepeat(
            withSequence(
                withTiming(1.08, { duration: 1200, easing: Easing.inOut(Easing.sin) }),
                withTiming(1, { duration: 1200, easing: Easing.inOut(Easing.sin) })
            ), -1, true
        );
        innerPulse.value = withRepeat(
            withSequence(
                withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
                withTiming(0.6, { duration: 1800, easing: Easing.inOut(Easing.sin) })
            ), -1, true
        );
    }, []);

    const outerStyle = useAnimatedStyle(() => ({ transform: [{ scale: pulse.value }] }));
    const ringStyle = useAnimatedStyle(() => ({ transform: [{ rotate: `${rotation.value}deg` }] }));
    const glowStyle = useAnimatedStyle(() => ({ opacity: innerPulse.value }));

    return (
        <Animated.View style={[styles.spinnerContainer, outerStyle]}>
            <Animated.View style={[styles.spinnerGlow, glowStyle]} />
            <Animated.View style={[styles.spinnerRing, ringStyle]}>
                <LinearGradient
                    colors={['rgba(0,122,255,0.6)', 'rgba(0,122,255,0)', 'rgba(0,122,255,0.3)']}
                    start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
                    style={styles.spinnerRingGradient}
                />
            </Animated.View>
            <BlurView intensity={40} tint="light" style={styles.spinnerInner}>
                <Ionicons name="sparkles" size={28} color={GLASS.accent} />
            </BlurView>
        </Animated.View>
    );
};

const { width, height } = Dimensions.get('window');
const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// ── AI clothing analysis helpers ────────────────────────────────────────────
const mapCategoryToType = (category: string, section: string): string => {
    const cat = category.toLowerCase();
    const sec = section.toLowerCase();
    if (sec === 'tops' || cat.includes('shirt') || cat.includes('blouse') || cat.includes('sweater') || cat.includes('hoodie') || cat === 't-shirt') return 'tops';
    if (sec === 'bottoms' || cat.includes('pant') || cat.includes('jean') || cat.includes('skirt') || cat.includes('short')) return 'bottoms';
    if (sec === 'shoes' || cat.includes('shoe') || cat.includes('sneaker') || cat.includes('boot') || cat.includes('sandal')) return 'shoes';
    if (sec === 'accessories' || cat.includes('bag') || cat.includes('hat') || cat.includes('scarf') || cat.includes('belt') || cat.includes('watch')) return 'accessories';
    if (sec === 'outerwear' || cat.includes('jacket') || cat.includes('coat')) return 'outerwear';
    if (cat.includes('sport') || cat.includes('gym') || cat.includes('legging')) return 'sportswear';
    return 'tops';
};

const mapColorToId = (colorName: string): string => {
    const name = (colorName || '').toLowerCase();
    if (name.includes('black') || name.includes('charcoal') || name.includes('ebony')) return 'black';
    if (name.includes('grey') || name.includes('gray') || name.includes('silver')) return 'grey';
    if (name.includes('beige') || name.includes('cream') || name.includes('tan') || name.includes('khaki') || name.includes('sand')) return 'beige';
    if (name.includes('white') || name.includes('off-white') || name.includes('ivory')) return 'white';
    if (name.includes('brown') || name.includes('camel') || name.includes('chocolate') || name.includes('rust')) return 'brown';
    if (name.includes('green') || name.includes('olive') || name.includes('forest') || name.includes('mint')) return 'green';
    if (name.includes('red') || name.includes('burgundy') || name.includes('wine') || name.includes('pink') || name.includes('coral') || name.includes('maroon')) return 'red';
    if (name.includes('blue') || name.includes('navy') || name.includes('indigo') || name.includes('denim') || name.includes('cobalt') || name.includes('teal')) return 'blue';
    return 'beige';
};

interface ClothingItem {
    _id: string;
    id?: string;
    type?: string;
    itemType?: string;
    color?: string;
    colorHex?: string;
    style?: string;
    description?: string;
    imageUrl?: string;
    image?: string;
    category?: string;
    wearCount?: number;
    lastWorn?: string;
    createdAt?: string;
    isFavorite?: boolean; // Added for favorite filter
}

// Category Chip from Design
const FilterChip = ({
    label,
    isSelected,
    onPress
}: {
    label: string;
    isSelected: boolean;
    onPress: () => void;
}) => {
    return (
        <TouchableOpacity
            style={[
                styles.filterChip,
                isSelected ? styles.filterChipSelected : styles.filterChipUnselected
            ]}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.8}
            accessibilityLabel={`${label} filter`}
            accessibilityRole="button"
            accessibilityState={{ selected: isSelected }}
        >
            <Text style={[
                styles.filterChipText,
                isSelected ? styles.filterChipTextSelected : styles.filterChipTextUnselected
            ]}>
                {label}
            </Text>
        </TouchableOpacity>
    );
};

// Premium Clothing Grid Item
const ClothingGridItem = ({
    item,
    onPress,
    t
}: {
    item: ClothingItem;
    onPress: () => void;
    t: (key: string) => string;
}) => {
    let imageUrl = item.imageUrl || item.image;
    let finalImageSource: { uri: string } | null = imageUrl ? { uri: imageUrl } : null;

    if (imageUrl && imageUrl.startsWith('basic_clothing_')) {
        const basicId = imageUrl.replace('basic_clothing_', '');
        const basicItem = BASIC_CLOTHING_ITEMS.find(b => b.id === basicId);
        if (basicItem && basicItem.image) {
            finalImageSource = { uri: basicItem.image };
        }
    }

    return (
        <TouchableOpacity
            style={styles.gridItemContainer}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.9}
            accessibilityLabel={`${item.category || item.type || t('wardrobe.clothing')} item${item.color ? `, ${item.color}` : ''}`}
            accessibilityRole="button"
        >
            <Animated.View style={styles.gridItem}>
                <View style={styles.imageContainer}>
                    {finalImageSource ? (
                        <Image
                            source={finalImageSource}
                            style={styles.itemImage}
                            resizeMode="cover"
                        />
                    ) : (
                        <View style={styles.itemPlaceholder}>
                            <Ionicons name="shirt-outline" size={32} color={colors.text.tertiary} />
                        </View>
                    )}
                </View>
            </Animated.View>
        </TouchableOpacity>
    );
};

const MyClosetScreen = () => {
    const navigation = useAppNavigation();
    const isFocused = useIsFocused();
    const { t } = useTranslation();
    const [items, setItems] = useState<ClothingItem[]>([]);

    // Updated Category filters to match design
    const CATEGORIES = [
        { id: 'favorite', label: t('closet.favorite'), icon: 'heart' },
        { id: 'all', label: t('closet.all'), icon: 'grid' },
        { id: 'tops', label: t('closet.tops'), icon: 'shirt' },
        { id: 'bottoms', label: t('closet.bottoms'), icon: 'bookmark' },
        { id: 'shoes', label: t('closet.shoes'), icon: 'footsteps' },
        { id: 'accessories', label: t('closet.accessories'), icon: 'glasses' },
    ];

    const player = useVideoPlayer(require('../assets/videos/closet.mov'), (player) => {
        player.loop = true;
        player.muted = true;
        player.play();
    });

    useEffect(() => {
        if (isFocused) {
            player.play();
        } else {
            player.pause();
        }
    }, [isFocused, player]);
    const [filteredItems, setFilteredItems] = useState<ClothingItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCategory, setSelectedCategory] = useState('tops'); // Default from screenshot
    const [viewMode, setViewMode] = useState<'clothes' | 'collections'>('clothes');
    const [searchQuery, setSearchQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [filterCategory, setFilterCategory] = useState('All');

    // AI Studio State
    const { user } = useAuthStore();
    const [isUploadingOverlay, setIsUploadingOverlay] = useState(false);
    const [uploadStatusMsg, setUploadStatusMsg] = useState('');

    // AI Studio Upload & Auto-Generate Logic (Serverless - External AI)
    const pickImage = async (useCamera = false) => {
        try {
            const options: ImagePicker.ImagePickerOptions = {
                mediaTypes: ['images'],
                allowsEditing: true,
                quality: 0.8,
                base64: true,
            };

            let result;
            if (useCamera) {
                const current = await ImagePicker.getCameraPermissionsAsync();
                if (!current.granted) {
                    if (!current.canAskAgain) {
                        Alert.alert(
                            t('myCloset.cameraAccessDisabled'),
                            t('myCloset.enableCameraAccess'),
                            [
                                { text: t('common.cancel'), style: 'cancel' },
                                {
                                    text: t('myCloset.openSettings'),
                                    onPress: () => Platform.OS === 'ios' ? Linking.openURL('app-settings:') : Linking.openSettings(),
                                },
                            ]
                        );
                        return;
                    }
                    const { status } = await ImagePicker.requestCameraPermissionsAsync();
                    if (status !== 'granted') {
                        Alert.alert(t('myCloset.permissionNeeded'), t('myCloset.cameraPermissionRequired'));
                        return;
                    }
                }
                result = await ImagePicker.launchCameraAsync(options);
            } else {
                result = await ImagePicker.launchImageLibraryAsync(options);
            }

            if (!result.canceled && result.assets[0]) {
                const asset = result.assets[0];
                runMagicPipeline(asset.base64 || null);
            }
        } catch (error: any) {
            const msg: string = error?.message ?? '';
            if (msg.toLowerCase().includes('simulator') || msg.toLowerCase().includes('not available')) {
                Alert.alert(t('myCloset.simulatorDetected'), t('myCloset.cameraNotAvailable'));
            } else {
                console.error('Image picker error:', error);
                Alert.alert(t('common.error'), t('myCloset.failedOpenCamera'));
            }
        }
    };

    const runMagicPipeline = async (b64: string | null) => {
        if (!b64) {
            Alert.alert(t('common.error'), t('myCloset.couldNotReadImage'));
            return;
        }

        // Ensure base64 has proper data URI prefix for the API
        const imageData = b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`;

        setIsUploadingOverlay(true);
        setUploadStatusMsg(t('myCloset.analyzingItem'));

        try {
            console.log('[MyCloset] Starting AI Studio photo processing...');
            
            // AI Studio should preserve the garment and only remove the background.
            const result = await ExternalAIService.processStudioPhoto(imageData);

            console.log('[MyCloset] AI Studio result:', {
                success: result.success,
                hasImageUrl: !!result.imageUrl,
                hasCutoutUrl: !!result.cutoutUrl,
                hasClassification: !!result.classification,
                processingTimeMs: result.processingTimeMs,
            });

            if (!result.success) {
                throw new Error("AI processing failed");
            }

            // Validate that we have a valid image URL
            if (!result.imageUrl) {
                throw new Error("No processed image returned from AI");
            }

            setUploadStatusMsg(t('myCloset.backgroundRemoved'));

            const cat = (result.classification?.category || '').toLowerCase();
            const sec = (result.classification?.section || '').toLowerCase();

            setIsUploadingOverlay(false);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

            // Navigate to editor for user review before saving
            navigation.navigate('ClothingDetailEditor', {
                imageUri: result.imageUrl,
                detectedType: mapCategoryToType(cat, sec),
                detectedColor: mapColorToId(result.classification?.attributes?.color || ''),
                detectedStyle: result.classification?.attributes?.style,
                detectedMaterial: result.classification?.attributes?.material ?? undefined,
                aiConfidence: result.classification?.confidence,
                detectedDescription: result.description ?? undefined,
            });

        } catch (error: any) {
            console.error('[MyCloset] AI Studio error:', error);
            Alert.alert(
                t('common.error'), 
                error.message || t('myCloset.uploadError'),
                [{ text: t('common.ok'), style: 'default' }]
            );
        } finally {
            setIsUploadingOverlay(false);
        }
    };

    const openLegacyCamera = async () => {
        try {
            const { status } = await ImagePicker.requestCameraPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert(t('myCloset.permissionNeeded'), t('myCloset.cameraPermissionRequired'));
                return;
            }

            const result = await ImagePicker.launchCameraAsync({
                mediaTypes: ['images'],
                allowsEditing: false,
                quality: 0.8,
                base64: true,
            });

            if (!result.canceled && result.assets[0]) {
                const asset = result.assets[0];
                setIsUploadingOverlay(true);
                setUploadStatusMsg(t('myCloset.analyzingClothing'));
                try {
                    const aiResult = await ExternalAIService.classifyOnly(asset.base64 || '');
                    const cat = (aiResult.classification?.category || '').toLowerCase();
                    const sec = (aiResult.classification?.section || '').toLowerCase();
                    navigation.navigate('ClothingDetailEditor', {
                        imageUri: asset.uri,
                        detectedType: mapCategoryToType(cat, sec),
                        detectedColor: mapColorToId(aiResult.classification?.attributes?.color || ''),
                        detectedStyle: aiResult.classification?.attributes?.style,
                        detectedMaterial: aiResult.classification?.attributes?.material ?? undefined,
                        aiConfidence: aiResult.classification?.confidence,
                    });
                } catch {
                    navigation.navigate('ClothingDetailEditor', { imageUri: asset.uri });
                } finally {
                    setIsUploadingOverlay(false);
                }
            }
        } catch (error: any) {
            const msg: string = error?.message ?? '';
            if (msg.toLowerCase().includes('simulator') || msg.toLowerCase().includes('not available')) {
                Alert.alert(t('myCloset.simulatorDetected'), t('myCloset.cameraNotAvailable'));
            } else {
                console.error('Legacy camera error:', error);
                Alert.alert(t('common.error'), t('myCloset.failedOpenCamera'));
            }
        }
    };

    const handleUploadChoice = () => {
        Alert.alert(
            t('myCloset.addToCloset'),
            t('myCloset.uploadPhotoDescription'),
            [
                { text: t('myCloset.aiStudioPhoto'), onPress: () => pickImage(false) },
                { text: t('myCloset.aiStudioCamera'), onPress: () => pickImage(true) },
                { text: t('myCloset.legacyCamera'), onPress: openLegacyCamera },
                { text: t('common.cancel'), style: "cancel" }
            ]
        );
    };

    const confirmDelete = (item: ClothingItem) => {
        Alert.alert(
            t('myCloset.deleteItem'),
            t('myCloset.deleteItemConfirmation'),
            [
                { text: t('common.cancel'), style: "cancel" },
                {
                    text: t('common.delete'),
                    style: "destructive",
                    onPress: () => deleteItem(item)
                }
            ]
        );
    };

    const deleteItem = async (item: ClothingItem) => {
        const idToDelete = item._id || item.id;
        if (!idToDelete) return;

        try {
            // Optimistic UI update
            setItems(prev => prev.filter(i => (i._id || i.id) !== idToDelete));

            const { error } = await supabase
                .from('clothing_items')
                .delete()
                .eq('id', idToDelete);

            if (error) throw error;
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (error) {
            console.error('Failed to delete item:', error);
            Alert.alert(t('common.error'), t('myCloset.failedDeleteItem'));
            loadItems(); // Revert on failure
        }
    };

    // Load wardrobe items
    const loadItems = useCallback(async () => {
        try {
            setLoading(true);

            // Fetch from Supabase
            const { data, error } = await supabase
                .from('clothing_items')
                .select('*')
                .order('created_at', { ascending: false });

            if (data) {
                // Map snake_case to camelCase
                const mappedItems: ClothingItem[] = data.map(item => ({
                    _id: item.id,
                    id: item.id,
                    type: item.type,
                    itemType: item.type,
                    color: item.color && item.color.length > 0 ? item.color[0] : 'various',
                    imageUrl: item.image_url,
                    image: item.image_url,
                    category: item.category,
                    wearCount: item.wear_count,
                    createdAt: item.created_at,
                    isFavorite: false, // Default for now
                }));

                setItems(mappedItems);
                logger.debug(`Loaded ${mappedItems.length} items`, {
                    images: mappedItems.map(i => ({ type: i.type, hasImage: !!i.imageUrl, imgLen: i.imageUrl?.length || 0 })),
                });
            }
        } catch (error) {
            console.error('Failed to load wardrobe:', error);
            // local storage
            const localItems = await AsyncStorage.getItem('myWardrobeItems');
            if (localItems) {
                setItems(JSON.parse(localItems));
            }
        } finally {
            setLoading(false);
        }
    }, []);

    useFocusEffect(
        useCallback(() => {
            loadItems();
        }, [loadItems])
    );

    // Filter items
    useEffect(() => {
        let result = [...items];

        if (viewMode === 'clothes') {
            if (selectedCategory !== 'all') {
                if (selectedCategory === 'favorite') {
                    result = result.filter(item => item.isFavorite);
                } else {
                    result = result.filter(item => {
                        const cat = (item.category || '').toLowerCase();
                        const type = (item.type || item.itemType || '').toLowerCase();
                        
                        // Use exact word matching with word boundaries to prevent substring matches
                        const matchesExact = (value: string, keywords: string[]): boolean => {
                            return keywords.some(keyword => {
                                const regex = new RegExp(`\\b${keyword}\\b`, 'i');
                                return regex.test(value);
                            });
                        };

                        switch (selectedCategory) {
                            case 'tops':
                                if (cat === 'tops' || cat === 'top') return true;
                                return matchesExact(cat, ['shirt', 'blouse', 'pullover', 'hoodie', 'sweater', 't-shirt', 'tshirt', 'polo', 'cardigan']) ||
                                       matchesExact(type, ['shirt', 'blouse', 'pullover', 'hoodie', 'sweater', 't-shirt', 'tshirt', 'polo', 'cardigan']);
                            case 'bottoms':
                                if (cat === 'bottoms' || cat === 'bottom') return true;
                                return matchesExact(cat, ['pant', 'skirt', 'jean', 'trouser', 'short', 'legging']) ||
                                       matchesExact(type, ['pant', 'skirt', 'jean', 'trouser', 'short', 'legging']);
                            case 'shoes':
                                if (cat === 'shoes' || cat === 'shoe') return true;
                                return matchesExact(cat, ['sneaker', 'boot', 'sandal', 'heel', 'loafer', 'slipper']) ||
                                       matchesExact(type, ['sneaker', 'boot', 'sandal', 'heel', 'loafer', 'slipper']);
                            case 'accessories':
                                if (cat === 'accessories' || cat === 'accessory') return true;
                                return matchesExact(cat, ['bag', 'hat', 'scarf', 'belt', 'sunglasses', 'watch', 'jewelry']) ||
                                       matchesExact(type, ['bag', 'hat', 'scarf', 'belt', 'sunglasses', 'watch', 'jewelry']);
                            default:
                                return cat === selectedCategory || type === selectedCategory;
                        }
                    });
                }
            }
        }

        // Text Search
        if (searchQuery.trim().length > 0) {
            const query = searchQuery.toLowerCase();
            result = result.filter(item => {
                const nameMatch = (item.type || item.itemType || item.category || '').toLowerCase().includes(query);
                const colorMatch = (item.color || '').toLowerCase().includes(query);
                const descMatch = (item.description || '').toLowerCase().includes(query);
                return nameMatch || colorMatch || descMatch;
            });
        }

        setFilteredItems(result);
    }, [items, selectedCategory, viewMode, searchQuery]);


    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
                {/* Header */}
                <View style={styles.headerContainer}>
                    {isSearching ? (
                        <View style={styles.searchBarContainer}>
                            <Ionicons name="search" size={20} color={colors.text.secondary} />
                            <TextInput
                                style={styles.searchBarInput}
                                placeholder={t('myCloset.searchCloset')}
                                value={searchQuery}
                                onChangeText={setSearchQuery}
                                autoFocus
                                returnKeyType="search"
                            />
                            <TouchableOpacity onPress={() => { setIsSearching(false); setSearchQuery(''); }}>
                                <Ionicons name="close-circle" size={18} color={colors.text.tertiary} />
                            </TouchableOpacity>
                        </View>
                    ) : (
                        <View style={styles.headerContent}>
                            <View style={styles.headerLeft}>
                                {items.length > 0 && (
                                    <TouchableOpacity
                                        style={styles.headerIconButton}
                                        onPress={() => setIsSearching(true)}
                                        accessibilityLabel={t('myCloset.searchCloset')}
                                        accessibilityRole="button"
                                    >
                                        <Ionicons name="search" size={22} color={colors.text.secondary} />
                                    </TouchableOpacity>
                                )}
                            </View>

                            <View style={styles.headerCenter}>
                                <Text style={styles.headerTitle} accessibilityRole="header">{t('wardrobe.title')}</Text>
                            </View>

                            <View style={styles.headerRight}>
                                {items.length > 0 && (
                                    <TouchableOpacity
                                        style={styles.headerUploadButton}
                                        onPress={handleUploadChoice}
                                        accessibilityLabel={t('myCloset.uploadClothing')}
                                        accessibilityRole="button"
                                    >
                                        <Ionicons name="add" size={18} color="#0A1931" />
                                        <Text style={styles.headerUploadText}>{t('wardrobe.upload')}</Text>
                                    </TouchableOpacity>
                                )}
                            </View>
                        </View>
                    )}
                </View>


                {/* Segmented Control */}
                <View style={styles.segmentContainer}>
                    <View style={styles.segmentBackground}>
                        <TouchableOpacity
                            style={[styles.segmentButton, viewMode === 'clothes' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setViewMode('clothes');
                            }}
                            accessibilityLabel={t('wardrobe.clothes')}
                            accessibilityRole="tab"
                            accessibilityState={{ selected: viewMode === 'clothes' }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'clothes' && styles.segmentTextActive]}>{t('wardrobe.clothes')}</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.segmentButton, viewMode === 'collections' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setViewMode('collections');
                            }}
                            accessibilityLabel={t('wardrobe.collections')}
                            accessibilityRole="tab"
                            accessibilityState={{ selected: viewMode === 'collections' }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'collections' && styles.segmentTextActive]}>{t('wardrobe.collections')}</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Filter Chips */}
                {viewMode === 'clothes' && (
                    <View style={styles.filterContainer}>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.filterContentRaw}
                        >
                            {CATEGORIES.map((cat) => (
                                <FilterChip
                                    key={cat.id}
                                    label={cat.label}
                                    isSelected={selectedCategory === cat.id}
                                    onPress={() => setSelectedCategory(cat.id)}
                                />
                            ))}
                        </ScrollView>
                    </View>
                )}

                {/* Content */}
                {viewMode === 'collections' ? (
                    /* ── Collections View ── */
                    <View style={styles.emptyStateContainer}>
                        <View style={styles.collectionsIconWrap}>
                            <Ionicons name="albums-outline" size={56} color={colors.text.tertiary} />
                        </View>
                        <Text style={styles.emptyTitle}>{t('wardrobe.noCollections')}</Text>
                        <Text style={styles.emptySubtitle}>
                            Group your outfits into collections — for work, weekends, seasons, or any occasion.
                        </Text>
                        <TouchableOpacity
                            style={styles.emptyButton}
                            onPress={() => navigation.navigate('AIOutfit', { source: 'wardrobe' })}
                            accessibilityLabel={t('myCloset.createFirstCollection')}
                            accessibilityRole="button"
                        >
                            <Text style={styles.emptyButtonText}>{t('wardrobe.createFirstLook')}</Text>
                        </TouchableOpacity>
                    </View>
                ) : loading ? (
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator size="small" color={colors.text.primary} />
                    </View>
                ) : items.length === 0 ? (
                    <View style={styles.emptyStateContainer}>
                        <View style={styles.videoContainer}>
                            <VideoView
                                style={styles.video}
                                player={player}
                                allowsFullscreen={false}
                                allowsPictureInPicture={false}
                                contentFit="contain"
                            />
                        </View>
                        <Text style={styles.emptyTitle}>{t('wardrobe.emptyCloset')}</Text>
                        <Text style={styles.emptySubtitle}>{t('wardrobe.emptyClosetSubtitle')}</Text>

                        <TouchableOpacity
                            style={styles.emptyButton}
                            onPress={handleUploadChoice}
                            accessibilityLabel={t('myCloset.scanWardrobe')}
                            accessibilityRole="button"
                        >
                            <Text style={styles.emptyButtonText}>{t('wardrobe.scanWardrobe')}</Text>
                        </TouchableOpacity>
                    </View>
                ) : (
                    <FlatList
                        data={filteredItems}
                        keyExtractor={(item, index) => item._id || item.id || String(index)}
                        numColumns={2}
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                        renderItem={({ item }) => (
                            <ClothingGridItem
                                item={item}
                                onPress={() => {
                                    Haptics.selectionAsync();
                                    navigation.navigate('ClothingDetail', {
                                        itemId: item.id || (item as any)._id,
                                        fullItem: item
                                    });
                                }}
                                t={t}
                            />
                        )}
                        ListFooterComponent={<View style={{ height: 100 }} />}
                    />
                )}

            </SafeAreaView>

            {/* Floating Ask Stylist button — Liquid Glass */}
            <TouchableOpacity
                style={styles.stylistFAB}
                onPress={() => navigation.navigate('AIOutfit', { source: 'wardrobe' })}
                activeOpacity={0.88}
                accessibilityLabel={t('myCloset.askAIStylist')}
                accessibilityRole="button"
            >
                <View style={styles.stylistFABGlass}>
                    <Ionicons name="chatbubble-ellipses" size={20} color={colors.text.primary} />
                    <Text style={styles.stylistFABText}>{t('wardrobe.askStylist')}</Text>
                </View>
            </TouchableOpacity>

            {isUploadingOverlay && (
                <Animated.View entering={FadeIn.duration(300)} style={StyleSheet.absoluteFill}>
                    <BlurView intensity={70} tint="light" style={styles.overlayBlur}>
                        <LiquidGlassSpinner />
                        <Animated.Text entering={FadeInUp.delay(200)} style={styles.overlayTitle}>
                            AI Studio
                        </Animated.Text>
                        <Animated.Text entering={FadeInUp.delay(300)} style={styles.overlaySubtitle}>
                            {uploadStatusMsg}
                        </Animated.Text>
                    </BlurView>
                </Animated.View>
            )}
        </View >
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
    },
    backgroundOrbTop: {
        position: 'absolute',
        top: -100,
        right: -80,
        width: 280,
        height: 280,
        borderRadius: 140,
        backgroundColor: 'rgba(188, 210, 245, 0.42)',
    },
    backgroundOrbBottom: {
        position: 'absolute',
        left: -120,
        bottom: 120,
        width: 300,
        height: 300,
        borderRadius: 150,
        backgroundColor: 'rgba(216, 229, 252, 0.34)',
    },
    safeArea: {
        flex: 1,
    },

    // Header
    headerContainer: {
        backgroundColor: 'transparent',
        marginTop: -5,
        paddingBottom: 10,
    },
    headerContent: {
        minHeight: 52,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
    },
    headerLeft: {
        flex: 1,
        alignItems: 'flex-start',
    },
    headerCenter: {
        flex: 2,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerRight: {
        flex: 1,
        alignItems: 'flex-end',
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: 0.3,
    },
    headerIconButton: {
        width: 44,
        height: 44,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.82)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    headerUploadButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.84)',
        paddingHorizontal: 14,
        paddingVertical: 9,
        borderRadius: 18,
        marginRight: 2,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    headerUploadText: {
        fontSize: 14,
        fontWeight: '600',
        color: '#0A1931',
        marginLeft: 2,
    },
    searchBarContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderRadius: 18,
        paddingHorizontal: 14,
        paddingVertical: 10,
        marginHorizontal: 16,
        marginVertical: 4,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    searchBarInput: {
        flex: 1,
        marginLeft: 8,
        fontSize: 16,
        color: '#000',
    },

    // Segmented Control
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 0,
        paddingBottom: 14,
        backgroundColor: 'transparent',
    },
    segmentBackground: {
        flexDirection: 'row',
        backgroundColor: 'rgba(255,255,255,0.84)',
        borderRadius: 26,
        padding: 4,
        width: 300,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    segmentButton: {
        flex: 1,
        paddingVertical: 10,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 20,
    },
    segmentButtonActive: {
        backgroundColor: '#FFFFFF',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 10,
        elevation: 2,
    },
    segmentText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#8E8E93',
    },
    segmentTextActive: {
        color: '#0A1931',
        fontWeight: '600',
    },

    // Filters
    filterContainer: {
        marginBottom: 18,
    },
    filterContentRaw: {
        paddingHorizontal: 16,
        paddingRight: 8,
    },
    filterChip: {
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 18,
        marginRight: 8,
        backgroundColor: 'rgba(255,255,255,0.86)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    filterChipSelected: {
        backgroundColor: '#173A65',
        borderColor: '#173A65',
    },
    filterChipUnselected: {
        backgroundColor: 'rgba(255,255,255,0.86)',
    },
    filterChipText: {
        fontSize: 15,
        fontWeight: '500',
    },
    filterChipTextSelected: {
        color: '#FFFFFF',
    },
    filterChipTextUnselected: {
        color: '#3C3C43',
    },

    // Grid
    scrollContent: {
        paddingHorizontal: 16,
        paddingBottom: 100,
        flexGrow: 1,
    },
    loadingContainer: {
        paddingTop: 50,
        alignItems: 'center',
    },
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -6, // Account for item padding
    },
    gridItemContainer: {
        width: '50%', // 2 columns
        padding: 6,
    },
    gridItem: {
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderRadius: 22,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.06)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.05,
        shadowRadius: 12,
        elevation: 3,
    },
    imageContainer: {
        aspectRatio: 3 / 4, // Portrait ratio for clothes
        backgroundColor: '#F9FBFF',
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemImage: {
        width: '100%',
        height: '100%',
    },
    itemPlaceholder: {
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Empty State
    emptyStateContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingTop: 60,
        marginHorizontal: 16,
        marginTop: 4,
        paddingHorizontal: 20,
        paddingBottom: 24,
        borderRadius: 30,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.72)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 20,
        elevation: 6,
    },
    collectionsIconWrap: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: '#F4F8FF',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    videoContainer: {
        width: 250,
        height: 250,
        marginBottom: 20,
    },
    video: {
        width: '100%',
        height: '100%',
    },
    emptyTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: '#0A1931',
        marginBottom: 8,
    },
    emptySubtitle: {
        fontSize: 15,
        color: '#8E8E93',
        textAlign: 'center',
        maxWidth: 260,
        marginBottom: 24,
    },
    emptyButton: {
        backgroundColor: '#173A65',
        paddingVertical: 14,
        paddingHorizontal: 32,
        borderRadius: 30,
    },
    emptyButtonText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: '600',
    },
    // Overlay logic
    overlayBlur: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 40,
    },
    overlayTitle: {
        fontSize: 24,
        fontWeight: '700',
        color: GLASS.textPrimary,
        marginBottom: 8,
    },
    overlaySubtitle: {
        fontSize: 16,
        color: GLASS.textSecondary,
        textAlign: 'center',
        lineHeight: 22,
    },
    spinnerContainer: {
        width: 110,
        height: 110,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 28,
    },
    spinnerGlow: {
        position: 'absolute',
        width: 110,
        height: 110,
        borderRadius: 55,
        backgroundColor: GLASS.accentGlow,
    },
    spinnerRing: {
        position: 'absolute',
        width: 100,
        height: 100,
        borderRadius: 50,
        overflow: 'hidden',
    },
    spinnerRingGradient: {
        flex: 1,
        borderRadius: 50,
        borderWidth: 2.5,
        borderColor: 'transparent',
    },
    spinnerInner: {
        width: 72,
        height: 72,
        borderRadius: 36,
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 0.5,
        borderColor: GLASS.border,
        backgroundColor: GLASS.bg,
    },

    // Floating Ask Stylist button — Liquid Glass
    stylistFAB: {
        position: 'absolute',
        bottom: 100,
        left: '50%',
        transform: [{ translateX: -58 }],
        borderRadius: 28,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.12,
        shadowRadius: 12,
        elevation: 8,
    },
    stylistFABGlass: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 7,
        backgroundColor: 'rgba(255,255,255,0.72)',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 28,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.5)',
    },
    stylistFABText: {
        color: colors.text.primary,
        fontSize: 14,
        fontWeight: '700',
        letterSpacing: 0.2,
    },
});

export default MyClosetScreen;
