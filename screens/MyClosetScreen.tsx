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
import { useNavigation, useFocusEffect, useIsFocused } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
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

// Updated Category filters to match design
const CATEGORIES = [
    { id: 'favorite', label: 'Favorite', icon: 'heart' }, // Filled heart will be handled in render
    { id: 'all', label: 'All', icon: 'grid' },
    { id: 'tops', label: 'Tops', icon: 'shirt' },
    { id: 'bottoms', label: 'Bottoms', icon: 'bookmark' }, // Placeholder icon
    { id: 'shoes', label: 'Shoes', icon: 'footsteps' },
    { id: 'accessories', label: 'Accessories', icon: 'glasses' },
];

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
    onPress
}: {
    item: ClothingItem;
    onPress: () => void;
}) => {
    const imageUrl = item.imageUrl || item.image;

    return (
        <TouchableOpacity
            style={styles.gridItemContainer}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.9}
            accessibilityLabel={`${item.category || item.type || 'Clothing'} item${item.color ? `, ${item.color}` : ''}`}
            accessibilityRole="button"
        >
            <Animated.View style={styles.gridItem}>
                <View style={styles.imageContainer}>
                    {imageUrl ? (
                        <Image
                            source={{ uri: imageUrl }}
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
    const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
    const isFocused = useIsFocused();
    const [items, setItems] = useState<ClothingItem[]>([]);
    
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
                            'Camera Access Disabled',
                            'Enable camera access in Settings to continue.',
                            [
                                { text: 'Cancel', style: 'cancel' },
                                {
                                    text: 'Open Settings',
                                    onPress: () => Platform.OS === 'ios' ? Linking.openURL('app-settings:') : Linking.openSettings(),
                                },
                            ]
                        );
                        return;
                    }
                    const { status } = await ImagePicker.requestCameraPermissionsAsync();
                    if (status !== 'granted') {
                        Alert.alert('Permission needed', 'Camera permission is required.');
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
                Alert.alert('Simulator Detected', 'Camera is not available on the simulator. Please use a physical device or pick from your gallery instead.');
            } else {
                console.error('Image picker error:', error);
                Alert.alert('Error', 'Failed to open camera');
            }
        }
    };

    const runMagicPipeline = async (b64: string | null) => {
        if (!b64) {
            Alert.alert("Error", "Could not read image data");
            return;
        }

        setIsUploadingOverlay(true);
        setUploadStatusMsg("Analyzing with AI...");

        try {
            // Call External AI Service directly (serverless)
            const result = await ExternalAIService.processClothingImage(b64);
            
            if (!result.success) {
                throw new Error("AI processing failed");
            }

            setUploadStatusMsg("Saving to Your Wardrobe...");

            // Map category
            let categoryStr = "tops";
            const cat = (result.classification?.category || '').toLowerCase();
            if (cat.includes("bottom") || cat.includes("pant") || cat.includes("skirt")) categoryStr = "bottoms";
            else if (cat.includes("shoe")) categoryStr = "shoes";
            else if (cat.includes("accessory") || cat.includes("hat") || cat.includes("bag")) categoryStr = "accessories";
            else if (cat.includes("dress")) categoryStr = "tops";
            else if (cat.includes("outerwear")) categoryStr = "tops";

            const itemToSave = {
                user_id: user?.id,
                type: result.classification?.category || 'clothing',
                category: categoryStr,
                sub_category: (result.classification?.category || 'clothing').toLowerCase(),
                color: ['various'], // Could be enhanced with color detection
                primary_color: 'various',
                style: result.classification?.attributes?.style || "Casual",
                description: result.description || `${result.classification?.category || 'clothing'}`,
                material: null,
                image_url: result.imageUrl,
                created_at: new Date().toISOString(),
            };

            const { data: savedData, error: saveError } = await supabase.from('clothing_items').insert([itemToSave]).select().single();
            if (saveError) throw saveError;

            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            loadItems();
            setIsUploadingOverlay(false);

            // Navigate to detail screen
            if (savedData) {
                const mappedItem = {
                    _id: savedData.id,
                    id: savedData.id,
                    type: savedData.type,
                    itemType: savedData.type,
                    color: savedData.color && savedData.color.length > 0 ? savedData.color[0] : 'various',
                    imageUrl: savedData.image_url,
                    image: savedData.image_url,
                    category: savedData.category,
                    wearCount: savedData.wear_count,
                    createdAt: savedData.created_at,
                    isFavorite: false,
                };

                navigation.navigate('ClothingDetail', {
                    itemId: mappedItem.id,
                    fullItem: mappedItem
                });
            }

        } catch (error: any) {
            console.error(error);
            Alert.alert("Upload Error", error.message);
        } finally {
            setIsUploadingOverlay(false);
        }
    };

    const openLegacyCamera = async () => {
        try {
            const { status } = await ImagePicker.requestCameraPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert('Permission needed', 'Camera permission is required.');
                return;
            }
            
            const result = await ImagePicker.launchCameraAsync({
                mediaTypes: ['images'],
                allowsEditing: false,
                quality: 0.8,
            });

            if (!result.canceled && result.assets[0]) {
                navigation.navigate('ClothingDetailEditor', { 
                    imageUri: result.assets[0].uri 
                });
            }
        } catch (error: any) {
            const msg: string = error?.message ?? '';
            if (msg.toLowerCase().includes('simulator') || msg.toLowerCase().includes('not available')) {
                Alert.alert("Simulator Detected", "Camera is not available on the simulator. Please use a physical device or pick from your gallery instead.");
            } else {
                console.error('Legacy camera error:', error);
                Alert.alert("Error", "Failed to open camera");
            }
        }
    };

    const handleUploadChoice = () => {
        Alert.alert(
            "Add to Closet",
            "Upload a photo to be carefully scanned and magically enhanced by AI, or just use the standard camera mode?",
            [
                { text: "AI Studio Photo", onPress: () => pickImage(false) },
                { text: "AI Studio Camera", onPress: () => pickImage(true) },
                { text: "Legacy Camera", onPress: openLegacyCamera },
                { text: "Cancel", style: "cancel" }
            ]
        );
    };

    const confirmDelete = (item: ClothingItem) => {
        Alert.alert(
            "Delete Item",
            "Are you sure you want to delete this clothing piece? This action cannot be undone.",
            [
                { text: "Cancel", style: "cancel" },
                {
                    text: "Delete",
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
            Alert.alert("Error", "Failed to delete item. It has been restored to your closet.");
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
                console.log('[Closet] Loaded', mappedItems.length, 'items. Image URLs:',
                    mappedItems.map(i => ({ type: i.type, hasImage: !!i.imageUrl, imgLen: i.imageUrl?.length || 0 })));
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
                        switch (selectedCategory) {
                            case 'tops':
                                if (cat === 'tops' || cat === 'top') return true;
                                return ['top', 'shirt', 'blouse', 'coat', 'dress', 'pullover', 'jacket', 'hoodie', 'sweater', 't-shirt', 'polo', 'cardigan'].some(k => cat.includes(k) || type.includes(k));
                            case 'bottoms':
                                if (cat === 'bottoms' || cat === 'bottom') return true;
                                return ['bottom', 'pant', 'skirt', 'jean', 'trouser', 'short', 'legging'].some(k => cat.includes(k) || type.includes(k));
                            case 'shoes':
                                if (cat === 'shoes' || cat === 'shoe') return true;
                                return ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer', 'slipper', 'feet'].some(k => cat.includes(k) || type.includes(k));
                            case 'accessories':
                                if (cat === 'accessories' || cat === 'accessory') return true;
                                return ['accessor', 'bag', 'hat', 'scarf', 'belt', 'sunglasses', 'watch', 'jewelry'].some(k => cat.includes(k) || type.includes(k));
                            default:
                                return cat.includes(selectedCategory.replace('s', '')) || type.includes(selectedCategory.replace('s', ''));
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
            <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
                {/* Header - raised 15px */}
                <View style={{ marginTop: -6 }}>
                    <View style={[styles.header, items.length === 0 && { justifyContent: 'center' }]}>
                    {isSearching ? (
                        <View style={{ flexDirection: 'row', alignItems: 'center', flex: 1, backgroundColor: '#E5E5EA', borderRadius: 20, paddingHorizontal: 12, paddingVertical: 6, marginHorizontal: 10 }}>
                            <Ionicons name="search" size={20} color={colors.text.secondary} />
                            <TextInput
                                style={{ flex: 1, marginLeft: 8, fontSize: 15, color: colors.text.primary }}
                                placeholder="Search closet..."
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
                        <>
                            <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                                <Text style={styles.headerTitle} accessibilityRole="header">My Closet</Text>
                            </View>
                            {items.length > 0 ? (
                                <>
                                    <TouchableOpacity
                                        style={styles.headerButtonLeft}
                                        onPress={() => setIsSearching(true)}
                                        accessibilityLabel="Search closet"
                                        accessibilityRole="button"
                                    >
                                        <Ionicons name="search" size={20} color={colors.text.secondary} />
                                        <Text style={styles.headerButtonText}>Search</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity
                                        style={styles.headerButtonRight}
                                        onPress={handleUploadChoice}
                                        accessibilityLabel="Upload clothing item"
                                        accessibilityRole="button"
                                    >
                                        <Ionicons name="add" size={22} color={colors.text.secondary} />
                                        <Text style={styles.headerButtonText}>Upload</Text>
                                    </TouchableOpacity>
                                </>
                            ) : <View style={{ width: 40 }} />}
                        </>
                    )}
                </View>
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
                            accessibilityLabel="Clothes"
                            accessibilityRole="tab"
                            accessibilityState={{ selected: viewMode === 'clothes' }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'clothes' && styles.segmentTextActive]}>Clothes</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.segmentButton, viewMode === 'collections' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setViewMode('collections');
                            }}
                            accessibilityLabel="Collections"
                            accessibilityRole="tab"
                            accessibilityState={{ selected: viewMode === 'collections' }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'collections' && styles.segmentTextActive]}>Collections</Text>
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
                        <Text style={styles.emptyTitle}>No collections yet</Text>
                        <Text style={styles.emptySubtitle}>
                            Group your outfits into collections — for work, weekends, seasons, or any occasion.
                        </Text>
                        <TouchableOpacity
                            style={styles.emptyButton}
                            onPress={() => (navigation as any).navigate('AIOutfit')}
                            accessibilityLabel="Create first collection"
                            accessibilityRole="button"
                        >
                            <Text style={styles.emptyButtonText}>Create First Look</Text>
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
                        <Text style={styles.emptyTitle}>Your closet is empty</Text>
                        <Text style={styles.emptySubtitle}>Start adding items to build your digital wardrobe.</Text>

                        <TouchableOpacity
                            style={styles.emptyButton}
                            onPress={handleUploadChoice}
                            accessibilityLabel="Scan wardrobe"
                            accessibilityRole="button"
                        >
                            <Text style={styles.emptyButtonText}>Scan Wardrobe</Text>
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
                            />
                        )}
                        ListFooterComponent={<View style={{ height: 100 }} />}
                    />
                )}

            </SafeAreaView>

            {/* Floating Ask Stylist button — Liquid Glass */}
            <TouchableOpacity
                style={styles.stylistFAB}
                onPress={() => (navigation as any).navigate('StylistChat')}
                activeOpacity={0.88}
                accessibilityLabel="Ask AI Stylist"
                accessibilityRole="button"
            >
                <View style={styles.stylistFABGlass}>
                    <Ionicons name="chatbubble-ellipses" size={20} color={colors.text.primary} />
                    <Text style={styles.stylistFABText}>Ask Stylist</Text>
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
        backgroundColor: '#FFFFFF', // iOS System Gray 6 (light mode typical background)
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingTop: 0,
        paddingBottom: 4,
        backgroundColor: '#FFFFFF',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: 0.3,
    },
    headerButtonLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFFFFF',
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderRadius: 20, // Pill shape
        // Subtle shadow
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,
    },
    headerButtonRight: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFFFFF',
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderRadius: 20, // Pill shape
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,

    },
    headerButtonText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#3C3C43', // iOS gray
        marginLeft: 6,
    },

    // Segmented Control
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 8,
        paddingBottom: 12,
        backgroundColor: '#FFFFFF',
    },
    segmentBackground: {
        flexDirection: 'row',
        backgroundColor: '#E5E5EA', // iOS System Gray 5
        borderRadius: 24, // Rounded
        padding: 4,
        width: 300,
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
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
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
        marginBottom: 16,
    },
    filterContentRaw: {
        paddingHorizontal: 16,
        paddingRight: 8,
    },
    filterChip: {
        paddingVertical: 8,
        paddingHorizontal: 20,
        borderRadius: 20,
        marginRight: 8,
        backgroundColor: '#E5E5EA', // Default gray
    },
    filterChipSelected: {
        backgroundColor: '#303030', // Dark grey/black active state
    },
    filterChipUnselected: {
        backgroundColor: '#E5E5EA',
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
        backgroundColor: '#FFFFFF',
        borderRadius: 16,
        overflow: 'hidden',
        // Minimal shadow for clean look
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,
    },
    imageContainer: {
        aspectRatio: 3 / 4, // Portrait ratio for clothes
        backgroundColor: '#FFFFFF',
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
    },
    collectionsIconWrap: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: '#F5F5F5',
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
        backgroundColor: '#0A1931',
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
