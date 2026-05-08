import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Dimensions,
    Platform,
    KeyboardAvoidingView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { CachedImage } from '../components/ui/CachedImage';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withSequence,
    withTiming,
    withSpring,
    Easing,
} from 'react-native-reanimated';
import AppColors from '../constants/AppColors';
import Config from '../src/config/env';
import { useTranslation } from 'react-i18next';

const { width } = Dimensions.get('window');
const ITEM_SIZE = (width - 60) / 2; // 2 column grid with padding

const ALICEVISION_API = Config.api.alicevisionUrl;

// Outfit item type
interface OutfitItem {
    id: string;
    name: string;
    category: string;
    imageUrl: string;
    color?: string;
}

// Pulsing animation for loading
const PulsingDot = ({ delay = 0 }: { delay?: number }) => {
    const opacity = useSharedValue(0.3);

    useEffect(() => {
        const timer = setTimeout(() => {
            opacity.value = withRepeat(
                withSequence(
                    withTiming(1, { duration: 500 }),
                    withTiming(0.3, { duration: 500 })
                ),
                -1,
                false
            );
        }, delay);
        return () => clearTimeout(timer);
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: opacity.value,
    }));

    return (
        <Animated.View
            style={[
                {
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: AppColors.primary,
                    marginHorizontal: 3,
                },
                animatedStyle,
            ]}
        />
    );
};

// Floating sparkle effect
const FloatingSparkle = () => {
    const translateY = useSharedValue(0);
    const scale = useSharedValue(1);

    useEffect(() => {
        translateY.value = withRepeat(
            withSequence(
                withTiming(-8, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
                withTiming(0, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) })
            ),
            -1,
            true
        );
        scale.value = withRepeat(
            withSequence(
                withTiming(1.1, { duration: 1500 }),
                withTiming(1, { duration: 1500 })
            ),
            -1,
            true
        );
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [
            { translateY: translateY.value },
            { scale: scale.value },
        ],
    }));

    return (
        <Animated.View style={[{ alignItems: 'center', marginBottom: 24 }, animatedStyle]}>
            <View style={{
                width: 80,
                height: 80,
                borderRadius: 40,
                backgroundColor: AppColors.surface,
                alignItems: 'center',
                justifyContent: 'center',
                borderWidth: 1,
                borderColor: AppColors.border,
            }}>
                <Ionicons name="sparkles" size={36} color={AppColors.primary} />
            </View>
        </Animated.View>
    );
};

// Individual outfit item card with photo
const OutfitItemCard = ({ item, index }: { item: OutfitItem; index: number }) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const handlePressIn = () => {
        scale.value = withSpring(0.96, { damping: 15 });
    };

    const handlePressOut = () => {
        scale.value = withSpring(1, { damping: 15 });
    };

    return (
        <Animated.View
            entering={FadeInUp.delay(100 + index * 80).springify()}
            style={{ width: ITEM_SIZE, marginBottom: 16 }}
        >
            <TouchableOpacity
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                activeOpacity={1}
            >
                <Animated.View style={[{
                    backgroundColor: AppColors.surface,
                    borderRadius: 20,
                    overflow: 'hidden',
                    borderWidth: 1,
                    borderColor: AppColors.border,
                }, animatedStyle]}>
                    {/* Item Image */}
                    <CachedImage
                        uri={item.imageUrl}
                        style={{
                            width: '100%',
                            height: ITEM_SIZE,
                            backgroundColor: '#F8F8F8',
                        }}
                        contentFit="cover"
                        fadeIn={false}
                    />

                    {/* Item Info */}
                    <View style={{ padding: 12 }}>
                        <Text style={{
                            fontSize: 11,
                            color: AppColors.textMuted,
                            textTransform: 'uppercase',
                            letterSpacing: 0.5,
                            marginBottom: 4,
                        }}>
                            {item.category}
                        </Text>
                        <Text style={{
                            fontSize: 14,
                            fontWeight: '600',
                            color: AppColors.text,
                        }} numberOfLines={2}>
                            {item.name}
                        </Text>
                    </View>
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

const MeetingOutfitScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();

    // Simple state
    const [step, setStep] = useState<'input' | 'loading' | 'result'>('input');
    const [description, setDescription] = useState('');
    const [generatedOutfit, setGeneratedOutfit] = useState<{
        title: string;
        style: string;
        items: OutfitItem[];
    } | null>(null);

    const buttonScale = useSharedValue(1);

    const buttonAnimStyle = useAnimatedStyle(() => ({
        transform: [{ scale: buttonScale.value }],
    }));

    const handlePressIn = () => {
        buttonScale.value = withSpring(0.96, { damping: 15 });
    };

    const handlePressOut = () => {
        buttonScale.value = withSpring(1, { damping: 15 });
    };

    const handleCreateStyle = async () => {
        if (!description.trim()) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setStep('loading');

        try {
            // Call AI to generate outfit
            const response = await fetch(`${ALICEVISION_API}/generate-outfit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    description: description.trim(),
                    inferFromDescription: true,
                }),
            });

            if (!response.ok) throw new Error('Failed');

            const data = await response.json();
            setGeneratedOutfit(data);
            setStep('result');

        } catch (err) {
            // Smart fallback with individual item photos
            const outfit = generateSmartFallback(description);
            setGeneratedOutfit(outfit);
            setStep('result');
        }
    };

    const generateSmartFallback = (text: string) => {
        const isFormal = /interview|presentation|meeting|client|ceo|board|formal|important/i.test(text);
        const isCasual = /casual|coffee|friend|lunch|team|relaxed/i.test(text);

        if (isFormal) {
            return {
                title: 'Professional Power Look',
                style: 'Business Formal',
                items: [
                    {
                        id: '1',
                        name: 'Navy Wool Blazer',
                        category: 'Jacket',
                        imageUrl: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400',
                        color: 'Navy',
                    },
                    {
                        id: '2',
                        name: 'White Dress Shirt',
                        category: 'Top',
                        imageUrl: 'https://images.unsplash.com/photo-1603252109303-2751441dd157?w=400',
                        color: 'White',
                    },
                    {
                        id: '3',
                        name: 'Charcoal Wool Trousers',
                        category: 'Pants',
                        imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400',
                        color: 'Charcoal',
                    },
                    {
                        id: '4',
                        name: 'Brown Leather Oxfords',
                        category: 'Shoes',
                        imageUrl: 'https://images.unsplash.com/photo-1614252369475-531eba835eb1?w=400',
                        color: 'Brown',
                    },
                    {
                        id: '5',
                        name: 'Leather Belt',
                        category: 'Accessory',
                        imageUrl: 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400',
                        color: 'Brown',
                    },
                    {
                        id: '6',
                        name: 'Minimalist Watch',
                        category: 'Accessory',
                        imageUrl: 'https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400',
                        color: 'Silver',
                    },
                ],
            };
        } else if (isCasual) {
            return {
                title: 'Smart Casual Look',
                style: 'Casual',
                items: [
                    {
                        id: '1',
                        name: 'Merino Wool Sweater',
                        category: 'Top',
                        imageUrl: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400',
                        color: 'Navy',
                    },
                    {
                        id: '2',
                        name: 'Oxford Button-Down',
                        category: 'Shirt',
                        imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400',
                        color: 'Light Blue',
                    },
                    {
                        id: '3',
                        name: 'Dark Slim Jeans',
                        category: 'Pants',
                        imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400',
                        color: 'Indigo',
                    },
                    {
                        id: '4',
                        name: 'White Leather Sneakers',
                        category: 'Shoes',
                        imageUrl: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400',
                        color: 'White',
                    },
                ],
            };
        }

        return {
            title: 'Versatile Modern Look',
            style: 'Business Casual',
            items: [
                {
                    id: '1',
                    name: 'Cotton Sport Coat',
                    category: 'Jacket',
                    imageUrl: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400',
                    color: 'Khaki',
                },
                {
                    id: '2',
                    name: 'Light Blue Button-Down',
                    category: 'Shirt',
                    imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400',
                    color: 'Light Blue',
                },
                {
                    id: '3',
                    name: 'Khaki Chinos',
                    category: 'Pants',
                    imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400',
                    color: 'Khaki',
                },
                {
                    id: '4',
                    name: 'Brown Leather Loafers',
                    category: 'Shoes',
                    imageUrl: 'https://images.unsplash.com/photo-1582897085656-c636d006a246?w=400',
                    color: 'Brown',
                },
                {
                    id: '5',
                    name: 'Canvas Belt',
                    category: 'Accessory',
                    imageUrl: 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400',
                    color: 'Brown',
                },
            ],
        };
    };

    const handleStartOver = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setStep('input');
        setDescription('');
        setGeneratedOutfit(null);
    };

    // Input Step - Clean, minimal design
    const renderInputStep = () => (
        <View style={{ flex: 1, paddingHorizontal: 24 }}>
            {/* Hero section */}
            <Animated.View
                entering={FadeIn.duration(600)}
                style={{ alignItems: 'center', paddingTop: 40, paddingBottom: 32 }}
            >
                <FloatingSparkle />

                <Text style={{
                    fontSize: 28,
                    fontWeight: '700',
                    color: AppColors.text,
                    textAlign: 'center',
                    marginBottom: 12,
                    letterSpacing: -0.5,
                }}>
                    {t('meetingOutfit.createYourStyle')}
                </Text>

                <Text style={{
                    fontSize: 16,
                    color: AppColors.textSecondary,
                    textAlign: 'center',
                    lineHeight: 24,
                    paddingHorizontal: 20,
                }}>
                    {t('meetingOutfit.describeEvent')}
                </Text>
            </Animated.View>

            {/* Input area */}
            <Animated.View entering={FadeInUp.delay(200).springify()}>
                <TextInput
                    style={{
                        backgroundColor: AppColors.surface,
                        borderRadius: 20,
                        padding: 20,
                        fontSize: 17,
                        color: AppColors.text,
                        minHeight: 140,
                        textAlignVertical: 'top',
                        borderWidth: 1,
                        borderColor: AppColors.border,
                        lineHeight: 26,
                    }}
                    placeholder={t('meetingOutfit.placeholder')}
                    placeholderTextColor={AppColors.textMuted}
                    value={description}
                    onChangeText={setDescription}
                    multiline
                    numberOfLines={5}
                    maxLength={500}
                />
            </Animated.View>

            {/* Create Style Button */}
            <Animated.View
                entering={FadeInUp.delay(300).springify()}
                style={{ marginTop: 24 }}
            >
                <TouchableOpacity
                    onPressIn={handlePressIn}
                    onPressOut={handlePressOut}
                    onPress={handleCreateStyle}
                    disabled={!description.trim()}
                    activeOpacity={1}
                >
                    <Animated.View style={[
                        {
                            backgroundColor: description.trim() ? AppColors.primary : AppColors.border,
                            paddingVertical: 18,
                            borderRadius: 16,
                            alignItems: 'center',
                            flexDirection: 'row',
                            justifyContent: 'center',
                        },
                        buttonAnimStyle,
                    ]}>
                        <Ionicons
                            name="sparkles"
                            size={20}
                            color={AppColors.background}
                            style={{ marginRight: 10 }}
                        />
                        <Text style={{
                            fontSize: 17,
                            fontWeight: '600',
                            color: AppColors.background,
                        }}>
                            {t('meetingOutfit.generateOutfit')}
                        </Text>
                    </Animated.View>
                </TouchableOpacity>
            </Animated.View>

            {/* Hint */}
            <Animated.View
                entering={FadeInUp.delay(400)}
                style={{ marginTop: 20, alignItems: 'center' }}
            >
                <Text style={{
                    fontSize: 13,
                    color: AppColors.textMuted,
                    textAlign: 'center',
                }}>
                    ✨ AI creates a complete outfit: sweater, pants, shoes & more
                </Text>
            </Animated.View>
        </View>
    );

    // Loading Step
    const renderLoadingStep = () => (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 40 }}>
            <Animated.View entering={FadeIn.duration(400)}>
                <View style={{
                    width: 100,
                    height: 100,
                    borderRadius: 50,
                    backgroundColor: AppColors.surface,
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: 32,
                    borderWidth: 1,
                    borderColor: AppColors.border,
                }}>
                    <ActivityIndicator size="large" color={AppColors.primary} />
                </View>
            </Animated.View>

            <Animated.Text
                entering={FadeInUp.delay(100)}
                style={{
                    fontSize: 24,
                    fontWeight: '700',
                    color: AppColors.text,
                    textAlign: 'center',
                    marginBottom: 12,
                }}
            >
                Crafting your outfit...
            </Animated.Text>

            <Animated.Text
                entering={FadeInUp.delay(200)}
                style={{
                    fontSize: 15,
                    color: AppColors.textSecondary,
                    textAlign: 'center',
                    lineHeight: 22,
                }}
            >
                Selecting sweater, pants, shoes and accessories
            </Animated.Text>

            <View style={{ flexDirection: 'row', marginTop: 28 }}>
                <PulsingDot delay={0} />
                <PulsingDot delay={150} />
                <PulsingDot delay={300} />
            </View>
        </View>
    );

    // Result Step - Grid of individual item photos
    const renderResultStep = () => {
        if (!generatedOutfit) return null;

        return (
            <ScrollView
                style={{ flex: 1 }}
                contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 40 }}
                showsVerticalScrollIndicator={false}
            >
                {/* Header */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={{ marginTop: 8, marginBottom: 20 }}
                >
                    <Text style={{
                        fontSize: 13,
                        color: AppColors.textMuted,
                        textTransform: 'uppercase',
                        letterSpacing: 1,
                        marginBottom: 6,
                    }}>
                        {generatedOutfit.style}
                    </Text>
                    <Text style={{
                        fontSize: 26,
                        fontWeight: '700',
                        color: AppColors.text,
                        letterSpacing: -0.5,
                    }}>
                        {generatedOutfit.title}
                    </Text>
                    <Text style={{
                        fontSize: 15,
                        color: AppColors.textSecondary,
                        marginTop: 8,
                    }}>
                        {generatedOutfit.items.length} items curated for your event
                    </Text>
                </Animated.View>

                {/* Items Grid */}
                <View style={{
                    flexDirection: 'row',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                }}>
                    {generatedOutfit.items.map((item, index) => (
                        <OutfitItemCard key={item.id} item={item} index={index} />
                    ))}
                </View>

                {/* Action Button */}
                <Animated.View
                    entering={FadeInUp.delay(500).springify()}
                    style={{ marginTop: 8 }}
                >
                    <TouchableOpacity
                        onPress={handleStartOver}
                        style={{
                            backgroundColor: AppColors.primary,
                            paddingVertical: 18,
                            borderRadius: 16,
                            alignItems: 'center',
                            flexDirection: 'row',
                            justifyContent: 'center',
                        }}
                    >
                        <Ionicons
                            name="add"
                            size={22}
                            color={AppColors.background}
                            style={{ marginRight: 8 }}
                        />
                        <Text style={{
                            fontSize: 17,
                            fontWeight: '600',
                            color: AppColors.background,
                        }}>
                            Create Another Style
                        </Text>
                    </TouchableOpacity>
                </Animated.View>
            </ScrollView>
        );
    };

    return (
        <View style={{ flex: 1, backgroundColor: AppColors.background }}>
            <SafeAreaView style={{ flex: 1 }}>
                {/* Header */}
                <View style={{
                    flexDirection: 'row',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    paddingHorizontal: 20,
                    paddingVertical: 16,
                }}>
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="close" size={28} color={AppColors.text} />
                    </TouchableOpacity>

                    {step === 'result' && (
                        <TouchableOpacity
                            onPress={() => { /* Share functionality */ }}
                            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        >
                            <Ionicons name="share-outline" size={24} color={AppColors.text} />
                        </TouchableOpacity>
                    )}

                    {step !== 'result' && <View style={{ width: 28 }} />}
                </View>

                {/* Content */}
                <KeyboardAvoidingView
                    style={{ flex: 1 }}
                    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                >
                    {step === 'input' && renderInputStep()}
                    {step === 'loading' && renderLoadingStep()}
                    {step === 'result' && renderResultStep()}
                </KeyboardAvoidingView>
            </SafeAreaView>
        </View>
    );
};

export default MeetingOutfitScreen;
