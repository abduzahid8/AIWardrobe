import React, { useState, useRef, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
    ActivityIndicator,
    Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import * as Location from 'expo-location';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withSequence,
    withTiming,
    withSpring,
    FadeIn,
    FadeInUp,
    FadeInDown,
    SlideInRight,
    Easing,
} from 'react-native-reanimated';
import { TahoeIconButton } from '../components/TahoeButton';
import AppColors from '../constants/AppColors';
import { useWardrobeItems } from '../src/hooks';

const { width, height } = Dimensions.get('window');

// API URLs from environment
const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://aiwardrobe-ivh4.onrender.com';
const ALICEVISION_API = process.env.EXPO_PUBLIC_ALICEVISION_API || 'http://localhost:5050';

// Use unified AppColors
const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    surfaceLight: AppColors.surfaceSecondary,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    success: '#34C759',
    aiGlow: 'rgba(100, 100, 255, 0.1)',
};

// Quick Occasion Suggestions - Alta-style
const OCCASION_SUGGESTIONS = [
    { id: '1', text: 'Date Night', icon: 'heart-outline', emoji: '💕', color: '#E91E63' },
    { id: '2', text: 'Job Interview', icon: 'briefcase-outline', emoji: '💼', color: '#3F51B5' },
    { id: '3', text: 'Trip/Travel', icon: 'airplane-outline', emoji: '✈️', color: '#00BCD4' },
    { id: '4', text: 'Brunch', icon: 'cafe-outline', emoji: '🥂', color: '#FF9800' },
    { id: '5', text: 'Office Day', icon: 'business-outline', emoji: '👔', color: '#607D8B' },
    { id: '6', text: 'Party', icon: 'sparkles-outline', emoji: '🎉', color: '#9C27B0' },
    { id: '7', text: 'Casual Outing', icon: 'walk-outline', emoji: '👟', color: '#4CAF50' },
    { id: '8', text: 'Wedding Guest', icon: 'flower-outline', emoji: '💐', color: '#F06292' },
];

// Outfit Adjustment Options - for refining suggestions
const ADJUSTMENT_OPTIONS = [
    { id: 'casual', label: 'More casual', icon: 'sunny-outline' },
    { id: 'formal', label: 'More formal', icon: 'business-outline' },
    { id: 'colors', label: 'Different colors', icon: 'color-palette-outline' },
    { id: 'layers', label: 'Add layers', icon: 'layers-outline' },
    { id: 'weather', label: 'Weather-appropriate', icon: 'partly-sunny-outline' },
];

interface WeatherContext {
    temp: number;
    condition: string;
    location?: string;
}

// Outfit item from AI response
interface OutfitItemType {
    category?: string;
    specificType?: string;
    primaryColor?: string;
    colorHex?: string;
}

interface OutfitSuggestion {
    items?: OutfitItemType[];
    confidence?: number;
    reasoning?: string;
}

// Occasion suggestion type
interface OccasionSuggestion {
    id: string;
    text: string;
    icon: string;
    emoji: string;
    color: string;
}

// Helper to fetch current weather
async function fetchWeather(): Promise<WeatherContext> {
    try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
            return { temp: 20, condition: 'clear' };
        }

        const location = await Location.getCurrentPositionAsync({});
        const response = await fetch(
            `${API_URL}/weather/coords`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lat: location.coords.latitude,
                    lon: location.coords.longitude
                })
            }
        );

        if (response.ok) {
            const data = await response.json();
            return {
                temp: Math.round(data.temp || data.temperature || 20),
                condition: data.condition || data.description || 'clear',
                location: data.city || data.location
            };
        }
    } catch (error) {
        console.log('Weather fetch failed, using defaults:', error);
    }
    return { temp: 20, condition: 'clear' };
}

// Typing Indicator Animation
const TypingIndicator = () => {
    const dot1Opacity = useSharedValue(0.3);
    const dot2Opacity = useSharedValue(0.3);
    const dot3Opacity = useSharedValue(0.3);

    useEffect(() => {
        dot1Opacity.value = withRepeat(
            withSequence(
                withTiming(1, { duration: 400 }),
                withTiming(0.3, { duration: 400 })
            ),
            -1
        );
        setTimeout(() => {
            dot2Opacity.value = withRepeat(
                withSequence(
                    withTiming(1, { duration: 400 }),
                    withTiming(0.3, { duration: 400 })
                ),
                -1
            );
        }, 150);
        setTimeout(() => {
            dot3Opacity.value = withRepeat(
                withSequence(
                    withTiming(1, { duration: 400 }),
                    withTiming(0.3, { duration: 400 })
                ),
                -1
            );
        }, 300);
    }, []);

    const dot1Style = useAnimatedStyle(() => ({ opacity: dot1Opacity.value }));
    const dot2Style = useAnimatedStyle(() => ({ opacity: dot2Opacity.value }));
    const dot3Style = useAnimatedStyle(() => ({ opacity: dot3Opacity.value }));

    return (
        <View style={styles.typingContainer}>
            <View style={styles.aiBubble}>
                <View style={styles.typingDots}>
                    <Animated.View style={[styles.typingDot, dot1Style]} />
                    <Animated.View style={[styles.typingDot, dot2Style]} />
                    <Animated.View style={[styles.typingDot, dot3Style]} />
                </View>
            </View>
        </View>
    );
};

// Chat Message Bubble with Adjustment Buttons
const ChatBubble = ({ message, isAI, outfit, onAdjust }: { message: string; isAI: boolean; outfit?: OutfitSuggestion; onAdjust?: (adjustment: string) => void }) => {
    return (
        <Animated.View
            entering={FadeInUp.springify()}
            style={[styles.bubbleContainer, isAI ? styles.aiBubbleContainer : styles.userBubbleContainer]}
        >
            {isAI && (
                <View style={styles.aiAvatarSmall}>
                    <Ionicons name="sparkles" size={16} color={COLORS.primary} />
                </View>
            )}
            <View style={[styles.bubble, isAI ? styles.aiBubble : styles.userBubble]}>
                <Text style={[styles.bubbleText, isAI ? styles.aiBubbleText : styles.userBubbleText]}>
                    {message}
                </Text>

                {/* Outfit recommendations */}
                {outfit && outfit.items && outfit.items.length > 0 && (
                    <View style={styles.outfitPreview}>
                        <View style={styles.outfitItems}>
                            {outfit.items.slice(0, 4).map((item: OutfitItemType, idx: number) => (
                                <View key={idx} style={styles.outfitItemCard}>
                                    <View style={[styles.outfitItemColor, { backgroundColor: item.colorHex || COLORS.surface }]} />
                                    <Text style={styles.outfitItemType} numberOfLines={1}>
                                        {item.specificType || item.category}
                                    </Text>
                                    <Text style={styles.outfitItemColorName} numberOfLines={1}>
                                        {item.primaryColor}
                                    </Text>
                                </View>
                            ))}
                        </View>
                        <View style={styles.outfitMeta}>
                            <View style={styles.confidenceBadge}>
                                <Ionicons name="checkmark-circle" size={14} color={COLORS.success} />
                                <Text style={styles.confidenceText}>
                                    {Math.round((outfit.confidence || 0.85) * 100)}% match
                                </Text>
                            </View>
                            <Text style={styles.reasoningText} numberOfLines={2}>
                                {outfit.reasoning}
                            </Text>
                        </View>

                        {/* Adjustment buttons - "Adjust until right" */}
                        {onAdjust && (
                            <View style={styles.adjustmentSection}>
                                <Text style={styles.adjustmentLabel}>Not quite right? Adjust:</Text>
                                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.adjustmentScroll}>
                                    {ADJUSTMENT_OPTIONS.map((option) => (
                                        <TouchableOpacity
                                            key={option.id}
                                            style={styles.adjustmentChip}
                                            onPress={() => {
                                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                                onAdjust(option.label);
                                            }}
                                        >
                                            <Ionicons name={option.icon as any} size={14} color={COLORS.primary} />
                                            <Text style={styles.adjustmentChipText}>{option.label}</Text>
                                        </TouchableOpacity>
                                    ))}
                                </ScrollView>
                            </View>
                        )}
                    </View>
                )}
            </View>
        </Animated.View>
    );
};

// Occasion Card - Enhanced Alta-style
const OccasionCard = ({ occasion, onPress }: { occasion: OccasionSuggestion; onPress: () => void }) => {
    return (
        <TouchableOpacity
            style={[styles.occasionCard, { borderColor: occasion.color + '40' }]}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                onPress();
            }}
            activeOpacity={0.7}
        >
            <View style={[styles.occasionIconBg, { backgroundColor: occasion.color + '20' }]}>
                <Text style={styles.occasionEmoji}>{occasion.emoji}</Text>
            </View>
            <Text style={styles.occasionCardText}>{occasion.text}</Text>
            <Ionicons name="arrow-forward" size={14} color={COLORS.textMuted} />
        </TouchableOpacity>
    );
};

interface ChatMessage {
    id: string;
    text: string;
    isAI: boolean;
    outfit?: OutfitSuggestion;
    suggestedOutfits?: OutfitSuggestion[];
}

const OutfitAIScreen = () => {
    const navigation = useNavigation();
    const scrollViewRef = useRef<ScrollView>(null);
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: '1',
            text: "Hi! I'm your AI stylist. Tell me about your occasion, and I'll create the perfect outfit from your wardrobe. What are you dressing for today?",
            isAI: true,
        }
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [showSuggestions, setShowSuggestions] = useState(true);
    const [weather, setWeather] = useState<WeatherContext>({ temp: 20, condition: 'clear' });

    // Get wardrobe items from hook
    const { items: wardrobeItems, loading: wardrobeLoading } = useWardrobeItems();

    // Fetch weather on mount
    useEffect(() => {
        fetchWeather().then(setWeather);
    }, []);

    const scrollToBottom = () => {
        setTimeout(() => {
            scrollViewRef.current?.scrollToEnd({ animated: true });
        }, 100);
    };

    const sendMessage = async (text: string) => {
        if (!text.trim()) return;

        setShowSuggestions(false);
        setIsLoading(true);

        // Add user message
        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            text: text.trim(),
            isAI: false,
        };
        setMessages(prev => [...prev, userMessage]);
        setMessage('');
        scrollToBottom();

        try {
            // Format wardrobe items for AI
            const formattedWardrobe = wardrobeItems.map(item => ({
                id: item.id,
                type: item.type || item.itemType,
                color: item.color,
                style: item.style,
                description: item.description,
                hasImage: !!(item.image || item.imageUrl)
            }));

            console.log(`🤖 Sending to AI Stylist:`);
            console.log(`   - Message: ${text.trim()}`);
            console.log(`   - Wardrobe items: ${formattedWardrobe.length}`);
            console.log(`   - Weather: ${weather.temp}°C, ${weather.condition}`);

            // Try AliceVision /stylist/chat endpoint first
            let response = await fetch(`${ALICEVISION_API}/stylist/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text.trim(),
                    session_id: null, // New session
                    images: []
                }),
            }).catch(() => null);

            // Fallback to /outfit/chat if stylist endpoint fails
            if (!response?.ok) {
                response = await fetch(`${ALICEVISION_API}/outfit/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text.trim(),
                        wardrobe_items: formattedWardrobe,
                        context: {
                            weather: weather,
                            occasion: detectOccasion(text),
                        }
                    }),
                });
            }

            if (response?.ok) {
                const data = await response.json();
                console.log('✅ AI Response received');

                // Add AI response
                const aiMessage: ChatMessage = {
                    id: (Date.now() + 1).toString(),
                    text: data.message || data.response || "I'd recommend a smart casual look for that occasion!",
                    isAI: true,
                    suggestedOutfits: data.suggestedOutfits,
                    outfit: data.suggestedOutfits?.[0],
                };
                setMessages(prev => [...prev, aiMessage]);
            } else {
                // Try Node.js backend fallback
                const fallbackResponse = await fetch(`${API_URL}/ai-chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: text.trim() }),
                }).catch(() => null);

                if (fallbackResponse?.ok) {
                    const data = await fallbackResponse.json();
                    const aiMessage: ChatMessage = {
                        id: (Date.now() + 1).toString(),
                        text: data.text || "Let me suggest some outfit ideas for you!",
                        isAI: true,
                    };
                    setMessages(prev => [...prev, aiMessage]);
                } else {
                    // Final fallback
                    const aiMessage: ChatMessage = {
                        id: (Date.now() + 1).toString(),
                        text: "I'm having trouble connecting right now. Make sure the AI service is running and try again!",
                        isAI: true,
                    };
                    setMessages(prev => [...prev, aiMessage]);
                }
            }
        } catch (error) {
            console.error('Chat error:', error);
            // Graceful fallback with style tips
            const aiMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                text: generateFallbackResponse(text, wardrobeItems.length, weather),
                isAI: true,
            };
            setMessages(prev => [...prev, aiMessage]);
        }

        setIsLoading(false);
        scrollToBottom();
    };

    const handleOccasionPress = (occasion: OccasionSuggestion) => {
        sendMessage(`I need an outfit for ${occasion.text.toLowerCase()}`);
    };

    // Handle outfit adjustments - "until they are right"
    const handleAdjustOutfit = (adjustment: string) => {
        sendMessage(`Please adjust the outfit suggestion: ${adjustment}`);
    };

    // Helper to detect occasion from message
    function detectOccasion(text: string): string {
        const lower = text.toLowerCase();
        if (lower.includes('interview') || lower.includes('job')) return 'interview';
        if (lower.includes('date') || lower.includes('romantic')) return 'date';
        if (lower.includes('dinner') || lower.includes('restaurant')) return 'dinner';
        if (lower.includes('meeting') || lower.includes('business')) return 'business';
        if (lower.includes('workout') || lower.includes('gym')) return 'gym';
        if (lower.includes('party') || lower.includes('club')) return 'party';
        if (lower.includes('casual') || lower.includes('everyday')) return 'casual';
        return 'general';
    }

    // Generate fallback response when AI is unavailable
    function generateFallbackResponse(query: string, itemCount: number, weather: WeatherContext): string {
        const occasion = detectOccasion(query);
        const isWarm = weather.temp > 20;

        let tips = "I'd love to help! Here are some style tips:\n\n";

        switch (occasion) {
            case 'interview':
                tips += "• For interviews: Navy blazer + white shirt + dark trousers\n";
                tips += "• Keep accessories minimal and professional\n";
                tips += "• Polished shoes make a great impression";
                break;
            case 'date':
                tips += "• For dates: Something that makes you feel confident!\n";
                tips += "• Smart casual usually works well\n";
                tips += "• Add one statement piece to stand out";
                break;
            case 'dinner':
                tips += "• For dinner: Smart jeans + nice blouse/shirt\n";
                tips += "• Elevate with a blazer or nice jacket\n";
                tips += "• Comfortable but polished shoes";
                break;
            default:
                tips += isWarm
                    ? "• Light fabrics and breathable materials work best\n• Try neutral colors for versatility"
                    : "• Layer up with sweaters or light jackets\n• Warmer tones complement the season";
        }

        if (itemCount > 0) {
            tips += `\n\n💡 You have ${itemCount} items in your wardrobe to mix and match!`;
        } else {
            tips += "\n\n📸 Scan your wardrobe to get personalized recommendations!";
        }

        return tips;
    }

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={styles.header}
                >
                    <TahoeIconButton
                        icon="arrow-back"
                        onPress={() => navigation.goBack()}
                        color={COLORS.text}
                    />

                    <View style={styles.headerCenter}>
                        <View style={styles.headerAIBadge}>
                            <Ionicons name="sparkles" size={14} color={COLORS.primary} />
                        </View>
                        <Text style={styles.headerTitle}>AI Stylist</Text>
                    </View>

                    <TahoeIconButton
                        icon="options-outline"
                        onPress={() => { }}
                        color={COLORS.text}
                    />
                </Animated.View>

                {/* Chat Messages */}
                <ScrollView
                    ref={scrollViewRef}
                    contentContainerStyle={styles.chatContent}
                    showsVerticalScrollIndicator={false}
                    keyboardShouldPersistTaps="handled"
                >
                    {/* AI Welcome */}
                    <Animated.View
                        entering={FadeIn.delay(100).duration(400)}
                        style={styles.welcomeSection}
                    >
                        <View style={styles.aiAvatarLarge}>
                            <Ionicons name="sparkles" size={32} color={COLORS.primary} />
                        </View>
                        <Text style={styles.welcomeTitle}>Your AI Stylist</Text>
                        <Text style={styles.welcomeSubtitle}>
                            Powered by vision + fashion intelligence
                        </Text>
                    </Animated.View>

                    {/* Messages */}
                    {messages.map((msg) => (
                        <ChatBubble
                            key={msg.id}
                            message={msg.text}
                            isAI={msg.isAI}
                            outfit={msg.outfit}
                            onAdjust={msg.outfit ? handleAdjustOutfit : undefined}
                        />
                    ))}

                    {/* Typing indicator */}
                    {isLoading && <TypingIndicator />}

                    {/* Quick Occasion Suggestions */}
                    {showSuggestions && (
                        <Animated.View
                            entering={FadeInUp.delay(300).springify()}
                            style={styles.suggestionsSection}
                        >
                            <Text style={styles.suggestionsTitle}>What's the occasion?</Text>
                            <Text style={styles.suggestionsSubtitle}>Tell me your plans and I'll find the perfect outfit</Text>
                            <View style={styles.occasionGrid}>
                                {OCCASION_SUGGESTIONS.map((occasion) => (
                                    <OccasionCard
                                        key={occasion.id}
                                        occasion={occasion}
                                        onPress={() => handleOccasionPress(occasion)}
                                    />
                                ))}
                            </View>
                        </Animated.View>
                    )}

                    <View style={{ height: 120 }} />
                </ScrollView>

                {/* Input Area */}
                <KeyboardAvoidingView
                    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                    keyboardVerticalOffset={10}
                >
                    <View style={styles.inputWrapper}>
                        <View style={styles.inputContainer}>
                            <TextInput
                                style={styles.textInput}
                                placeholder="Describe your occasion..."
                                placeholderTextColor={COLORS.textMuted}
                                value={message}
                                onChangeText={setMessage}
                                returnKeyType="send"
                                onSubmitEditing={() => sendMessage(message)}
                                editable={!isLoading}
                            />
                            <TouchableOpacity
                                style={[
                                    styles.sendButton,
                                    message.trim() && styles.sendButtonActive
                                ]}
                                onPress={() => sendMessage(message)}
                                disabled={isLoading || !message.trim()}
                            >
                                {isLoading ? (
                                    <ActivityIndicator size="small" color={COLORS.textMuted} />
                                ) : (
                                    <Ionicons
                                        name="arrow-up"
                                        size={20}
                                        color={message.trim() ? COLORS.background : COLORS.textMuted}
                                    />
                                )}
                            </TouchableOpacity>
                        </View>
                    </View>
                </KeyboardAvoidingView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
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
        borderBottomWidth: 1,
        borderBottomColor: COLORS.border,
    },
    headerCenter: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    headerAIBadge: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 8,
    },
    headerTitle: {
        fontSize: 17,
        fontWeight: '600',
        color: COLORS.text,
    },

    // Chat content
    chatContent: {
        paddingHorizontal: 16,
        paddingTop: 20,
    },

    // Welcome section
    welcomeSection: {
        alignItems: 'center',
        paddingVertical: 20,
        marginBottom: 20,
    },
    aiAvatarLarge: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    welcomeTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 4,
    },
    welcomeSubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
    },

    // Chat bubbles
    bubbleContainer: {
        marginBottom: 16,
        flexDirection: 'row',
    },
    aiBubbleContainer: {
        justifyContent: 'flex-start',
    },
    userBubbleContainer: {
        justifyContent: 'flex-end',
    },
    aiAvatarSmall: {
        width: 28,
        height: 28,
        borderRadius: 14,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 8,
        marginTop: 2,
    },
    bubble: {
        maxWidth: width * 0.75,
        borderRadius: 20,
        paddingHorizontal: 16,
        paddingVertical: 12,
    },
    aiBubble: {
        backgroundColor: COLORS.surfaceLight,
        borderBottomLeftRadius: 4,
    },
    userBubble: {
        backgroundColor: COLORS.primary,
        borderBottomRightRadius: 4,
        marginLeft: 'auto',
    },
    bubbleText: {
        fontSize: 15,
        lineHeight: 22,
    },
    aiBubbleText: {
        color: COLORS.text,
    },
    userBubbleText: {
        color: COLORS.background,
    },

    // Outfit preview
    outfitPreview: {
        marginTop: 12,
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
    },
    outfitItems: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 10,
    },
    outfitItemCard: {
        flex: 1,
        backgroundColor: COLORS.background,
        borderRadius: 12,
        padding: 8,
        alignItems: 'center',
    },
    outfitItemColor: {
        width: 32,
        height: 32,
        borderRadius: 16,
        marginBottom: 6,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    outfitItemType: {
        fontSize: 11,
        fontWeight: '600',
        color: COLORS.text,
        textAlign: 'center',
    },
    outfitItemColorName: {
        fontSize: 10,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },
    outfitMeta: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: 8,
    },
    confidenceBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.background,
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
    },
    confidenceText: {
        fontSize: 11,
        fontWeight: '600',
        color: COLORS.success,
        marginLeft: 4,
    },
    reasoningText: {
        flex: 1,
        fontSize: 12,
        color: COLORS.textSecondary,
        lineHeight: 16,
    },

    // Typing indicator
    typingContainer: {
        marginBottom: 16,
    },
    typingDots: {
        flexDirection: 'row',
        gap: 4,
    },
    typingDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: COLORS.textMuted,
    },

    // Suggestions
    suggestionsSection: {
        marginTop: 10,
        marginBottom: 20,
    },
    suggestionsTitle: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.textSecondary,
        marginBottom: 12,
    },
    occasionGrid: {
        gap: 10,
    },
    occasionCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surfaceLight,
        paddingHorizontal: 16,
        paddingVertical: 14,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    occasionIconBg: {
        width: 40,
        height: 40,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    occasionEmoji: {
        fontSize: 20,
    },
    occasionCardText: {
        flex: 1,
        fontSize: 16,
        fontWeight: '500',
        color: COLORS.text,
    },
    suggestionsSubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
        marginBottom: 16,
    },
    // Adjustment chips for refining outfits
    adjustmentSection: {
        marginTop: 12,
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
    },
    adjustmentLabel: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginBottom: 8,
    },
    adjustmentScroll: {
        marginHorizontal: -4,
    },
    adjustmentChip: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.background,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 16,
        marginHorizontal: 4,
        borderWidth: 1,
        borderColor: COLORS.primary + '40',
    },
    adjustmentChipText: {
        fontSize: 12,
        color: COLORS.primary,
        marginLeft: 4,
        fontWeight: '500',
    },

    // Input area
    inputWrapper: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 16,
        paddingBottom: Platform.OS === 'ios' ? 0 : 16,
        backgroundColor: COLORS.background,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surfaceLight,
        borderRadius: 24,
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    textInput: {
        flex: 1,
        height: 40,
        fontSize: 16,
        color: COLORS.text,
        marginRight: 10,
    },
    sendButton: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
    },
    sendButtonActive: {
        backgroundColor: COLORS.primary,
    },
});

export default OutfitAIScreen;
