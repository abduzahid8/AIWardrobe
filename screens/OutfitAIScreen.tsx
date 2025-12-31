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

// Quick Occasion Suggestions
const OCCASION_SUGGESTIONS = [
    { id: '1', text: 'Job interview', icon: 'briefcase-outline' },
    { id: '2', text: 'Casual dinner', icon: 'restaurant-outline' },
    { id: '3', text: 'Weekend brunch', icon: 'cafe-outline' },
    { id: '4', text: 'Business meeting', icon: 'people-outline' },
    { id: '5', text: 'Date night', icon: 'heart-outline' },
    { id: '6', text: 'Workout', icon: 'fitness-outline' },
];

// Weather context interface
interface WeatherContext {
    temp: number;
    condition: string;
    location?: string;
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

// Chat Message Bubble
const ChatBubble = ({ message, isAI, outfit }: { message: string; isAI: boolean; outfit?: any }) => {
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
                            {outfit.items.slice(0, 4).map((item: any, idx: number) => (
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
                    </View>
                )}
            </View>
        </Animated.View>
    );
};

// Occasion Chip
const OccasionChip = ({ occasion, onPress }: { occasion: any; onPress: () => void }) => {
    return (
        <TouchableOpacity
            style={styles.occasionChip}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
        >
            <Ionicons name={occasion.icon} size={16} color={COLORS.primary} />
            <Text style={styles.occasionChipText}>{occasion.text}</Text>
        </TouchableOpacity>
    );
};

interface ChatMessage {
    id: string;
    text: string;
    isAI: boolean;
    outfit?: any;
    suggestedOutfits?: any[];
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

    const handleOccasionPress = (occasion: any) => {
        sendMessage(`I need an outfit for ${occasion.text.toLowerCase()}`);
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
                            <Text style={styles.suggestionsTitle}>Choose an occasion</Text>
                            <View style={styles.occasionGrid}>
                                {OCCASION_SUGGESTIONS.map((occasion) => (
                                    <OccasionChip
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
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    occasionChip: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surfaceLight,
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    occasionChipText: {
        fontSize: 14,
        color: COLORS.text,
        marginLeft: 6,
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
