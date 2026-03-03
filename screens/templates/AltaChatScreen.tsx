/**
 * ALTA DAILY - EXACT DESIGN COPY
 * AI Chat Screen
 * 
 * Features from Alta:
 * - Close button top-left
 * - Clean white background
 * - Centered welcome with sparkle icon
 * - Suggestion chips
 * - Chat bubbles (white for AI, black for user)
 * - Input bar with: image, wardrobe, text input, mic
 * - Send button (black circle with arrow)
 */

import React, { useState, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ScrollView,
    TextInput,
    TouchableOpacity,
    KeyboardAvoidingView,
    Platform,
    StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInUp, FadeInDown } from 'react-native-reanimated';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS
const COLORS = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#0A1931',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E8E8E8',
    inputBg: '#F0F0F0',
};

// Suggestion chips
const SUGGESTIONS = [
    'Style me for a date night',
    'What should I wear to work?',
    'Help me pack for vacation',
    'Create an outfit from my closet',
];

// Chat message interface
interface Message {
    id: string;
    text: string;
    isAI: boolean;
}

// Suggestion Chip
const SuggestionChip = ({ text, onPress }: { text: string; onPress: () => void }) => (
    <TouchableOpacity
        style={styles.suggestionChip}
        onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
        }}
        activeOpacity={0.7}
    >
        <Text style={styles.suggestionText}>{text}</Text>
    </TouchableOpacity>
);

// Chat Bubble
const ChatBubble = ({ message }: { message: Message }) => (
    <Animated.View
        entering={FadeInUp.springify()}
        style={[
            styles.bubbleContainer,
            message.isAI ? styles.aiBubbleContainer : styles.userBubbleContainer
        ]}
    >
        <View style={[
            styles.bubble,
            message.isAI ? styles.aiBubble : styles.userBubble
        ]}>
            <Text style={[
                styles.bubbleText,
                message.isAI ? styles.aiBubbleText : styles.userBubbleText
            ]}>
                {message.text}
            </Text>
        </View>
    </Animated.View>
);

const AltaChatScreen = () => {
    const navigation = useNavigation();
    const scrollViewRef = useRef<ScrollView>(null);
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [showWelcome, setShowWelcome] = useState(true);

    const sendMessage = (text: string) => {
        if (!text.trim()) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setShowWelcome(false);

        // Add user message
        const userMsg: Message = { id: Date.now().toString(), text, isAI: false };
        setMessages(prev => [...prev, userMsg]);
        setMessage('');

        // Simulate AI response
        setTimeout(() => {
            const aiMsg: Message = {
                id: (Date.now() + 1).toString(),
                text: "I'd love to help you with that! Let me look through your wardrobe and create the perfect outfit for you.",
                isAI: true
            };
            setMessages(prev => [...prev, aiMsg]);
            scrollViewRef.current?.scrollToEnd({ animated: true });
        }, 1000);
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
            <SafeAreaView style={styles.safeArea}>

                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity
                        style={styles.closeButton}
                        onPress={() => navigation.goBack()}
                    >
                        <Ionicons name="close" size={26} color={COLORS.text} />
                    </TouchableOpacity>

                    <View style={{ flex: 1 }} />
                </View>

                {/* Chat Content */}
                <ScrollView
                    ref={scrollViewRef}
                    style={styles.scrollView}
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                    onContentSizeChange={() => scrollViewRef.current?.scrollToEnd()}
                >
                    {/* Welcome State */}
                    {showWelcome && (
                        <Animated.View
                            entering={FadeIn.delay(100).duration(400)}
                            style={styles.welcomeSection}
                        >
                            <View style={styles.aiIcon}>
                                <Ionicons name="sparkles" size={32} color={COLORS.text} />
                            </View>

                            <Text style={styles.welcomeTitle}>
                                How can I style you today?
                            </Text>
                            <Text style={styles.welcomeSubtitle}>
                                I can help you create outfits, plan for events, or explore your wardrobe.
                            </Text>

                            {/* Suggestions */}
                            <View style={styles.suggestionsContainer}>
                                {SUGGESTIONS.map((text, i) => (
                                    <SuggestionChip
                                        key={i}
                                        text={text}
                                        onPress={() => sendMessage(text)}
                                    />
                                ))}
                            </View>
                        </Animated.View>
                    )}

                    {/* Messages */}
                    {messages.map(msg => (
                        <ChatBubble key={msg.id} message={msg} />
                    ))}

                    <View style={{ height: 100 }} />
                </ScrollView>

                {/* Input Area */}
                <KeyboardAvoidingView
                    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                    keyboardVerticalOffset={10}
                >
                    <View style={styles.inputWrapper}>
                        <View style={styles.inputContainer}>
                            {/* Image button */}
                            <TouchableOpacity style={styles.inputIcon}>
                                <Ionicons name="image-outline" size={22} color={COLORS.textMuted} />
                            </TouchableOpacity>

                            {/* Wardrobe button */}
                            <TouchableOpacity style={styles.inputIcon}>
                                <Ionicons name="shirt-outline" size={22} color={COLORS.textMuted} />
                            </TouchableOpacity>

                            {/* Text Input */}
                            <TextInput
                                style={styles.textInput}
                                placeholder="Ask me anything..."
                                placeholderTextColor={COLORS.textMuted}
                                value={message}
                                onChangeText={setMessage}
                                onSubmitEditing={() => sendMessage(message)}
                                returnKeyType="send"
                                maxLength={500}
                            />

                            {/* Send or Mic */}
                            {message.trim() ? (
                                <TouchableOpacity
                                    style={styles.sendButton}
                                    onPress={() => sendMessage(message)}
                                >
                                    <Ionicons name="arrow-up" size={20} color={COLORS.background} />
                                </TouchableOpacity>
                            ) : (
                                <TouchableOpacity style={styles.inputIcon}>
                                    <Ionicons name="mic-outline" size={22} color={COLORS.textMuted} />
                                </TouchableOpacity>
                            )}
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
        paddingHorizontal: 12,
        paddingVertical: 8,
    },
    closeButton: {
        padding: 8,
    },

    // Content
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: 20,
        paddingTop: 20,
    },

    // Welcome
    welcomeSection: {
        alignItems: 'center',
        paddingVertical: 40,
    },
    aiIcon: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    welcomeTitle: {
        fontSize: 26,
        fontWeight: '700',
        color: COLORS.text,
        textAlign: 'center',
        marginBottom: 12,
    },
    welcomeSubtitle: {
        fontSize: 16,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 24,
        marginBottom: 32,
        paddingHorizontal: 16,
    },

    // Suggestions
    suggestionsContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: 10,
    },
    suggestionChip: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        backgroundColor: COLORS.surface,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    suggestionText: {
        fontSize: 14,
        fontWeight: '500',
        color: COLORS.text,
    },

    // Chat Bubbles
    bubbleContainer: {
        marginVertical: 6,
    },
    aiBubbleContainer: {
        alignItems: 'flex-start',
    },
    userBubbleContainer: {
        alignItems: 'flex-end',
    },
    bubble: {
        maxWidth: '85%',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 20,
    },
    aiBubble: {
        backgroundColor: COLORS.surface,
        borderBottomLeftRadius: 6,
    },
    userBubble: {
        backgroundColor: COLORS.text,
        borderBottomRightRadius: 6,
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

    // Input
    inputWrapper: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        backgroundColor: COLORS.background,
        borderTopWidth: 0.5,
        borderTopColor: COLORS.border,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.inputBg,
        borderRadius: 28,
        paddingHorizontal: 6,
        paddingVertical: 4,
    },
    inputIcon: {
        width: 44,
        height: 44,
        alignItems: 'center',
        justifyContent: 'center',
    },
    textInput: {
        flex: 1,
        fontSize: 16,
        color: COLORS.text,
        paddingVertical: 10,
    },
    sendButton: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: COLORS.text,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 4,
    },
});

export default AltaChatScreen;
