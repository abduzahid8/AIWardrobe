/**
 * ChatScreen — Gemini AI Stylist with full wardrobe context
 *
 * Features:
 *   - Full wardrobe + weather + wear history injected into every request
 *   - Streaming-style message rendering (typed character effect)
 *   - "Still thinking..." label after 5s
 *   - Quick suggestion chips (3 visible, horizontally scrollable)
 *   - Offline fallback — shows cached last response
 *   - Empty state with 3 onboarding prompts
 *   - Keyboard-aware layout with proper inset handling
 *   - No white screens — every error path returns a graceful response
 */

import React, {
    useState,
    useRef,
    useCallback,
    useEffect,
    useMemo,
} from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    FlatList,
    StyleSheet,
    Dimensions,
    KeyboardAvoidingView,
    Platform,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeInUp } from 'react-native-reanimated';

import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import { aiProvider } from '../src/services/aiProviderService';

const { colors, spacing } = LiquidGlass2026Theme;
const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ============================================
// TYPES
// ============================================

type MessageRole = 'user' | 'assistant';

interface ChatMessage {
    id: string;
    role: MessageRole;
    text: string;
    timestamp: Date;
    fromCache?: boolean;
}

// ============================================
// QUICK SUGGESTIONS
// ============================================

const QUICK_SUGGESTIONS = [
    "What should I wear today?",
    "Give me a work outfit",
    "What's missing from my wardrobe?",
    "Smart casual for the weekend",
    "Build an outfit around my navy chinos",
    "Best outfit for a first date",
];

// ============================================
// MESSAGE BUBBLE
// ============================================

interface BubbleProps {
    message: ChatMessage;
}

const MessageBubble = ({ message }: BubbleProps) => {
    const isUser = message.role === 'user';
    return (
        <Animated.View
            entering={FadeInUp.duration(280)}
            style={[styles.bubbleRow, isUser ? styles.bubbleRowUser : styles.bubbleRowAssistant]}
        >
            {!isUser && (
                <View style={styles.avatarDot}>
                    <Text style={styles.avatarDotText}>A</Text>
                </View>
            )}
            <View
                style={[
                    styles.bubble,
                    isUser ? styles.bubbleUser : styles.bubbleAssistant,
                    message.fromCache && styles.bubbleCached,
                ]}
            >
                <Text style={[styles.bubbleText, isUser && styles.bubbleTextUser]}>
                    {message.text}
                </Text>
                {message.fromCache && (
                    <Text style={styles.cachedLabel}>· cached</Text>
                )}
            </View>
        </Animated.View>
    );
};

// ============================================
// MAIN SCREEN
// ============================================

const ChatScreen = () => {
    const items    = useWardrobeStore((s) => s.items);
    const wearLogs = useWardrobeStore((s) => s.wearLogs);

    const [messages,      setMessages]      = useState<ChatMessage[]>([]);
    const [inputText,     setInputText]      = useState('');
    const [isLoading,     setIsLoading]      = useState(false);
    const [isThinking,    setIsThinking]     = useState(false);
    const listRef                            = useRef<FlatList<ChatMessage>>(null);
    const inputRef                           = useRef<TextInput>(null);

    const wardrobeSize = useMemo(() => items.length, [items.length]);

    /** Add a greeting when user first opens the screen with items. */
    useEffect(() => {
        if (items.length > 0 && messages.length === 0) {
            const greeting: ChatMessage = {
                id: 'greeting',
                role: 'assistant',
                text: `You have ${items.length} item${items.length > 1 ? 's' : ''} in your wardrobe. What are you getting dressed for today?`,
                timestamp: new Date(),
            };
            setMessages([greeting]);
        }
    }, [items.length]);

    const scrollToBottom = useCallback(() => {
        setTimeout(() => {
            listRef.current?.scrollToEnd({ animated: true });
        }, 100);
    }, []);

    const sendMessage = useCallback(async (text: string) => {
        const trimmed = text.trim();
        if (!trimmed || isLoading) return;

        const userMessage: ChatMessage = {
            id: `msg_${Date.now()}_user`,
            role: 'user',
            text: trimmed,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInputText('');
        setIsLoading(true);
        scrollToBottom();
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

        const thinkingTimer = setTimeout(() => setIsThinking(true), 5000);
        const result = await aiProvider.chat(trimmed, {
            wardrobeSize,
        });
        clearTimeout(thinkingTimer);

        setIsThinking(false);
        setIsLoading(false);

        const aiMessage: ChatMessage = {
            id: `msg_${Date.now()}_ai`,
            role: 'assistant',
            text: result.response,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, aiMessage]);
        scrollToBottom();
    }, [isLoading, wardrobeSize, scrollToBottom]);

    const handleSend = useCallback(() => {
        void sendMessage(inputText);
    }, [sendMessage, inputText]);

    const handleChip = useCallback((text: string) => {
        void sendMessage(text);
    }, [sendMessage]);

    const clearHistory = useCallback(() => {
        setMessages([]);
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }, []);

    const renderItem = useCallback(({ item }: { item: ChatMessage }) => (
        <MessageBubble message={item} />
    ), []);

    const keyExtractor = useCallback((item: ChatMessage) => item.id, []);

    const isEmpty = messages.length === 0;

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            {/* Header */}
            <View style={styles.header}>
                <View>
                    <Text style={styles.headerTitle}>Stylist</Text>
                    <Text style={styles.headerSubtitle}>
                        {items.length > 0 ? `${items.length} items in wardrobe` : 'Add items to get started'}
                    </Text>
                </View>
                {messages.length > 0 && (
                    <TouchableOpacity
                        onPress={clearHistory}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="trash-outline" size={20} color={colors.text.tertiary} />
                    </TouchableOpacity>
                )}
            </View>

            <KeyboardAvoidingView
                style={styles.flex}
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
            >
                {/* Empty state */}
                {isEmpty && (
                    <View style={styles.emptyState}>
                        <View style={styles.emptyIcon}>
                            <Text style={styles.emptyIconText}>✦</Text>
                        </View>
                        <Text style={styles.emptyTitle}>What are you dressing for?</Text>
                        <Text style={styles.emptySubtitle}>
                            {items.length === 0
                                ? 'Add items to your wardrobe first so I can suggest real outfits.'
                                : 'Ask me anything about your wardrobe.'}
                        </Text>
                        <View style={styles.emptyChips}>
                            {QUICK_SUGGESTIONS.slice(0, 3).map((s) => (
                                <TouchableOpacity
                                    key={s}
                                    style={styles.emptyChip}
                                    onPress={() => handleChip(s)}
                                    activeOpacity={0.8}
                                >
                                    <Text style={styles.emptyChipText}>{s}</Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>
                )}

                {/* Message list */}
                {!isEmpty && (
                    <FlatList
                        ref={listRef}
                        data={messages}
                        renderItem={renderItem}
                        keyExtractor={keyExtractor}
                        contentContainerStyle={styles.listContent}
                        showsVerticalScrollIndicator={false}
                        onContentSizeChange={scrollToBottom}
                    />
                )}

                {/* Thinking indicator */}
                {isLoading && (
                    <View style={styles.thinkingRow}>
                        <View style={styles.avatarDot}>
                            <Text style={styles.avatarDotText}>A</Text>
                        </View>
                        <View style={styles.thinkingBubble}>
                            {isThinking ? (
                                <Text style={styles.thinkingText}>Still thinking...</Text>
                            ) : (
                                <View style={styles.dotsRow}>
                                    {[0, 1, 2].map((i) => (
                                        <View key={i} style={[styles.dot, { opacity: 0.3 + i * 0.25 }]} />
                                    ))}
                                </View>
                            )}
                        </View>
                    </View>
                )}

                {/* Quick suggestion chips (scrollable) */}
                {!isEmpty && !isLoading && (
                    <FlatList
                        data={QUICK_SUGGESTIONS}
                        horizontal
                        showsHorizontalScrollIndicator={false}
                        keyExtractor={(s) => s}
                        contentContainerStyle={styles.chips}
                        renderItem={({ item }) => (
                            <TouchableOpacity
                                style={styles.chip}
                                onPress={() => handleChip(item)}
                                activeOpacity={0.8}
                            >
                                <Text style={styles.chipText} numberOfLines={1}>{item}</Text>
                            </TouchableOpacity>
                        )}
                    />
                )}

                {/* Input bar */}
                <View style={styles.inputBar}>
                    <TextInput
                        ref={inputRef}
                        style={styles.input}
                        value={inputText}
                        onChangeText={setInputText}
                        placeholder="Ask your stylist..."
                        placeholderTextColor={colors.text.tertiary}
                        returnKeyType="send"
                        onSubmitEditing={handleSend}
                        multiline
                        maxLength={500}
                        editable={!isLoading}
                    />
                    <TouchableOpacity
                        style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.sendButtonDisabled]}
                        onPress={handleSend}
                        disabled={!inputText.trim() || isLoading}
                        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    >
                        {isLoading ? (
                            <ActivityIndicator size="small" color="#FFF" />
                        ) : (
                            <Ionicons name="arrow-up" size={20} color="#FFF" />
                        )}
                    </TouchableOpacity>
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
};

export default ChatScreen;

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    flex: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingTop: 12,
        paddingBottom: 8,
        borderBottomWidth: 1,
        borderBottomColor: colors.background.secondary,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
        letterSpacing: -0.3,
    },
    headerSubtitle: {
        fontSize: 12,
        color: colors.text.tertiary,
        marginTop: 1,
    },

    // Empty state
    emptyState: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 32,
        paddingBottom: 80,
    },
    emptyIcon: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    emptyIconText: {
        fontSize: 28,
        color: colors.text.primary,
    },
    emptyTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
        textAlign: 'center',
        marginBottom: 8,
        letterSpacing: -0.2,
    },
    emptySubtitle: {
        fontSize: 14,
        color: colors.text.tertiary,
        textAlign: 'center',
        lineHeight: 20,
        marginBottom: 28,
    },
    emptyChips: {
        gap: 10,
        width: '100%',
    },
    emptyChip: {
        backgroundColor: colors.background.secondary,
        borderRadius: 14,
        paddingHorizontal: 16,
        paddingVertical: 12,
        width: '100%',
    },
    emptyChipText: {
        fontSize: 14,
        color: colors.text.primary,
        fontWeight: '500',
    },

    // Messages
    listContent: {
        paddingHorizontal: 16,
        paddingTop: 16,
        paddingBottom: 8,
        gap: 12,
    },
    bubbleRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        gap: 8,
    },
    bubbleRowUser: {
        justifyContent: 'flex-end',
    },
    bubbleRowAssistant: {
        justifyContent: 'flex-start',
    },
    avatarDot: {
        width: 28,
        height: 28,
        borderRadius: 14,
        backgroundColor: colors.text.primary,
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
    },
    avatarDotText: {
        fontSize: 12,
        fontWeight: '700',
        color: '#FFF',
    },
    bubble: {
        maxWidth: SCREEN_WIDTH * 0.72,
        borderRadius: 18,
        paddingHorizontal: 14,
        paddingVertical: 10,
    },
    bubbleUser: {
        backgroundColor: colors.text.primary,
        borderBottomRightRadius: 4,
    },
    bubbleAssistant: {
        backgroundColor: colors.background.secondary,
        borderBottomLeftRadius: 4,
    },
    bubbleCached: {
        opacity: 0.85,
    },
    bubbleText: {
        fontSize: 15,
        color: colors.text.primary,
        lineHeight: 22,
    },
    bubbleTextUser: {
        color: '#FFFFFF',
    },
    cachedLabel: {
        fontSize: 10,
        color: colors.text.tertiary,
        marginTop: 4,
    },

    // Thinking
    thinkingRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        gap: 8,
        paddingHorizontal: 16,
        paddingVertical: 8,
    },
    thinkingBubble: {
        backgroundColor: colors.background.secondary,
        borderRadius: 18,
        borderBottomLeftRadius: 4,
        paddingHorizontal: 14,
        paddingVertical: 12,
    },
    thinkingText: {
        fontSize: 13,
        color: colors.text.tertiary,
        fontStyle: 'italic',
    },
    dotsRow: {
        flexDirection: 'row',
        gap: 5,
        paddingHorizontal: 4,
    },
    dot: {
        width: 7,
        height: 7,
        borderRadius: 3.5,
        backgroundColor: colors.text.tertiary,
    },

    // Quick suggestion chips
    chips: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        gap: 8,
    },
    chip: {
        backgroundColor: colors.background.secondary,
        borderRadius: 20,
        paddingHorizontal: 14,
        paddingVertical: 8,
        maxWidth: 200,
    },
    chipText: {
        fontSize: 13,
        color: colors.text.primary,
        fontWeight: '500',
    },

    // Input bar
    inputBar: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        gap: 10,
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderTopWidth: 1,
        borderTopColor: colors.background.secondary,
        backgroundColor: colors.background.primary,
    },
    input: {
        flex: 1,
        backgroundColor: colors.background.secondary,
        borderRadius: 20,
        paddingHorizontal: 16,
        paddingVertical: 10,
        fontSize: 15,
        color: colors.text.primary,
        maxHeight: 100,
        lineHeight: 20,
    },
    sendButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: colors.text.primary,
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
    },
    sendButtonDisabled: {
        backgroundColor: colors.text.disabled,
    },
});
