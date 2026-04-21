/**
 * ALTA DAILY - PIXEL PERFECT AI SCREEN
 * Based on exact design specification
 */

import React, { useState, useRef, useCallback } from 'react';
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
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import Animated, {
    FadeIn,
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from '../../src/config/env';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS
const ALTA = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#0A1931',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
    inputBg: '#F0F0F0',
};

const ALICEVISION_API = Config.api.alicevisionUrl;
const API_URL = Config.api.url;

const SUGGESTIONS = [
    'Style me for a date night',
    'What should I wear today?',
    'Help me pack for vacation',
    'Create an outfit from my closet',
];

interface Message {
    id: string;
    text: string;
    isAI: boolean;
}

// Suggestion Chip with scale animation
const SuggestionChip = ({ text, onPress }: { text: string; onPress: () => void }) => {
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
        >
            <Animated.View style={[styles.suggestionChip, animatedStyle]}>
                <Text style={styles.suggestionText}>{text}</Text>
            </Animated.View>
        </TouchableOpacity>
    );
};

// Chat Bubble
const ChatBubble = ({ message }: { message: Message }) => (
    <Animated.View
        entering={FadeInUp.springify()}
        style={[styles.bubbleContainer, message.isAI ? styles.aiBubbleContainer : styles.userBubbleContainer]}
    >
        <View style={[styles.bubble, message.isAI ? styles.aiBubble : styles.userBubble]}>
            <Text style={[styles.bubbleText, message.isAI ? styles.aiBubbleText : styles.userBubbleText]}>
                {message.text}
            </Text>
        </View>
    </Animated.View>
);

const AltaAIScreen = () => {
    const navigation = useNavigation();
    const scrollViewRef = useRef<ScrollView>(null);
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [showWelcome, setShowWelcome] = useState(true);
    const [loading, setLoading] = useState(false);
    const [wardrobe, setWardrobe] = useState<any[]>([]);

    useFocusEffect(useCallback(() => {
        AsyncStorage.getItem('wardrobeItems').then(data => {
            if (data) setWardrobe(JSON.parse(data));
        });
    }, []));

    const sendMessage = async (text: string) => {
        if (!text.trim() || loading) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setShowWelcome(false);
        setLoading(true);

        const userMsg: Message = { id: Date.now().toString(), text, isAI: false };
        setMessages(prev => [...prev, userMsg]);
        setMessage('');

        try {
            let response = await fetch(`${ALICEVISION_API}/stylist/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, wardrobe_items: wardrobe.slice(0, 10) }),
            }).catch(() => null);

            if (!response?.ok) {
                response = await fetch(`${API_URL}/ai-chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text }),
                });
            }

            const data = await response?.json();
            const aiText = data?.response || data?.reply || data?.message ||
                "I'd love to help you style an outfit! Tell me about the occasion.";

            const aiMsg: Message = { id: (Date.now() + 1).toString(), text: aiText, isAI: true };
            setMessages(prev => [...prev, aiMsg]);
        } catch (e) {
            const aiMsg: Message = {
                id: (Date.now() + 1).toString(),
                text: "I'm having trouble connecting. Let me suggest some styling tips!",
                isAI: true
            };
            setMessages(prev => [...prev, aiMsg]);
        } finally {
            setLoading(false);
            scrollViewRef.current?.scrollToEnd({ animated: true });
        }
    };

    const pickImage = async () => {
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.8,
        });

        if (!result.canceled && result.assets[0]) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            sendMessage("Can you help me style this item?");
        }
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={ALTA.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header */}
                <View style={styles.header}>
                    <View style={styles.headerCenter}>
                        <Text style={styles.headerTitle}>AI Stylist</Text>
                    </View>
                </View>

                {/* Chat Content */}
                <ScrollView
                    ref={scrollViewRef}
                    style={styles.scrollView}
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {showWelcome && (
                        <Animated.View entering={FadeIn.delay(100).duration(400)} style={styles.welcomeSection}>
                            <View style={styles.aiIcon}>
                                <Ionicons name="sparkles" size={32} color={ALTA.text} />
                            </View>
                            <Text style={styles.welcomeTitle}>How can I style you today?</Text>
                            <Text style={styles.welcomeSubtitle}>
                                I can help you create outfits, plan for events, or explore your wardrobe.
                            </Text>
                            <View style={styles.suggestionsContainer}>
                                {SUGGESTIONS.map((text, i) => (
                                    <SuggestionChip key={i} text={text} onPress={() => sendMessage(text)} />
                                ))}
                            </View>
                        </Animated.View>
                    )}

                    {messages.map(msg => <ChatBubble key={msg.id} message={msg} />)}

                    {loading && (
                        <View style={styles.loadingContainer}>
                            <ActivityIndicator size="small" color={ALTA.text} />
                        </View>
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>

                {/* Input Area */}
                <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
                    <View style={styles.inputWrapper}>
                        <View style={styles.inputContainer}>
                            <TouchableOpacity style={styles.inputIcon} onPress={pickImage}>
                                <Ionicons name="image-outline" size={22} color={ALTA.textMuted} />
                            </TouchableOpacity>
                            <TouchableOpacity style={styles.inputIcon} onPress={() => (navigation as any).navigate('Home')}>
                                <Ionicons name="shirt-outline" size={22} color={ALTA.textMuted} />
                            </TouchableOpacity>
                            <TextInput
                                style={styles.textInput}
                                placeholder="Ask me anything..."
                                placeholderTextColor={ALTA.textMuted}
                                value={message}
                                onChangeText={setMessage}
                                onSubmitEditing={() => sendMessage(message)}
                                returnKeyType="send"
                                maxLength={500}
                            />
                            {message.trim() ? (
                                <TouchableOpacity style={styles.sendButton} onPress={() => sendMessage(message)}>
                                    <Ionicons name="arrow-up" size={20} color={ALTA.background} />
                                </TouchableOpacity>
                            ) : (
                                <TouchableOpacity style={styles.inputIcon}>
                                    <Ionicons name="mic-outline" size={22} color={ALTA.textMuted} />
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
    container: { flex: 1, backgroundColor: ALTA.background },
    safeArea: { flex: 1 },

    // Header
    header: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 0.5,
        borderBottomColor: ALTA.border,
    },
    headerCenter: { alignItems: 'center' },
    headerTitle: { fontSize: 17, fontWeight: '600', color: ALTA.text },

    // Content
    scrollView: { flex: 1 },
    scrollContent: { paddingHorizontal: 20, paddingTop: 20 },

    // Welcome
    welcomeSection: { alignItems: 'center', paddingVertical: 40 },
    aiIcon: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: ALTA.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    welcomeTitle: { fontSize: 26, fontWeight: '700', color: ALTA.text, textAlign: 'center', marginBottom: 12 },
    welcomeSubtitle: { fontSize: 16, color: ALTA.textSecondary, textAlign: 'center', lineHeight: 24, marginBottom: 32, paddingHorizontal: 16 },

    // Suggestions
    suggestionsContainer: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10 },
    suggestionChip: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        backgroundColor: ALTA.surface,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: ALTA.border,
    },
    suggestionText: { fontSize: 14, fontWeight: '500', color: ALTA.text },

    // Bubbles
    bubbleContainer: { marginVertical: 6 },
    aiBubbleContainer: { alignItems: 'flex-start' },
    userBubbleContainer: { alignItems: 'flex-end' },
    bubble: { maxWidth: '85%', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 20 },
    aiBubble: { backgroundColor: ALTA.surface, borderBottomLeftRadius: 6 },
    userBubble: { backgroundColor: ALTA.text, borderBottomRightRadius: 6 },
    bubbleText: { fontSize: 15, lineHeight: 22 },
    aiBubbleText: { color: ALTA.text },
    userBubbleText: { color: ALTA.background },

    loadingContainer: { padding: 20, alignItems: 'flex-start' },

    // Input
    inputWrapper: { paddingHorizontal: 16, paddingVertical: 12, backgroundColor: ALTA.background, borderTopWidth: 0.5, borderTopColor: ALTA.border },
    inputContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: ALTA.inputBg, borderRadius: 28, paddingHorizontal: 6, paddingVertical: 4 },
    inputIcon: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
    textInput: { flex: 1, fontSize: 16, color: ALTA.text, paddingVertical: 10 },
    sendButton: { width: 36, height: 36, borderRadius: 18, backgroundColor: ALTA.text, alignItems: 'center', justifyContent: 'center', marginRight: 4 },
});

export default AltaAIScreen;
