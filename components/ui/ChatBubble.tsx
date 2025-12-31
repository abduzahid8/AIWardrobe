// Alta Daily-inspired Chat Bubble Component
// Features: User/AI variants, typing indicator, image previews, outfit cards

import React from 'react';
import { View, Text, StyleSheet, Image, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../src/hooks/useTheme';
import { Ionicons } from '@expo/vector-icons';

interface ChatBubbleProps {
    /** Message text */
    message?: string;
    /** Whether this is from the AI or user */
    isAI?: boolean;
    /** Optional image URL */
    imageUri?: string;
    /** Show typing indicator */
    isTyping?: boolean;
    /** Timestamp */
    timestamp?: string;
    /** Custom embedded content (e.g., outfit card) */
    children?: React.ReactNode;
    /** Custom style */
    style?: ViewStyle;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({
    message,
    isAI = false,
    imageUri,
    isTyping = false,
    timestamp,
    children,
    style,
}) => {
    const { colors, spacing, borderRadius, typography } = useTheme();

    if (isTyping) {
        return (
            <View style={[styles.container, styles.aiContainer, style]}>
                <View
                    style={[
                        styles.bubble,
                        styles.aiBubble,
                        { backgroundColor: colors.surfaceHighlight, borderRadius: borderRadius.l },
                    ]}
                >
                    <View style={styles.typingIndicator}>
                        <View style={[styles.dot, { backgroundColor: colors.text.secondary }]} />
                        <View style={[styles.dot, { backgroundColor: colors.text.secondary }]} />
                        <View style={[styles.dot, { backgroundColor: colors.text.secondary }]} />
                    </View>
                </View>
            </View>
        );
    }

    return (
        <View
            style={[
                styles.container,
                isAI ? styles.aiContainer : styles.userContainer,
                style,
            ]}
        >
            {/* AI Avatar (if AI message) */}
            {isAI && (
                <View
                    style={[
                        styles.avatar,
                        { backgroundColor: colors.primary, borderRadius: borderRadius.m },
                    ]}
                >
                    <Ionicons name="sparkles" size={18} color={colors.button.primaryText} />
                </View>
            )}

            <View
                style={[
                    styles.bubble,
                    isAI ? styles.aiBubble : styles.userBubble,
                    {
                        borderRadius: borderRadius.l,
                        maxWidth: '75%',
                    },
                ]}
            >
                {isAI ? (
                    // AI message - flat background
                    <View
                        style={[
                            styles.bubbleContent,
                            {
                                backgroundColor: colors.surfaceHighlight,
                                padding: spacing.m,
                                borderRadius: borderRadius.l,
                            },
                        ]}
                    >
                        {imageUri && (
                            <Image
                                source={{ uri: imageUri }}
                                style={[styles.image, { borderRadius: borderRadius.m, marginBottom: spacing.s }]}
                                resizeMode="cover"
                            />
                        )}
                        {message && (
                            <Text style={[styles.messageText, { color: colors.text.primary }]}>
                                {message}
                            </Text>
                        )}
                        {children}
                    </View>
                ) : (
                    // User message - gradient background
                    <LinearGradient
                        colors={[colors.primary, colors.primaryDark]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={[
                            styles.bubbleContent,
                            { padding: spacing.m, borderRadius: borderRadius.l },
                        ]}
                    >
                        {imageUri && (
                            <Image
                                source={{ uri: imageUri }}
                                style={[styles.image, { borderRadius: borderRadius.m, marginBottom: spacing.s }]}
                                resizeMode="cover"
                            />
                        )}
                        {message && (
                            <Text style={[styles.messageText, { color: colors.button.primaryText }]}>
                                {message}
                            </Text>
                        )}
                        {children}
                    </LinearGradient>
                )}
            </View>

            {/* Timestamp */}
            {timestamp && (
                <Text
                    style={[
                        styles.timestamp,
                        {
                            color: colors.text.muted,
                            fontSize: 11,
                            marginTop: spacing.xs,
                            textAlign: isAI ? 'left' : 'right',
                        },
                    ]}
                >
                    {timestamp}
                </Text>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: 16,
    },
    aiContainer: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        paddingRight: 48,
    },
    userContainer: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        justifyContent: 'flex-end',
        paddingLeft: 48,
    },
    avatar: {
        width: 36,
        height: 36,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 12,
    },
    bubble: {
        overflow: 'hidden',
    },
    aiBubble: {},
    userBubble: {},
    bubbleContent: {
        overflow: 'hidden',
    },
    messageText: {
        fontSize: 16,
        lineHeight: 22,
    },
    image: {
        width: '100%',
        height: 150,
    },
    timestamp: {
        fontWeight: '400',
    },
    typingIndicator: {
        flexDirection: 'row',
        gap: 6,
        padding: 12,
    },
    dot: {
        width: 8,
        height: 8,
        borderRadius: 4,
    },
});
