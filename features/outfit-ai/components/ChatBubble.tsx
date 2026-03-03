/**
 * ChatBubble — Renders a single chat message with styling based on role.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { colors, spacing, borderRadius } from '../../../src/theme';

interface ChatBubbleProps {
    role: 'user' | 'assistant' | 'system';
    content: string;
    index?: number;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ role, content, index = 0 }) => {
    const isUser = role === 'user';

    return (
        <Animated.View
            entering={FadeInDown.delay(index * 50).duration(300)}
            style={[styles.bubble, isUser ? styles.userBubble : styles.aiBubble]}
        >
            <Text style={[styles.text, isUser ? styles.userText : styles.aiText]}>
                {content}
            </Text>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    bubble: {
        maxWidth: '80%',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.l,
        marginBottom: spacing.s,
    },
    userBubble: {
        alignSelf: 'flex-end',
        backgroundColor: colors.button.primary,
        borderBottomRightRadius: 4,
    },
    aiBubble: {
        alignSelf: 'flex-start',
        backgroundColor: colors.surface,
        borderBottomLeftRadius: 4,
    },
    text: {
        fontSize: 15,
        lineHeight: 22,
    },
    userText: {
        color: '#FFF',
    },
    aiText: {
        color: colors.text.primary,
    },
});

export default ChatBubble;
