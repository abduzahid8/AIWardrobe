/**
 * AIStylistScreen — Unified AI stylist combining chat + outfit generation.
 *
 * Consolidates the overlapping AIAssistant (text chat) and OutfitAIScreen
 * (outfit occasion chat) into a single screen with a tab toggle.
 *
 * Routes `AIChat` and `OutfitAI` both point here now; the initial tab
 * is determined by the `initialTab` route param.
 */

import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn } from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { colors, spacing, borderRadius } from '../src/theme';

// Lazy import existing screens as embedded views
import AIAssistant from './AIAssistant';
import OutfitAIScreen from './OutfitAIScreen';

type TabMode = 'chat' | 'outfit';

interface RouteParams {
    initialTab?: TabMode;
}

const AIStylistScreen = () => {
    const navigation = useNavigation();
    const route = useRoute();
    const params = (route.params as RouteParams) || {};
    const [activeTab, setActiveTab] = useState<TabMode>(params.initialTab || 'chat');

    const switchTab = (tab: TabMode) => {
        if (tab !== activeTab) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            setActiveTab(tab);
        }
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <Animated.View entering={FadeIn} style={styles.header}>
                    <TouchableOpacity
                        style={styles.backButton}
                        onPress={() => navigation.goBack()}
                    >
                        <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
                    </TouchableOpacity>

                    <Text style={styles.title}>AI Stylist</Text>

                    <View style={styles.placeholder} />
                </Animated.View>

                {/* Segmented Control */}
                <Animated.View entering={FadeIn.delay(100)} style={styles.segmentContainer}>
                    <View style={styles.segmentControl}>
                        <TouchableOpacity
                            style={[styles.segment, activeTab === 'chat' && styles.segmentActive]}
                            onPress={() => switchTab('chat')}
                        >
                            <Ionicons
                                name="chatbubbles-outline"
                                size={16}
                                color={activeTab === 'chat' ? '#FFF' : colors.text.secondary}
                            />
                            <Text style={[styles.segmentText, activeTab === 'chat' && styles.segmentTextActive]}>
                                Style Chat
                            </Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.segment, activeTab === 'outfit' && styles.segmentActive]}
                            onPress={() => switchTab('outfit')}
                        >
                            <Ionicons
                                name="shirt-outline"
                                size={16}
                                color={activeTab === 'outfit' ? '#FFF' : colors.text.secondary}
                            />
                            <Text style={[styles.segmentText, activeTab === 'outfit' && styles.segmentTextActive]}>
                                Outfit AI
                            </Text>
                        </TouchableOpacity>
                    </View>
                </Animated.View>

                {/* Tab Content — render both but hide inactive for state persistence */}
                <View style={styles.content}>
                    <View style={[styles.tabPane, activeTab !== 'chat' && styles.tabHidden]}>
                        <AIAssistant />
                    </View>
                    <View style={[styles.tabPane, activeTab !== 'outfit' && styles.tabHidden]}>
                        <OutfitAIScreen />
                    </View>
                </View>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
    },
    backButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
    },
    title: {
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
    },
    placeholder: {
        width: 40,
    },
    segmentContainer: {
        paddingHorizontal: spacing.l,
        paddingBottom: spacing.s,
    },
    segmentControl: {
        flexDirection: 'row',
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        padding: 3,
    },
    segment: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 10,
        borderRadius: borderRadius.l - 2,
        gap: 6,
    },
    segmentActive: {
        backgroundColor: colors.button.primary,
    },
    segmentText: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    segmentTextActive: {
        color: '#FFF',
    },
    content: {
        flex: 1,
    },
    tabPane: {
        flex: 1,
    },
    tabHidden: {
        display: 'none',
    },
});

export default AIStylistScreen;
