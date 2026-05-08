/**
 * AITryOnScreen — the 3rd tab ("AI").
 *
 * Access matrix:
 *   Premium/VIP users (isPremium=true)  → full try-on experience
 *   Free users                         → "Premium Feature" upsell
 */

import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import TryOnFeatureScreen from '../features/try-on/AITryOnScreen';
import useSubscriptionStore from '../store/subscriptionStore';
import { useTranslation } from 'react-i18next';
import { useNavigation } from '@react-navigation/native';

export default function AITryOnScreen(props: any) {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const { isPremium, isLoading } = useSubscriptionStore();

    if (isLoading) {
        return (
            <SafeAreaView style={styles.center}>
                <ActivityIndicator size="large" color="#183A67" />
            </SafeAreaView>
        );
    }

    if (!isPremium) {
        return (
            <LinearGradient colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']} style={styles.flex}>
                <SafeAreaView style={styles.center}>
                    <View style={styles.iconWrap}>
                        <Ionicons name="sparkles" size={44} color="#183A67" />
                    </View>
                    <Text style={styles.title}>{t('aiTryOn.premiumFeature', 'Premium Feature')}</Text>
                    <Text style={styles.subtitle}>
                        {t(
                            'aiTryOn.premiumFeatureBody',
                            'Virtual Try-On is available exclusively for Pro and Max subscribers. Upgrade to start trying on clothes!'
                        )}
                    </Text>
                    <TouchableOpacity
                        style={styles.upgradeButton}
                        onPress={() => navigation.navigate('Paywall' as never)}
                    >
                        <Text style={styles.upgradeButtonText}>
                            {t('aiTryOn.upgradeToAccess', 'Upgrade to Access')}
                        </Text>
                    </TouchableOpacity>
                </SafeAreaView>
            </LinearGradient>
        );
    }

    return <TryOnFeatureScreen {...props} />;
}

const styles = StyleSheet.create({
    flex: { flex: 1 },
    center: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 32,
    },
    iconWrap: {
        width: 88,
        height: 88,
        borderRadius: 44,
        backgroundColor: 'rgba(24,58,103,0.08)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    title: {
        fontSize: 26,
        fontWeight: '700',
        color: '#112A4A',
        marginBottom: 12,
        textAlign: 'center',
    },
    subtitle: {
        fontSize: 15,
        lineHeight: 22,
        color: '#5F6D84',
        textAlign: 'center',
        marginBottom: 24,
    },
    upgradeButton: {
        backgroundColor: '#183A67',
        paddingHorizontal: 24,
        paddingVertical: 14,
        borderRadius: 12,
        shadowColor: '#183A67',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.25,
        shadowRadius: 8,
        elevation: 4,
    },
    upgradeButtonText: {
        color: '#FFFFFF',
        fontSize: 16,
        fontWeight: '600',
    },
});
