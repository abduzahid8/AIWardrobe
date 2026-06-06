import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Platform } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { useNetInfo } from '@react-native-community/netinfo';

/**
 * NetworkBanner — shows a banner when the device cannot reach the network.
 *
 * Uses @react-native-community/netinfo which relies on native OS APIs
 * to determine true network connection status reliably.
 */
const NetworkBanner: React.FC = () => {
    const { t } = useTranslation();
    const netInfo = useNetInfo();
    const slideAnim = useRef(new Animated.Value(-50)).current;

    // netInfo.isConnected is null while loading. We default to true to avoid flicker.
    // Some simulators return false for isInternetReachable even when connected.
    // We strictly look at `isConnected` which means Wi-Fi or Cellular is active.
    const isOnline = netInfo.isConnected ?? true;

    useEffect(() => {
        Animated.spring(slideAnim, {
            toValue: isOnline ? -50 : 0,
            useNativeDriver: true,
            friction: 8,
        }).start();
    }, [isOnline, slideAnim]);

    return (
        <Animated.View
            style={[
                styles.banner,
                { transform: [{ translateY: slideAnim }] },
            ]}
            accessibilityLabel={isOnline ? undefined : t('network.noInternetConnection')}
            accessibilityRole="alert"
            pointerEvents={isOnline ? 'none' : 'auto'}
        >
            <Ionicons name="cloud-offline-outline" size={16} color="#FFF" />
            <ScaledText style={styles.text}>{t('network.noInternetConnection')}</ScaledText>
        </Animated.View>
    );
};

export default NetworkBanner;

const styles = StyleSheet.create({
    banner: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        backgroundColor: '#EF4444',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingTop: Platform.OS === 'ios' ? 50 : 30,
        paddingBottom: 8,
        zIndex: 9999,
        gap: 6,
    },
    text: {
        color: '#FFF',
        fontSize: 13,
        fontWeight: '600',
    },
});
