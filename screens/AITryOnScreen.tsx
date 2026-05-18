import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { useNavigation, useRoute } from '@react-navigation/native';

export default function AITryOnScreen() {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const route = useRoute();
    const asTab = (route.params as any)?.asTab === true;

    return (
        <LinearGradient colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']} style={styles.flex}>
            <SafeAreaView style={styles.flex}>
                {!asTab && navigation.canGoBack() && (
                    <TouchableOpacity
                        style={styles.backButton}
                        onPress={() => navigation.goBack()}
                        activeOpacity={0.85}
                    >
                        <Ionicons name="chevron-back" size={18} color="#183A67" />
                        <Text style={styles.backButtonText}>{t('common.back', 'Back')}</Text>
                    </TouchableOpacity>
                )}
                <View style={styles.center}>
                    <View style={styles.iconWrap}>
                        <Ionicons name="sparkles" size={44} color="#183A67" />
                    </View>
                    <Text style={styles.title}>{t('aiTryOn.comingSoon', 'Coming Soon')}</Text>
                    <Text style={styles.subtitle}>
                        {t(
                            'aiTryOn.comingSoonBody',
                            "Virtual Try-On is getting an upgrade. We'll add it back soon!"
                        )}
                    </Text>
                </View>
            </SafeAreaView>
        </LinearGradient>
    );
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
    backButton: {
        alignSelf: 'flex-start',
        marginTop: 12,
        marginLeft: 20,
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 6,
        elevation: 3,
        zIndex: 10,
    },
    backButtonText: {
        fontSize: 14,
        fontWeight: '600',
        color: '#183A67',
        marginLeft: 6,
    },
});
