import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, CommonActions } from '@react-navigation/native';

/**
 * NotFoundScreen — fallback for unknown routes or deep-link errors.
 */
const NotFoundScreen = () => {
    const navigation = useNavigation();

    const handleGoHome = () => {
        navigation.dispatch(
            CommonActions.reset({
                index: 0,
                routes: [{ name: 'Main' }],
            })
        );
    };

    return (
        <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.container}>
            <View style={styles.content}>
                <Ionicons name="compass-outline" size={64} color="rgba(255,255,255,0.3)" />
                <Text style={styles.code} accessibilityRole="header">404</Text>
                <Text style={styles.title}>Page Not Found</Text>
                <Text style={styles.message}>
                    The screen you're looking for doesn't exist or has been moved.
                </Text>
                <TouchableOpacity
                    onPress={handleGoHome}
                    style={styles.button}
                    activeOpacity={0.8}
                    accessibilityLabel="Go to home screen"
                    accessibilityRole="button"
                >
                    <Text style={styles.buttonText}>Go Home</Text>
                </TouchableOpacity>
            </View>
        </LinearGradient>
    );
};

export default NotFoundScreen;

const styles = StyleSheet.create({
    container: { flex: 1 },
    content: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 32,
    },
    code: {
        fontSize: 64,
        fontWeight: '900',
        color: 'rgba(255,255,255,0.15)',
        marginTop: 16,
    },
    title: {
        fontSize: 22,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 8,
    },
    message: {
        fontSize: 15,
        color: 'rgba(255,255,255,0.5)',
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 32,
    },
    button: {
        backgroundColor: '#FFF',
        paddingVertical: 14,
        paddingHorizontal: 32,
        borderRadius: 16,
    },
    buttonText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#000',
    },
});
