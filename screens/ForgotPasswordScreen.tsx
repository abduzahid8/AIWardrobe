import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform, ActivityIndicator, Alert, ScrollView } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import * as Linking from 'expo-linking';
import { supabase } from '../lib/supabase';
import { useTranslation } from 'react-i18next';

const ForgotPasswordScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSent, setIsSent] = useState(false);

    const isEmailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

    const handleResetPassword = async () => {
        if (!isEmailValid) {
            Alert.alert(t('forgotPassword.invalidEmail'), t('forgotPassword.enterValidEmail'));
            return;
        }

        setIsLoading(true);
        try {
            const redirectTo = Linking.createURL('reset-password');
            const { error } = await supabase.auth.resetPasswordForEmail(email, {
                redirectTo,
            });

            if (error) throw error;

            setIsSent(true);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (error: any) {
            const message = error.message || t('forgotPassword.failedSendResetEmail');
            if (message.includes('rate limit') || message.includes('too many')) {
                Alert.alert(t('forgotPassword.tooManyRequests'), t('forgotPassword.waitFewMinutes'));
            } else {
                Alert.alert(t('forgotPassword.error'), message);
            }
        } finally {
            setIsLoading(false);
        }
    };

    if (isSent) {
        return (
            <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.flex1}>
                <View style={styles.successContainer}>
                    <View style={styles.successIcon}>
                        <Ionicons name="mail-outline" size={48} color="#FFD700" />
                    </View>
                    <ScaledText style={styles.successTitle}>{t('forgotPasswordScreen.checkYourEmail')}</ScaledText>
                    <ScaledText style={styles.successMessage}>
                        {t('forgotPassword.resetLinkSent')}{'\n'}
                        <ScaledText style={styles.emailHighlight}>{email}</ScaledText>
                    </ScaledText>
                    <ScaledText style={styles.successHint}>
                        {t('forgotPassword.checkSpamFolder')}
                    </ScaledText>
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        style={styles.backButton}
                        accessibilityLabel={t('forgotPasswordScreen.backToSignIn')}
                        accessibilityRole="button"
                    >
                        <ScaledText style={styles.backButtonText}>{t('forgotPasswordScreen.backToSignIn')}</ScaledText>
                    </TouchableOpacity>
                </View>
            </LinearGradient>
        );
    }

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.flex1}
        >
            <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.flex1}>
                <ScrollView
                    contentContainerStyle={styles.container}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                    accessible={false}
                >
                    {/* Back arrow */}
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        style={styles.backArrow}
                        accessibilityLabel={t('resetPassword.goBack')}
                        accessibilityRole="button"
                    >
                        <Ionicons name="arrow-back" size={24} color="#FFF" />
                    </TouchableOpacity>

                    <ScaledText style={styles.title}>{t('forgotPasswordScreen.resetPassword')}</ScaledText>
                    <ScaledText style={styles.subtitle}>
                        {t('forgotPasswordScreen.enterEmail')}
                    </ScaledText>

                    <View style={styles.inputContainer}>
                        <TextInput
                            style={[styles.input, email && !isEmailValid && styles.inputError]}
                            value={email}
                            onChangeText={setEmail}
                            placeholder={t('auth.email')}
                            placeholderTextColor="rgba(255,255,255,0.4)"
                            keyboardType="email-address"
                            autoCapitalize="none"
                            autoFocus
                            maxLength={255}
                            accessibilityLabel={t('auth.email')}
                        />
                        {email && !isEmailValid && (
                            <ScaledText style={styles.errorHint}>{t('forgotPasswordScreen.pleaseEnterValidEmail')}</ScaledText>
                        )}
                    </View>

                    <TouchableOpacity
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            handleResetPassword();
                        }}
                        style={[styles.submitButton, (!isEmailValid || isLoading) && styles.submitButtonDisabled]}
                        disabled={!isEmailValid || isLoading}
                        activeOpacity={0.8}
                        accessibilityLabel={t('forgotPasswordScreen.sendResetLink')}
                        accessibilityRole="button"
                    >
                        {isLoading ? (
                            <ActivityIndicator color="#000" />
                        ) : (
                            <ScaledText style={styles.submitButtonText}>{t('forgotPasswordScreen.sendResetLink')}</ScaledText>
                        )}
                    </TouchableOpacity>
                </ScrollView>
            </LinearGradient>
        </KeyboardAvoidingView>
    );
};

export default ForgotPasswordScreen;

const styles = StyleSheet.create({
    flex1: { flex: 1 },
    container: {
        flex: 1,
        justifyContent: 'center',
        paddingHorizontal: 24,
        width: '100%',
        maxWidth: 400,
        alignSelf: 'center',
    },
    backArrow: {
        position: 'absolute',
        top: 60,
        left: 24,
        padding: 8,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        color: '#FFF',
        textAlign: 'center',
        marginBottom: 12,
        letterSpacing: 0.5,
    },
    subtitle: {
        fontSize: 15,
        color: 'rgba(255,255,255,0.6)',
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 32,
    },
    inputContainer: {
        marginBottom: 20,
    },
    input: {
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.1)',
        paddingVertical: 16,
        paddingHorizontal: 20,
        borderRadius: 16,
        fontSize: 16,
        color: '#FFF',
    },
    inputError: {
        borderColor: '#EF4444',
    },
    errorHint: {
        color: '#EF4444',
        fontSize: 12,
        marginTop: 6,
        marginLeft: 6,
    },
    submitButton: {
        backgroundColor: '#FFF',
        paddingVertical: 16,
        borderRadius: 16,
        alignItems: 'center',
    },
    submitButtonDisabled: {
        opacity: 0.5,
    },
    submitButtonText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#000',
    },
    // Success state
    successContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 32,
    },
    successIcon: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
    },
    successTitle: {
        fontSize: 24,
        fontWeight: '800',
        color: '#FFF',
        marginBottom: 12,
    },
    successMessage: {
        fontSize: 15,
        color: 'rgba(255,255,255,0.7)',
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 8,
    },
    emailHighlight: {
        color: '#FFD700',
        fontWeight: '600',
    },
    successHint: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.4)',
        textAlign: 'center',
        marginBottom: 32,
    },
    backButton: {
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.2)',
        paddingVertical: 14,
        paddingHorizontal: 32,
        borderRadius: 16,
    },
    backButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: '#FFF',
    },
});
