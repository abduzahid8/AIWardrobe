import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    Alert,
    StyleSheet,
    KeyboardAvoidingView,
    Platform,
    TouchableWithoutFeedback,
    Keyboard,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import { supabase } from '../lib/supabase';
import { useTranslation } from 'react-i18next';

import useAuthStore from '../store/auth';

/**
 * ResetPasswordScreen — deep link target for password reset.
 * User arrives here via the email link from ForgotPasswordScreen.
 * Allows setting a new password with validation.
 */
const ResetPasswordScreen = () => {
    const navigation = useAppNavigation();
    const { t } = useTranslation();
    const { session, isAuthenticated } = useAuthStore();
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // If no session is found, the user might have arrived here manually
    // or the deep link token might have expired.
    if (!session || !isAuthenticated) {
        return (
            <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.container}>
                <View style={styles.formContainer}>
                    <View style={styles.iconContainer}>
                        <Ionicons name="alert-circle-outline" size={40} color="#EF4444" />
                    </View>
                    <Text style={styles.title}>{t('resetPassword.invalidSession')}</Text>
                    <Text style={styles.subtitle}>
                        {t('resetPassword.resetLinkExpired')}
                    </Text>
                    <TouchableOpacity
                        onPress={() => navigation.navigate('SignIn')}
                        style={styles.submitButton}
                    >
                        <Text style={styles.submitButtonText}>{t('resetPassword.backToSignIn')}</Text>
                    </TouchableOpacity>
                </View>
            </LinearGradient>
        );
    }

    // Password validation
    const passwordChecks = {
        minLength: password.length >= 8,
        hasLowercase: /[a-z]/.test(password),
        hasUppercase: /[A-Z]/.test(password),
        hasNumber: /[0-9]/.test(password),
    };
    const isPasswordValid = Object.values(passwordChecks).every(Boolean);
    const passwordsMatch = password === confirmPassword && confirmPassword.length > 0;
    const canSubmit = isPasswordValid && passwordsMatch;

    const handleResetPassword = async () => {
        if (!canSubmit || isLoading) return;

        setIsLoading(true);
        try {
            const { error } = await supabase.auth.updateUser({ password });

            if (error) throw error;

            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            Alert.alert(
                t('resetPassword.passwordUpdated'),
                t('resetPassword.passwordResetSuccess'),
                [
                    {
                        text: t('auth.signIn'),
                        onPress: () => {
                            navigation.reset({
                                index: 0,
                                routes: [{ name: 'SignIn' }],
                            });
                        },
                    },
                ],
            );
        } catch (error: any) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            const msg = error.message || t('resetPassword.failedResetPassword');

            if (msg.includes('same_password') || msg.includes('different from your old')) {
                Alert.alert(t('resetPassword.samePassword'), t('resetPassword.newPasswordDifferent'));
            } else if (msg.includes('weak_password') || msg.includes('too weak')) {
                Alert.alert(t('resetPassword.weakPassword'), t('resetPassword.chooseStronger'));
            } else {
                Alert.alert(t('resetPassword.error'), msg);
            }
        } finally {
            setIsLoading(false);
        }
    };

    const CheckItem = ({ valid, text }: { valid: boolean; text: string }) => (
        <View style={styles.checkRow}>
            <Ionicons
                name={valid ? 'checkmark-circle' : 'ellipse-outline'}
                size={16}
                color={valid ? '#FFD700' : 'rgba(255,255,255,0.3)'}
            />
            <Text style={[styles.checkText, valid && styles.checkTextValid]}>{text}</Text>
        </View>
    );

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
                <LinearGradient
                    colors={['#0A0A0A', '#1A1C29', '#16213E']}
                    style={styles.gradient}
                >
                    <View style={styles.formContainer}>
                        {/* Back Button */}
                        <TouchableOpacity
                            onPress={() => navigation.goBack()}
                            style={styles.backButton}
                            accessibilityLabel={t('resetPassword.goBack')}
                            accessibilityRole="button"
                        >
                            <Ionicons name="arrow-back" size={24} color="#FFF" />
                        </TouchableOpacity>

                        <View style={styles.iconContainer}>
                            <Ionicons name="lock-closed-outline" size={40} color="#FFD700" />
                        </View>

                        <Text style={styles.title} accessibilityRole="header">
                            {t('resetPassword.title')}
                        </Text>
                        <Text style={styles.subtitle}>
                            {t('resetPassword.subtitle')}
                        </Text>

                        {/* New Password */}
                        <View style={styles.inputContainer}>
                            <TextInput
                                style={styles.input}
                                value={password}
                                onChangeText={setPassword}
                                placeholder={t('resetPassword.newPassword')}
                                placeholderTextColor="rgba(255,255,255,0.4)"
                                secureTextEntry
                                accessibilityLabel={t('resetPassword.newPassword')}
                                maxLength={128}
                            />
                        </View>

                        {/* Password Requirements */}
                        {password.length > 0 && (
                            <View style={styles.checksContainer}>
                                <CheckItem valid={passwordChecks.minLength} text={t('resetPassword.minLength')} />
                                <CheckItem valid={passwordChecks.hasLowercase} text={t('resetPassword.lowercase')} />
                                <CheckItem valid={passwordChecks.hasUppercase} text={t('resetPassword.uppercase')} />
                                <CheckItem valid={passwordChecks.hasNumber} text={t('resetPassword.number')} />
                            </View>
                        )}

                        {/* Confirm Password */}
                        <View style={styles.inputContainer}>
                            <TextInput
                                style={[
                                    styles.input,
                                    confirmPassword.length > 0 && !passwordsMatch && styles.inputError,
                                ]}
                                value={confirmPassword}
                                onChangeText={setConfirmPassword}
                                placeholder={t('resetPassword.confirmPassword')}
                                placeholderTextColor="rgba(255,255,255,0.4)"
                                secureTextEntry
                                accessibilityLabel={t('resetPassword.confirmPassword')}
                                maxLength={128}
                            />
                            {confirmPassword.length > 0 && !passwordsMatch && (
                                <Text style={styles.errorHint}>{t('resetPassword.passwordsDontMatch')}</Text>
                            )}
                        </View>

                        {/* Submit */}
                        <TouchableOpacity
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                handleResetPassword();
                            }}
                            style={[styles.submitButton, !canSubmit && styles.submitButtonDisabled]}
                            disabled={isLoading || !canSubmit}
                            activeOpacity={0.8}
                            accessibilityLabel={isLoading ? t('resetPassword.resettingPassword') : t('resetPassword.resetPassword')}
                            accessibilityRole="button"
                        >
                            <Text style={styles.submitButtonText}>
                                {isLoading ? t('resetPassword.resetting') : t('resetPassword.resetPassword')}
                            </Text>
                        </TouchableOpacity>
                    </View>
                </LinearGradient>
            </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
    );
};

export default ResetPasswordScreen;

const styles = StyleSheet.create({
    container: { flex: 1 },
    gradient: { flex: 1, justifyContent: 'center' },
    formContainer: {
        paddingHorizontal: 24,
        width: '100%',
        maxWidth: 400,
        alignSelf: 'center',
    },
    backButton: {
        position: 'absolute',
        top: -60,
        left: 24,
    },
    iconContainer: {
        width: 72,
        height: 72,
        borderRadius: 36,
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
        alignItems: 'center',
        justifyContent: 'center',
        alignSelf: 'center',
        marginBottom: 20,
    },
    title: {
        fontSize: 24,
        fontWeight: '800',
        color: '#FFF',
        textAlign: 'center',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 15,
        color: 'rgba(255,255,255,0.5)',
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 28,
    },
    inputContainer: { marginBottom: 16 },
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
    checksContainer: {
        backgroundColor: 'rgba(255, 255, 255, 0.03)',
        padding: 12,
        borderRadius: 12,
        marginBottom: 16,
    },
    checkRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 6,
    },
    checkText: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.4)',
        marginLeft: 8,
    },
    checkTextValid: {
        color: 'rgba(255,255,255,0.9)',
    },
    submitButton: {
        backgroundColor: '#FFF',
        paddingVertical: 16,
        borderRadius: 16,
        alignItems: 'center',
        marginTop: 8,
    },
    submitButtonDisabled: {
        opacity: 0.5,
    },
    submitButtonText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#000',
    },
});
