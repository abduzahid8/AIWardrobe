import React, { useEffect, useRef, useState } from 'react';
import { View, TextInput, TouchableOpacity, Alert, StyleSheet, KeyboardAvoidingView, Platform, ScrollView } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context';
import { ScaledText } from '../components/ui/ScaledText';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';

import useAuthStore from '../store/auth';
import { authEvents } from '../store/authEvents';

/**
 * ResetPasswordScreen — deep link target for password reset.
 * User arrives here via the email link from ForgotPasswordScreen.
 * Allows setting a new password with validation.
 */
const ResetPasswordScreen = () => {
    const navigation = useAppNavigation();
    const { t } = useTranslation();
    const { session, isAuthenticated, passwordRecovery } = useAuthStore();
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const recoveryOwner = useRef<string | null>(null);
    const inFlight = useRef(false);
    const mounted = useRef(true);
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; };
    }, []);
    const validRecovery = passwordRecovery?.ownerId === session?.user.id
        && passwordRecovery?.revision === authEvents.getRevision();
    if (!recoveryOwner.current && session?.user.id && isAuthenticated && validRecovery) recoveryOwner.current = session.user.id;

    // If no session is found, the user might have arrived here manually
    // or the deep link token might have expired.
    if (!session || !isAuthenticated || !validRecovery || session.user.id !== recoveryOwner.current) {
        return (
            <LinearGradient colors={['#0A0A0A', '#1A1C29', '#16213E']} style={styles.container}>
                <View style={styles.formContainer}>
                    <View style={styles.iconContainer}>
                        <Ionicons name="alert-circle-outline" size={40} color="#EF4444" />
                    </View>
                    <ScaledText style={styles.title}>{t('resetPassword.invalidSession')}</ScaledText>
                    <ScaledText style={styles.subtitle}>
                        {t('resetPassword.resetLinkExpired')}
                    </ScaledText>
                    <TouchableOpacity
                        onPress={() => navigation.navigate('SignIn')}
                        style={styles.submitButton}
                    >
                        <ScaledText style={styles.submitButtonText}>{t('resetPassword.backToSignIn')}</ScaledText>
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
        if (!canSubmit || inFlight.current) return;
        const owner = recoveryOwner.current!;
        const revision = authEvents.getRevision();
        const sameOwner = () => mounted.current && authEvents.getRevision() === revision
            && useAuthStore.getState().isAuthenticated
            && useAuthStore.getState().session?.user.id === owner
            && useAuthStore.getState().passwordRecovery?.ownerId === owner
            && useAuthStore.getState().passwordRecovery?.revision === revision;

        inFlight.current = true;
        setIsLoading(true);
        try {
            await useAuthStore.getState().updatePassword(password, owner);
            if (!sameOwner()) return;

            void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => undefined);
            Alert.alert(
                t('resetPassword.passwordUpdated'),
                t('resetPassword.passwordResetSuccess'),
                [
                    {
                        text: t('auth.signIn'),
                        onPress: async () => {
                            if (!sameOwner()) return;
                            try {
                                await useAuthStore.getState().logout(owner);
                                if (mounted.current && !useAuthStore.getState().isAuthenticated) {
                                    navigation.reset({ index: 0, routes: [{ name: 'SignIn' }] });
                                }
                            } catch {
                                if (sameOwner()) Alert.alert(t('resetPassword.error'), t('resetPassword.failedResetPassword'));
                            }
                        },
                    },
                ],
            );
        } catch (error: any) {
            if (!sameOwner()) return;
            void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => undefined);
            const msg = error.message || t('resetPassword.failedResetPassword');

            if (msg.includes('same_password') || msg.includes('different from your old')) {
                Alert.alert(t('resetPassword.samePassword'), t('resetPassword.newPasswordDifferent'));
            } else if (msg.includes('weak_password') || msg.includes('too weak')) {
                Alert.alert(t('resetPassword.weakPassword'), t('resetPassword.chooseStronger'));
            } else {
                Alert.alert(t('resetPassword.error'), msg);
            }
        } finally {
            inFlight.current = false;
            if (mounted.current) setIsLoading(false);
        }
    };

    const CheckItem = ({ valid, text }: { valid: boolean; text: string }) => (
        <View style={styles.checkRow}>
            <Ionicons
                name={valid ? 'checkmark-circle' : 'ellipse-outline'}
                size={16}
                color={valid ? '#FFD700' : 'rgba(255,255,255,0.3)'}
            />
            <ScaledText style={[styles.checkText, valid && styles.checkTextValid]}>{text}</ScaledText>
        </View>
    );

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            <LinearGradient
                colors={['#0A0A0A', '#1A1C29', '#16213E']}
                style={styles.gradient}
            >
                {/* SafeArea keeps the header clear of the status bar — without
                    it the back chevron sat under the clock where the system
                    status bar swallows all touches (looked fine, never fired). */}
                <SafeAreaView style={styles.safeArea} edges={['top']}>
                <ScrollView
                    contentContainerStyle={styles.formContainer}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                    accessible={false}
                >
                    {/* Back Button — in normal flow (the old absolute
                        top:-60 placement rendered under the status bar area
                        and was invisible/untappable, leaving gesture-only
                        exit with no affordance). */}
                    <View style={styles.headerRow}>
                        <TouchableOpacity
                            // Deep links can land here with no history behind
                            // (linking replaces root state) — plain goBack()
                            // would no-op and strand the user. Fall back to
                            // the main tabs in that case.
                            onPress={() => {
                                if (navigation.canGoBack()) navigation.goBack();
                                else navigation.reset({ index: 0, routes: [{ name: 'Main' }] });
                            }}
                            style={styles.backButton}
                            testID="resetBackButton"
                            accessibilityLabel={t('resetPassword.goBack')}
                            accessibilityRole="button"
                            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                        >
                            <Ionicons name="chevron-back" size={26} color="#FFF" />
                        </TouchableOpacity>
                    </View>

                    <View style={styles.iconContainer}>
                        <Ionicons name="lock-closed-outline" size={40} color="#FFD700" />
                    </View>

                    <ScaledText style={styles.title} accessibilityRole="header">
                        {t('resetPassword.title')}
                    </ScaledText>
                    <ScaledText style={styles.subtitle}>
                        {t('resetPassword.subtitle')}
                    </ScaledText>

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
                            <ScaledText style={styles.errorHint}>{t('resetPassword.passwordsDontMatch')}</ScaledText>
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
                        <ScaledText style={styles.submitButtonText}>
                            {isLoading ? t('resetPassword.resetting') : t('resetPassword.resetPassword')}
                        </ScaledText>
                    </TouchableOpacity>
                </ScrollView>
                </SafeAreaView>
            </LinearGradient>
        </KeyboardAvoidingView>
    );
};

export default ResetPasswordScreen;

const styles = StyleSheet.create({
    container: { flex: 1 },
    gradient: { flex: 1, justifyContent: 'center' },
    safeArea: { flex: 1 },
    formContainer: {
        paddingHorizontal: 24,
        width: '100%',
        maxWidth: 400,
        alignSelf: 'center',
    },
    headerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 12,
    },
    backButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.08)',
        alignItems: 'center',
        justifyContent: 'center',
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
