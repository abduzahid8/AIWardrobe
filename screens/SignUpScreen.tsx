import { AccountNameError } from '../src/services/accountNames';
import {
  Alert,
  Platform,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  View,
  KeyboardAvoidingView,
  ScrollView,
} from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import React, { useState, useMemo, useEffect, useRef } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import { SafeAreaView } from "react-native-safe-area-context";
import * as Haptics from "expo-haptics";
import * as AppleAuthentication from "expo-apple-authentication";
import { Ionicons } from "@expo/vector-icons";
import Animated, { FadeInDown } from 'react-native-reanimated';
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";
import { createLogger } from "../src/utils/logger";
import { SUPABASE_AUTH_ERRORS } from "../constants/authErrors";
import { useAccessibility } from '../hooks/useAccessibility';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { colors } = LiquidGlass2026Theme;

const logger = createLogger('SignUp');

const SignUpScreen = () => {
  const { t } = useTranslation();
  const { isReducedMotionEnabled } = useAccessibility();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [username, setUserName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [isSent, setIsSent] = useState(false);
  const [sentEmail, setSentEmail] = useState("");
  const navigation = useNavigation();
  const { register, signInWithApple, signInWithGoogle } = useAuthStore();
  const [appleAuthAvailable, setAppleAuthAvailable] = useState(false);
  const authRequestInFlight = useRef(false);

  const entering = (delay: number) =>
    isReducedMotionEnabled ? undefined : FadeInDown.delay(delay).duration(400);

  useEffect(() => {
    if (Platform.OS === 'ios') {
      AppleAuthentication.isAvailableAsync().then(setAppleAuthAvailable).catch(() => {});
    }
  }, []);

  const handleAppleSignUp = async () => {
    if (authRequestInFlight.current || isLoading || isGoogleLoading) return;
    authRequestInFlight.current = true;
    setIsLoading(true);
    try {
      await signInWithApple();
    } catch (error: any) {
      if (error?.code !== 'ERR_REQUEST_CANCELED') {
        Alert.alert(t('signUp.registrationFailed'), error.message || t('signIn.appleSignInFailed'));
      }
    } finally {
      authRequestInFlight.current = false;
      setIsLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    if (authRequestInFlight.current || isLoading || isGoogleLoading) return;
    authRequestInFlight.current = true;
    setIsGoogleLoading(true);
    try {
      await signInWithGoogle();
    } catch (error: any) {
      if (error?.message && !error?.message?.includes('cancel')) {
        Alert.alert(t('signUp.registrationFailed'), error.message || t('signIn.googleSignInFailed'));
      }
    } finally {
      authRequestInFlight.current = false;
      setIsGoogleLoading(false);
    }
  };

  // Password validation checks
  const passwordChecks = useMemo(() => ({
    minLength: password.length >= 8,
    hasLowercase: /[a-z]/.test(password),
    hasUppercase: /[A-Z]/.test(password),
    hasNumber: /[0-9]/.test(password),
  }), [password]);

  const isPasswordValid = Object.values(passwordChecks).every(Boolean);

  // Email validation (trimmed + lowercased — users paste with spaces/caps)
  const normalizedEmail = email.trim().toLowerCase();
  const isEmailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail);

  // Username validation (3-30 chars, letters, numbers, underscores only)
  const trimmedUsername = username.trim();
  const isUsernameValid = /^[a-zA-Z0-9_]{3,30}$/.test(trimmedUsername);

  // Can submit form?
  const canSubmit = isEmailValid && isPasswordValid && isUsernameValid;

  const handleSignUp = async () => {
    if (authRequestInFlight.current) return;
    if (!email || !password || !username) {
      Alert.alert(t('signUp.missingFields'), t('signUp.fillRequiredFields'));
      return;
    }

    if (!isEmailValid) {
      Alert.alert(t('signUp.invalidEmail'), t('signUp.enterValidEmail'));
      return;
    }

    if (!isPasswordValid) {
      Alert.alert(
        t('signUp.passwordRequirements'),
        t('signUp.passwordRequirementsText')
      );
      return;
    }

    if (!isUsernameValid) {
      Alert.alert(
        t('signUp.invalidUsername'),
        t('signUp.usernameRequirements')
      );
      return;
    }

    authRequestInFlight.current = true;
    setIsLoading(true);
    try {
      await register(normalizedEmail, password, trimmedUsername);
    } catch (error: any) {
      const errorMessage = error.message || t('signUp.registrationFailed');
      logger.error('Signup error', errorMessage);

      // Email-confirmation required is a SUCCESS state (account created,
      // user must click the email link) — show check-email UI, not an error.
      if (errorMessage.includes(SUPABASE_AUTH_ERRORS.EMAIL_CONFIRMATION_REQUIRED)) {
        setSentEmail(normalizedEmail);
        setIsSent(true);
        return;
      }
      if (error instanceof AccountNameError) {
        Alert.alert(
          t(error.code === 'username_exists' ? 'signUp.usernameTaken' : error.code === 'invalid_handle' ? 'signUp.invalidUsername' : 'signUp.connectionFailed'),
          t(error.code === 'username_exists' ? 'signUp.usernameAlreadyUse' : error.code === 'invalid_handle' ? 'signUp.usernameRequirements' : 'signUp.cannotConnectServer'),
        );
        return;
      }
      // Parse specific error messages
      if (errorMessage.includes(SUPABASE_AUTH_ERRORS.USER_ALREADY_REGISTERED) || errorMessage.includes("already registered")) {
        Alert.alert(t('signUp.emailTaken'), t('signUp.emailAlreadyRegistered'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.USERNAME_EXISTS)) {
        Alert.alert(t('signUp.usernameTaken'), t('signUp.usernameAlreadyUse'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.DATABASE_ERROR) && errorMessage.includes("username")) {
        Alert.alert(t('signUp.usernameTaken'), t('signUp.usernameAlreadyUse'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.NETWORK_FAILED)) {
        Alert.alert(t('signUp.connectionFailed'), t('signUp.cannotConnectServer'));
      } else {
        Alert.alert(t('signUp.registrationFailed'), errorMessage);
      }
    } finally {
      authRequestInFlight.current = false;
      setIsLoading(false);
    }
  };

  const busy = isLoading || isGoogleLoading;

  const PasswordCheckItem = ({ isValid, text }: { isValid: boolean; text: string }) => (
    <View style={styles.checkItem}>
      <Ionicons
        name={isValid ? "checkmark-circle" : "ellipse-outline"}
        size={16}
        color={isValid ? "#15803D" : "rgba(10,25,49,0.25)"}
      />
      <ScaledText style={[styles.checkText, isValid && styles.checkTextValid]}>{text}</ScaledText>
    </View>
  );

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
        style={StyleSheet.absoluteFill}
        pointerEvents="none"
      />
      <View pointerEvents="none" style={styles.backgroundOrbTop} />
      <View pointerEvents="none" style={styles.backgroundOrbBottom} />

      <SafeAreaView style={styles.safeArea} edges={['top', 'bottom']}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.flex}
        >
          {isSent ? (
            <ScrollView
              contentContainerStyle={styles.scrollContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.formContainer} testID="signUpCheckEmail">
                <Animated.View entering={entering(0)} style={styles.successIcon}>
                  <Ionicons name="mail-outline" size={40} color="#0A1931" />
                </Animated.View>
                <Animated.View entering={entering(80)}>
                  <ScaledText style={styles.successTitle}>
                    {t('signUp.checkYourEmail', { defaultValue: 'Check your email' })}
                  </ScaledText>
                  <ScaledText style={styles.successMessage}>
                    {t('signUp.confirmationLinkSent', { defaultValue: 'We sent a confirmation link to' })}{'\n'}
                    <ScaledText style={styles.emailHighlight}>{sentEmail}</ScaledText>
                  </ScaledText>
                  <TouchableOpacity
                    onPress={() => (navigation as any).navigate('SignIn')}
                    style={styles.primaryButton}
                    testID="signUpBackToSignIn"
                    accessibilityRole="button"
                    activeOpacity={0.85}
                  >
                    <ScaledText style={styles.primaryButtonText}>{t('auth.signIn')}</ScaledText>
                  </TouchableOpacity>
                </Animated.View>
              </View>
            </ScrollView>
          ) : (
            <ScrollView
              contentContainerStyle={styles.scrollContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.formContainer} testID="formContainer">
                <Animated.View entering={entering(0)}>
                  <ScaledText
                    style={styles.title}
                    testID="signUpTitle"
                    accessibilityRole="header"
                  >
                    {t('auth.signUp.signUp')}
                  </ScaledText>
                </Animated.View>

                {/* ── Social sign-up ─────────────────────────── */}
                <Animated.View entering={entering(80)}>
                  {appleAuthAvailable && (
                    <AppleAuthentication.AppleAuthenticationButton
                      buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_UP}
                      buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.BLACK}
                      cornerRadius={16}
                      style={styles.socialButton}
                      onPress={handleAppleSignUp}
                    />
                  )}

                  <TouchableOpacity
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                      handleGoogleSignUp();
                    }}
                    style={[styles.googleButton, busy && styles.buttonBusy]}
                    disabled={busy}
                    activeOpacity={0.85}
                    testID="googleSignUpButton"
                    accessibilityLabel={t('signUp.signUpWithGoogle')}
                    accessibilityRole="button"
                    accessible={true}
                  >
                    {isGoogleLoading ? (
                      <ScaledText style={styles.googleButtonText}>
                        {t('signUp.creatingAccount')}
                      </ScaledText>
                    ) : (
                      <>
                        <View style={styles.googleBadge}>
                          <ScaledText style={styles.googleBadgeText}>G</ScaledText>
                        </View>
                        <ScaledText style={styles.googleButtonText}>
                          {t('signUp.signUpWithGoogle')}
                        </ScaledText>
                      </>
                    )}
                  </TouchableOpacity>
                </Animated.View>

                <Animated.View entering={entering(140)} style={styles.divider}>
                  <View style={styles.dividerLine} />
                  <ScaledText style={styles.dividerText}>{t('signIn.or')}</ScaledText>
                  <View style={styles.dividerLine} />
                </Animated.View>

                {/* ── Email ─────────────────────────── */}
                <Animated.View entering={entering(180)}>
                  <View style={[styles.inputCard, email.length > 0 && !isEmailValid && styles.inputCardError]} testID="emailField">
                    <Ionicons
                      name="mail-outline"
                      size={20}
                      color={email.length > 0 && !isEmailValid ? '#B91C1C' : colors.text.tertiary}
                      style={styles.inputIcon}
                    />
                    <TextInput
                      style={styles.input}
                      value={email}
                      onChangeText={setEmail}
                      placeholder={t('auth.email')}
                      placeholderTextColor="rgba(10,25,49,0.4)"
                      keyboardType="email-address"
                      autoCapitalize="none"
                      autoCorrect={false}
                      returnKeyType="next"
                      testID="signUpEmailInput"
                      accessibilityLabel={t('auth.email')}
                      maxLength={255}
                      editable={!busy}
                    />
                    {email.length > 0 && (
                      <TouchableOpacity
                        onPress={() => setEmail('')}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        accessibilityLabel={t('common.close', { defaultValue: 'Clear' })}
                        accessibilityRole="button"
                      >
                        <Ionicons name="close-circle" size={20} color="rgba(10,25,49,0.3)" />
                      </TouchableOpacity>
                    )}
                  </View>
                  {email.length > 0 && !isEmailValid && (
                    <ScaledText style={styles.errorHint}>
                      {t('auth.signUp.pleaseEnterValidEmail', { defaultValue: 'Please enter a valid email' })}
                    </ScaledText>
                  )}
                </Animated.View>

                {/* ── Password + checklist ─────────────────────────── */}
                <Animated.View entering={entering(210)}>
                  <View style={styles.inputCard} testID="passwordField">
                    <Ionicons name="lock-closed-outline" size={20} color={colors.text.tertiary} style={styles.inputIcon} />
                    <TextInput
                      style={styles.input}
                      value={password}
                      onChangeText={setPassword}
                      placeholder={t('auth.password')}
                      placeholderTextColor="rgba(10,25,49,0.4)"
                      secureTextEntry={!showPassword}
                      returnKeyType="next"
                      testID="signUpPasswordInput"
                      accessibilityLabel={t('auth.password')}
                      maxLength={128}
                      editable={!busy}
                    />
                    <TouchableOpacity
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setShowPassword((v) => !v);
                      }}
                      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                      accessibilityLabel={
                        showPassword
                          ? t('signIn.hidePassword', { defaultValue: 'Hide password' })
                          : t('signIn.showPassword', { defaultValue: 'Show password' })
                      }
                      accessibilityRole="button"
                    >
                      <Ionicons
                        name={showPassword ? 'eye-off-outline' : 'eye-outline'}
                        size={20}
                        color="rgba(10,25,49,0.45)"
                      />
                    </TouchableOpacity>
                  </View>

                  {password.length > 0 && (
                    <View style={styles.passwordChecks}>
                      <ScaledText style={styles.checklistTitle}>{t('auth.signUp.passwordRequirements')}</ScaledText>
                      <PasswordCheckItem isValid={passwordChecks.minLength} text={t('resetPassword.minLength')} />
                      <PasswordCheckItem isValid={passwordChecks.hasLowercase} text={t('resetPassword.lowercase')} />
                      <PasswordCheckItem isValid={passwordChecks.hasUppercase} text={t('resetPassword.uppercase')} />
                      <PasswordCheckItem isValid={passwordChecks.hasNumber} text={t('resetPassword.number')} />
                    </View>
                  )}
                </Animated.View>

                {/* ── Username ─────────────────────────── */}
                <Animated.View entering={entering(240)}>
                  <View style={[styles.inputCard, username.length > 0 && !isUsernameValid && styles.inputCardError]}>
                    <Ionicons
                      name="person-outline"
                      size={20}
                      color={username.length > 0 && !isUsernameValid ? '#B91C1C' : colors.text.tertiary}
                      style={styles.inputIcon}
                    />
                    <TextInput
                      style={styles.input}
                      value={username}
                      onChangeText={setUserName}
                      placeholder={t('auth.username')}
                      placeholderTextColor="rgba(10,25,49,0.4)"
                      autoCapitalize="none"
                      autoCorrect={false}
                      returnKeyType="done"
                      onSubmitEditing={handleSignUp}
                      testID="signUpUsernameInput"
                      accessibilityLabel={t('auth.username')}
                      maxLength={30}
                      editable={!busy}
                    />
                    {username.length > 0 && (
                      <TouchableOpacity
                        onPress={() => setUserName('')}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        accessibilityLabel={t('common.close', { defaultValue: 'Clear' })}
                        accessibilityRole="button"
                      >
                        <Ionicons name="close-circle" size={20} color="rgba(10,25,49,0.3)" />
                      </TouchableOpacity>
                    )}
                  </View>
                  {username.length > 0 && !isUsernameValid && (
                    <ScaledText style={styles.errorHint}>{t('signUp.usernameRequirements')}</ScaledText>
                  )}
                </Animated.View>

                {/* ── Submit ─────────────────────────── */}
                <Animated.View entering={entering(260)}>
                  <TouchableOpacity
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                      handleSignUp();
                    }}
                    style={[styles.primaryButton, !canSubmit && !busy && styles.primaryButtonDim]}
                    disabled={busy}
                    activeOpacity={0.85}
                    testID="signUpSubmitButton"
                    accessibilityLabel={isLoading ? t('signUp.creatingAccount') : t('auth.signUp.signUp')}
                    accessibilityState={{ disabled: !canSubmit || busy }}
                    accessibilityRole="button"
                    accessible={true}
                  >
                    <ScaledText style={styles.primaryButtonText}>
                      {isLoading ? t('signUp.creatingAccount') : t("auth.signUp.signUp")}
                    </ScaledText>
                  </TouchableOpacity>
                </Animated.View>

                <Animated.View entering={entering(320)}>
                  <TouchableOpacity
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                      navigation.goBack();
                    }}
                    style={styles.linkButton}
                    testID="signInLink"
                    accessibilityRole="button"
                    disabled={busy}
                  >
                    <ScaledText style={styles.linkText}>
                      <ScaledText style={styles.linkTextMuted}>{t("auth.haveAccount")} </ScaledText>
                      {t("auth.signIn")}
                    </ScaledText>
                  </TouchableOpacity>
                </Animated.View>
              </View>
            </ScrollView>
          )}
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

export default SignUpScreen;

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  backgroundOrbTop: {
    position: 'absolute',
    top: -100,
    right: -80,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: 'rgba(188, 210, 245, 0.42)',
  },
  backgroundOrbBottom: {
    position: 'absolute',
    left: -120,
    bottom: 140,
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: 'rgba(216, 229, 252, 0.34)',
  },
  safeArea: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingVertical: 32,
  },
  formContainer: {
    paddingHorizontal: 24,
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#0A1931',
    textAlign: 'center',
    letterSpacing: 0.5,
    marginBottom: 28,
  },
  socialButton: {
    width: '100%',
    height: 56,
    marginBottom: 16,
  },
  googleButton: {
    width: '100%',
    minHeight: 56,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0A1931',
    borderRadius: 16,
    gap: 12,
    paddingVertical: 16,
    paddingHorizontal: 20,
    marginBottom: 4,
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.18,
    shadowRadius: 16,
    elevation: 6,
  },
  buttonBusy: {
    opacity: 0.7,
  },
  googleBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  googleBadgeText: {
    fontSize: 17,
    fontWeight: '800',
    color: '#0A1931',
    lineHeight: 20,
  },
  googleButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
    letterSpacing: 0.2,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 22,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(10,25,49,0.1)',
  },
  dividerText: {
    color: '#808080',
    fontSize: 13,
    marginHorizontal: 12,
  },
  inputCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderWidth: 1,
    borderColor: 'rgba(10,25,49,0.1)',
    borderRadius: 16,
    paddingVertical: 4,
    paddingHorizontal: 16,
    minHeight: 56,
    marginBottom: 12,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.04,
    shadowRadius: 12,
    elevation: 2,
  },
  inputCardError: {
    borderColor: '#B91C1C',
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: '#0A1931',
    paddingVertical: 14,
  },
  errorHint: {
    color: '#B91C1C',
    fontSize: 12,
    marginTop: -4,
    marginBottom: 8,
    marginLeft: 6,
  },
  passwordChecks: {
    marginTop: -2,
    marginBottom: 12,
    marginLeft: 2,
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: 'rgba(10,25,49,0.04)',
    borderWidth: 1,
    borderColor: 'rgba(10,25,49,0.08)',
    borderRadius: 12,
  },
  checklistTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: 'rgba(10,25,49,0.6)',
    marginBottom: 8,
  },
  checkItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  checkText: {
    fontSize: 13,
    color: 'rgba(10,25,49,0.45)',
    marginLeft: 8,
  },
  checkTextValid: {
    color: '#0A1931',
  },
  primaryButton: {
    backgroundColor: '#0A1931',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
    marginBottom: 12,
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 6,
    minHeight: 56,
  },
  primaryButtonDim: {
    opacity: 0.45,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFFFFF',
    letterSpacing: 0.2,
  },
  linkButton: {
    alignItems: 'center',
    marginTop: 12,
    marginBottom: 16,
    paddingVertical: 8,
  },
  linkText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#0A1931',
  },
  linkTextMuted: {
    color: '#808080',
    fontWeight: '400',
  },
  successIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(10,25,49,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(10,25,49,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
    alignSelf: 'center',
  },
  successTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: '#0A1931',
    marginBottom: 12,
    textAlign: 'center',
  },
  successMessage: {
    fontSize: 15,
    color: '#4D4D4D',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 32,
  },
  emailHighlight: {
    color: '#0A1931',
    fontWeight: '700',
  },
});
