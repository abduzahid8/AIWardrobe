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
import React, { useState, useEffect, useRef } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import { SafeAreaView } from "react-native-safe-area-context";
import * as Haptics from "expo-haptics";
import * as AppleAuthentication from "expo-apple-authentication";
import { Ionicons } from "@expo/vector-icons";
import Animated, { FadeInDown } from 'react-native-reanimated';
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";
import { SUPABASE_AUTH_ERRORS } from "../constants/authErrors";
import { useAccessibility } from '../hooks/useAccessibility';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { colors, radius } = LiquidGlass2026Theme;

type SignInStep = 'email' | 'password';

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const SignInScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const { isReducedMotionEnabled } = useAccessibility();

  const [step, setStep] = useState<SignInStep>('email');
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const { login, signInWithApple, signInWithGoogle } = useAuthStore();
  const [appleAuthAvailable, setAppleAuthAvailable] = useState(false);
  const authRequestInFlight = useRef(false);
  const passwordRef = useRef<TextInput>(null);

  const normalizedEmail = email.trim().toLowerCase();
  const isEmailValid = EMAIL_RE.test(normalizedEmail);
  const isPasswordValid = password.length > 0;

  const entering = (delay: number) =>
    isReducedMotionEnabled ? undefined : FadeInDown.delay(delay).duration(400);

  useEffect(() => {
    if (Platform.OS === 'ios') {
      AppleAuthentication.isAvailableAsync().then(setAppleAuthAvailable).catch(() => {});
    }
  }, []);

  const handleAppleSignIn = async () => {
    if (authRequestInFlight.current || isLoading || isGoogleLoading) return;
    authRequestInFlight.current = true;
    setIsLoading(true);
    try {
      await signInWithApple();
    } catch (error: any) {
      if (error?.code !== 'ERR_REQUEST_CANCELED') {
        Alert.alert(t('signIn.loginFailed'), error.message || t('signIn.appleSignInFailed'));
      }
    } finally {
      authRequestInFlight.current = false;
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    if (authRequestInFlight.current || isLoading || isGoogleLoading) return;
    authRequestInFlight.current = true;
    setIsGoogleLoading(true);
    try {
      await signInWithGoogle();
    } catch (error: any) {
      if (error?.message && !error?.message?.includes('cancel')) {
        Alert.alert(t('signIn.loginFailed'), error.message || t('signIn.googleSignInFailed'));
      }
    } finally {
      authRequestInFlight.current = false;
      setIsGoogleLoading(false);
    }
  };

  const handleEmailContinue = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (!email.trim()) {
      Alert.alert(t('common.error'), t('signIn.emailRequired'));
      return;
    }
    if (!isEmailValid) {
      Alert.alert(
        t('signUp.invalidEmail', { defaultValue: 'Invalid Email' }),
        t('signUp.enterValidEmail', { defaultValue: 'Please enter a valid email address' }),
      );
      return;
    }
    setStep('password');
    // Focus password field once it mounts
    requestAnimationFrame(() => {
      setTimeout(() => passwordRef.current?.focus(), 100);
    });
  };

  const handleBackToEmail = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setStep('email');
    setPassword('');
    setShowPassword(false);
  };

  const handleSignIn = async () => {
    if (authRequestInFlight.current) return;
    if (!normalizedEmail || !password) {
      Alert.alert(t('common.error'), t('signIn.emailPasswordRequired'));
      return;
    }
    if (isLoading || isGoogleLoading) return;
    authRequestInFlight.current = true;
    setIsLoading(true);
    try {
      await login(normalizedEmail, password);
    } catch (error: any) {
      const errorMessage = error.message || t('signIn.loginFailed');
      if (errorMessage.includes(SUPABASE_AUTH_ERRORS.INVALID_CREDENTIALS) || errorMessage.includes("invalid claim")) {
        Alert.alert(t('signIn.loginFailed'), t('signIn.invalidEmailPassword'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.EMAIL_NOT_CONFIRMED)) {
        Alert.alert(t('signIn.emailNotVerified'), t('signIn.verifyEmailBeforeLogin'));
      } else if (
        errorMessage.toLowerCase().includes(SUPABASE_AUTH_ERRORS.RATE_LIMITED) ||
        errorMessage.toLowerCase().includes(SUPABASE_AUTH_ERRORS.TOO_MANY_REQUESTS)
      ) {
        Alert.alert(t('forgotPassword.tooManyRequests'), t('forgotPassword.waitFewMinutes'));
      } else {
        Alert.alert(t('signIn.loginFailed'), errorMessage);
      }
    } finally {
      authRequestInFlight.current = false;
      setIsLoading(false);
    }
  };

  const busy = isLoading || isGoogleLoading;

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
          <ScrollView
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.formContainer} testID="formContainer">
              <Animated.View entering={entering(0)}>
                <ScaledText
                  style={styles.title}
                  testID="signInTitle"
                  accessibilityLabel={t('auth.signIn')}
                  accessibilityRole="header"
                >
                  {t('auth.signIn')}
                </ScaledText>
              </Animated.View>

              {step === 'email' ? (
                <>
                  {/* ── Step 1: Google-first ─────────────────────────── */}
                  <Animated.View entering={entering(80)}>
                    {appleAuthAvailable && (
                      <AppleAuthentication.AppleAuthenticationButton
                        buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
                        buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.BLACK}
                        cornerRadius={16}
                        style={styles.socialButton}
                        onPress={handleAppleSignIn}
                      />
                    )}

                    <TouchableOpacity
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                        handleGoogleSignIn();
                      }}
                      style={[styles.googleButton, busy && styles.buttonBusy]}
                      disabled={busy}
                      activeOpacity={0.85}
                      testID="googleSignInButton"
                      accessibilityLabel={t('signIn.signInWithGoogle')}
                      accessibilityRole="button"
                      accessible={true}
                    >
                      {isGoogleLoading ? (
                        <ScaledText style={styles.googleButtonText}>
                          {t('signIn.signingIn')}
                        </ScaledText>
                      ) : (
                        <>
                          <View style={styles.googleBadge}>
                            <ScaledText style={styles.googleBadgeText}>G</ScaledText>
                          </View>
                          <ScaledText style={styles.googleButtonText}>
                            {t('signIn.signInWithGoogle')}
                          </ScaledText>
                        </>
                      )}
                    </TouchableOpacity>
                  </Animated.View>

                  <Animated.View entering={entering(200)} style={styles.emailSection}>
                    <View style={[styles.inputCard, email.length > 0 && !isEmailValid && styles.inputCardError]} testID="emailField">
                      <Ionicons
                        name="mail-outline"
                        size={20}
                        color={email.length > 0 && !isEmailValid ? '#B91C1C' : colors.text.tertiary}
                        style={styles.inputIcon}
                      />
                      <TextInput
                        key="sign-in-email"
                        style={styles.input}
                        value={email}
                        secureTextEntry={false}
                        onChangeText={setEmail}
                        placeholder={t('auth.email')}
                        placeholderTextColor="rgba(10,25,49,0.4)"
                        keyboardType="email-address"
                        autoCapitalize="none"
                        autoCorrect={false}
                        returnKeyType="next"
                        onSubmitEditing={handleEmailContinue}
                        testID="emailInputNative"
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

                    <TouchableOpacity
                      onPress={handleEmailContinue}
                      style={[styles.primaryButton, !isEmailValid && !busy && styles.primaryButtonDim]}
                      disabled={busy}
                      activeOpacity={0.85}
                      testID="continueEmailButton"
                      accessibilityLabel={t('signIn.continue', { defaultValue: 'Continue' })}
                      accessibilityRole="button"
                      accessible={true}
                    >
                      <ScaledText style={styles.primaryButtonText}>
                        {t('signIn.continue', { defaultValue: 'Continue' })}
                      </ScaledText>
                      <Ionicons name="arrow-forward" size={18} color="#FFFFFF" />
                    </TouchableOpacity>
                  </Animated.View>
                </>
              ) : (
                <>
                  {/* ── Step 2: password only ────────────────────────── */}
                  <Animated.View entering={entering(80)}>
                    <View style={styles.emailChip} testID="emailChip">
                      <View style={styles.emailChipAvatar}>
                        <ScaledText style={styles.emailChipAvatarText}>
                          {normalizedEmail.charAt(0).toUpperCase()}
                        </ScaledText>
                      </View>
                      <ScaledText style={styles.emailChipText} numberOfLines={1} ellipsizeMode="middle">
                        {normalizedEmail}
                      </ScaledText>
                      <TouchableOpacity
                        onPress={handleBackToEmail}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        testID="changeEmailButton"
                        accessibilityLabel={t('signIn.change', { defaultValue: 'Change email' })}
                        accessibilityRole="button"
                      >
                        <ScaledText style={styles.emailChipChange}>
                          {t('signIn.change', { defaultValue: 'Change' })}
                        </ScaledText>
                      </TouchableOpacity>
                    </View>

                    <View style={styles.inputCard} testID="passwordField">
                      <Ionicons name="lock-closed-outline" size={20} color={colors.text.tertiary} style={styles.inputIcon} />
                      <TextInput
                        key="sign-in-password"
                        ref={passwordRef}
                        style={styles.input}
                        value={password}
                        onChangeText={setPassword}
                        placeholder={t('auth.password')}
                        placeholderTextColor="rgba(10,25,49,0.4)"
                        secureTextEntry={!showPassword}
                        returnKeyType="go"
                        onSubmitEditing={handleSignIn}
                        testID="passwordInputNative"
                        accessibilityLabel={t('auth.password')}
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

                    <TouchableOpacity
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        (navigation.navigate as any)("ForgotPassword", { email: normalizedEmail });
                      }}
                      style={styles.forgotButton}
                      testID="forgotPasswordButton"
                      accessibilityLabel={t('auth.forgotPassword')}
                      accessibilityRole="button"
                      accessible={true}
                    >
                      <ScaledText style={styles.forgotText}>{t('auth.forgotPassword')}</ScaledText>
                    </TouchableOpacity>

                    <TouchableOpacity
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        handleSignIn();
                      }}
                      // Keep tappable when password is empty so empty-submit
                      // reaches handleSignIn's native-alert guard
                      // (common.error / signIn.emailPasswordRequired).
                      // isPasswordValid only dims the button visually.
                      style={[styles.primaryButton, !isPasswordValid && !busy && styles.primaryButtonDim]}
                      disabled={busy}
                      activeOpacity={0.85}
                      testID="signInButton"
                      accessibilityLabel={t('auth.signIn')}
                      accessibilityState={{ disabled: !isPasswordValid || busy }}
                      accessibilityRole="button"
                      accessible={true}
                    >
                      <ScaledText style={styles.primaryButtonText}>
                        {isLoading ? t('signIn.signingIn') : t('auth.signIn')}
                      </ScaledText>
                    </TouchableOpacity>

                    <TouchableOpacity
                      onPress={handleBackToEmail}
                      style={styles.backButton}
                      testID="signInBackButton"
                      accessibilityLabel={t('common.back')}
                      accessibilityRole="button"
                      disabled={busy}
                    >
                      <Ionicons name="arrow-back" size={16} color={colors.text.secondary} />
                      <ScaledText style={styles.backText}>{t('common.back')}</ScaledText>
                    </TouchableOpacity>
                  </Animated.View>
                </>
              )}

              <Animated.View entering={entering(260)}>
                <TouchableOpacity
                  onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    (navigation.navigate as any)("SignUp");
                  }}
                  style={styles.linkButton}
                  testID="signUpLink"
                  accessibilityRole="button"
                  accessibilityLabel={t('auth.signUp.signUp')}
                  disabled={busy}
                >
                  <ScaledText style={styles.linkText}>
                    <ScaledText style={styles.linkTextMuted}>{t('auth.noAccount')} </ScaledText>
                    {t('auth.signUp.signUp')}
                  </ScaledText>
                </TouchableOpacity>
              </Animated.View>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

export default SignInScreen;

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
    marginBottom: 16,
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
  emailSection: {
    marginTop: 0,
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
  emailChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
    borderWidth: 1,
    borderColor: 'rgba(10,25,49,0.08)',
    borderRadius: radius.pill,
    paddingVertical: 8,
    paddingLeft: 8,
    paddingRight: 16,
    marginBottom: 16,
    gap: 10,
  },
  emailChipAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#0A1931',
    alignItems: 'center',
    justifyContent: 'center',
  },
  emailChipAvatarText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  emailChipText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#0A1931',
  },
  emailChipChange: {
    fontSize: 14,
    fontWeight: '700',
    color: '#173A65',
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
    marginTop: 4,
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
  forgotButton: {
    alignItems: 'center',
    marginBottom: 12,
    paddingVertical: 6,
  },
  forgotText: {
    fontSize: 14,
    color: '#4D4D4D',
    fontWeight: '600',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    marginBottom: 8,
  },
  backText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4D4D4D',
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
});
