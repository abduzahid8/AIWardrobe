import { Alert, Platform, StyleSheet, TextInput, TouchableOpacity, View, KeyboardAvoidingView, TouchableWithoutFeedback, Keyboard, ScrollView } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import React, { useState, useEffect } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import * as AppleAuthentication from "expo-apple-authentication";
import { Ionicons } from "@expo/vector-icons";
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";
import { SUPABASE_AUTH_ERRORS } from "../constants/authErrors";

const SignInScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { login, signInWithApple, signInWithGoogle } = useAuthStore();
  const [appleAuthAvailable, setAppleAuthAvailable] = useState(false);

  useEffect(() => {
    if (Platform.OS === 'ios') {
      AppleAuthentication.isAvailableAsync().then(setAppleAuthAvailable).catch(() => {});
    }
  }, []);

  const handleAppleSignIn = async () => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      await signInWithApple();
    } catch (error: any) {
      if (error?.code !== 'ERR_REQUEST_CANCELED') {
        Alert.alert(t('signIn.loginFailed'), error.message || t('signIn.appleSignInFailed'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      await signInWithGoogle();
    } catch (error: any) {
      if (error?.message && !error?.message?.includes('cancel')) {
        Alert.alert(t('signIn.loginFailed'), error.message || t('signIn.googleSignInFailed'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignIn = async () => {
    if (!email || !password) {
      Alert.alert(t('common.error'), t('signIn.emailPasswordRequired'));
      return;
    }
    if (isLoading) return;
    setIsLoading(true);
    try {
      await login(email, password);
    } catch (error: any) {
      const errorMessage = error.message || t('signIn.loginFailed');
      if (errorMessage.includes(SUPABASE_AUTH_ERRORS.INVALID_CREDENTIALS) || errorMessage.includes("invalid claim")) {
        Alert.alert(t('signIn.loginFailed'), t('signIn.invalidEmailPassword'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.EMAIL_NOT_CONFIRMED)) {
        Alert.alert(t('signIn.emailNotVerified'), t('signIn.verifyEmailBeforeLogin'));
      } else {
        Alert.alert(t('signIn.loginFailed'), errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior="padding"
      style={styles.container}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <LinearGradient
          colors={["#0A0A0A", "#1A1C29", "#16213E"]}
          style={styles.gradient}
        >
          <ScrollView
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.formContainer}>
              <ScaledText style={styles.title}>{t('auth.signIn')}</ScaledText>

              {/* Social sign-in buttons */}
              {appleAuthAvailable && (
                <AppleAuthentication.AppleAuthenticationButton
                  buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
                  buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE_OUTLINE}
                  cornerRadius={16}
                  style={styles.socialButton}
                  onPress={handleAppleSignIn}
                />
              )}

              <TouchableOpacity
                onPress={handleGoogleSignIn}
                style={styles.googleButton}
                disabled={isLoading}
                activeOpacity={0.8}
                accessibilityLabel={t('signIn.signInWithGoogle')}
                accessibilityRole="button"
              >
                <Ionicons name="logo-google" size={20} color="#FFF" style={styles.googleIcon} />
                <ScaledText style={styles.googleButtonText}>{t('signIn.signInWithGoogle')}</ScaledText>
              </TouchableOpacity>

              <View style={styles.divider}>
                <View style={styles.dividerLine} />
                <ScaledText style={styles.dividerText}>{t('signIn.or')}</ScaledText>
                <View style={styles.dividerLine} />
              </View>

              <View style={styles.inputContainer}>
                <TextInput
                  style={styles.input}
                  value={email}
                  onChangeText={setEmail}
                  placeholder={t('auth.email')}
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  keyboardType="email-address"
                  autoCapitalize="none"
                  accessibilityLabel={t('auth.email')}
                  maxLength={255}
                />
              </View>

              <View style={styles.inputContainer}>
                <TextInput
                  style={styles.input}
                  value={password}
                  onChangeText={setPassword}
                  placeholder={t('auth.password')}
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  secureTextEntry
                  accessibilityLabel={t('auth.password')}
                />
              </View>

              {/* Forgot Password */}
              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  (navigation.navigate as any)("ForgotPassword");
                }}
                style={styles.forgotButton}
                accessibilityLabel={t('auth.forgotPassword')}
                accessibilityRole="button"
              >
                <ScaledText style={styles.forgotText}>{t('auth.forgotPassword')}</ScaledText>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  handleSignIn();
                }}
                style={[styles.primaryButton, isLoading && styles.primaryButtonDisabled]}
                disabled={isLoading}
                activeOpacity={0.8}
                accessibilityLabel={isLoading ? t('signIn.signingIn') : t('auth.signIn')}
                accessibilityRole="button"
              >
                <ScaledText style={styles.primaryButtonText}>
                  {isLoading ? t('signIn.signingIn') : t('auth.signIn')}
                </ScaledText>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  (navigation.navigate as any)("SignUp");
                }}
                style={styles.linkButton}
              >
                <ScaledText style={styles.linkText}>
                  <ScaledText style={styles.linkTextMuted}>{t('auth.noAccount')} </ScaledText>
                  {t('auth.signUp.signUp')}
                </ScaledText>
              </TouchableOpacity>
            </View>
          </ScrollView>
        </LinearGradient>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
};

export default SignInScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 40,
  },
  formContainer: {
    paddingHorizontal: 24,
    width: "100%",
    maxWidth: 400,
    alignSelf: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#FFF",
    textAlign: "center",
    marginBottom: 32,
    letterSpacing: 0.5,
  },
  inputContainer: {
    marginBottom: 16,
  },
  input: {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 16,
    fontSize: 16,
    color: "#FFF",
  },
  primaryButton: {
    backgroundColor: "#FFF",
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: "center",
    marginTop: 8,
    marginBottom: 20,
  },
  primaryButtonDisabled: {
    opacity: 0.5,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#000",
  },
  forgotButton: {
    alignItems: "center",
    marginBottom: 8,
  },
  forgotText: {
    fontSize: 14,
    color: "rgba(255,255,255,0.5)",
    fontWeight: "500",
  },
  linkButton: {
    alignItems: "center",
    marginBottom: 32,
  },
  linkText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#FFF",
  },
  linkTextMuted: {
    color: "rgba(255,255,255,0.6)",
  },
  socialButton: {
    width: "100%",
    height: 50,
    marginBottom: 8,
  },
  googleButton: {
    width: "100%",
    height: 50,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.15)",
    borderRadius: 16,
    marginBottom: 8,
  },
  googleIcon: {
    marginRight: 10,
  },
  googleButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#FFF",
  },
  divider: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "rgba(255,255,255,0.15)",
  },
  dividerText: {
    color: "rgba(255,255,255,0.4)",
    fontSize: 13,
    marginHorizontal: 12,
  },
});
