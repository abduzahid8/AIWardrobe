import { Alert, StyleSheet, TextInput, TouchableOpacity, View, ScrollView, KeyboardAvoidingView, Platform, } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import React, { useState, useMemo, useEffect } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import * as AppleAuthentication from "expo-apple-authentication";
import { Ionicons } from "@expo/vector-icons";
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";
import { createLogger } from "../src/utils/logger";
import { SUPABASE_AUTH_ERRORS } from "../constants/authErrors";

const logger = createLogger('SignUp');

const SignUpScreen = () => {
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [gender, setGender] = useState("");
  const [username, setUserName] = useState("");
  const [profileImage, setProfileImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigation = useNavigation();
  const { register, signInWithApple, signInWithGoogle } = useAuthStore();
  const [appleAuthAvailable, setAppleAuthAvailable] = useState(false);

  useEffect(() => {
    if (Platform.OS === 'ios') {
      AppleAuthentication.isAvailableAsync().then(setAppleAuthAvailable).catch(() => {});
    }
  }, []);

  const handleAppleSignUp = async () => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      await signInWithApple();
    } catch (error: any) {
      if (error?.code !== 'ERR_REQUEST_CANCELED') {
        Alert.alert(t('signUp.registrationFailed'), error.message || t('signIn.appleSignInFailed'));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      await signInWithGoogle();
    } catch (error: any) {
      if (error?.message && !error?.message?.includes('cancel')) {
        Alert.alert(t('signUp.registrationFailed'), error.message || t('signIn.googleSignInFailed'));
      }
    } finally {
      setIsLoading(false);
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

  // Email validation
  const isEmailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  // Username validation (3-30 chars, letters, numbers, underscores only)
  const isUsernameValid = /^[a-zA-Z0-9_]{3,30}$/.test(username);

  // Gender validation
  const validGenders = ['male', 'female', 'other', 'prefer_not_to_say'];
  const isGenderValid = validGenders.includes(gender.toLowerCase());

  // Can submit form?
  const canSubmit = isEmailValid && isPasswordValid && isUsernameValid && isGenderValid;

  const handleSignUp = async () => {
    if (!email || !password || !username || !gender) {
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

    if (!isGenderValid) {
      Alert.alert(
        t('signUp.invalidGender'),
        t('signUp.genderOptions')
      );
      return;
    }

    setIsLoading(true);
    try {
      await register(email, password, username, gender.toLowerCase(), profileImage);
    } catch (error: any) {
      const errorMessage = error.message || t('signUp.registrationFailed');
      logger.error('Signup error', errorMessage);

      // Parse specific error messages
      if (errorMessage.includes(SUPABASE_AUTH_ERRORS.USER_ALREADY_REGISTERED) || errorMessage.includes("already registered")) {
        Alert.alert(t('signUp.emailTaken'), t('signUp.emailAlreadyRegistered'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.USERNAME_EXISTS)) {
        // This comes from our custom trigger or RLS if we implemented checks there, 
        // otherwise Supabase might return a generic database error 23505 for unique violation.
        Alert.alert(t('signUp.usernameTaken'), t('signUp.usernameAlreadyUse'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.DATABASE_ERROR) && errorMessage.includes("username")) {
        Alert.alert(t('signUp.usernameTaken'), t('signUp.usernameAlreadyUse'));
      } else if (errorMessage.includes(SUPABASE_AUTH_ERRORS.NETWORK_FAILED)) {
        Alert.alert(t('signUp.connectionFailed'), t('signUp.cannotConnectServer'));
      } else {
        Alert.alert(t('signUp.registrationFailed'), errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const PasswordCheckItem = ({ isValid, text }: { isValid: boolean; text: string }) => (
    <View style={styles.checkItem}>
      <Ionicons
        name={isValid ? "checkmark-circle" : "ellipse-outline"}
        size={16}
        color={isValid ? "#FFD700" : "rgba(255,255,255,0.3)"}
      />
      <ScaledText style={[styles.checkText, isValid && styles.checkTextValid]}>{text}</ScaledText>
    </View>
  );

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.flex1}
    >
      <LinearGradient
        colors={["#0A0A0A", "#1A1C29", "#16213E"]}
        style={styles.flex1}
      >
        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <ScaledText style={styles.title}>{t('auth.signUp.signUp')}</ScaledText>

          {/* Social sign-up buttons */}
          {appleAuthAvailable && (
            <AppleAuthentication.AppleAuthenticationButton
              buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_UP}
              buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE_OUTLINE}
              cornerRadius={16}
              style={styles.socialButton}
              onPress={handleAppleSignUp}
            />
          )}

          <TouchableOpacity
            onPress={handleGoogleSignUp}
            style={styles.googleButton}
            disabled={isLoading}
            activeOpacity={0.8}
            accessibilityLabel={t('signUp.signUpWithGoogle')}
            accessibilityRole="button"
          >
            <Ionicons name="logo-google" size={20} color="#FFF" style={styles.googleIcon} />
            <ScaledText style={styles.googleButtonText}>{t('signUp.signUpWithGoogle')}</ScaledText>
          </TouchableOpacity>

          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <ScaledText style={styles.dividerText}>{t('signIn.or')}</ScaledText>
            <View style={styles.dividerLine} />
          </View>

          {/* Email */}
          <View style={styles.inputContainer}>
            <TextInput
              style={[styles.input, email && !isEmailValid && styles.inputError]}
              value={email}
              onChangeText={setEmail}
              placeholder={t('auth.email')}
              placeholderTextColor="rgba(255,255,255,0.4)"
              keyboardType="email-address"
              autoCapitalize="none"
              accessibilityLabel={t('auth.email')}
              maxLength={255}
            />
            {email && !isEmailValid && (
              <ScaledText style={styles.errorHint}>{t('auth.signUp.pleaseEnterValidEmail')}</ScaledText>
            )}
          </View>

          {/* Password with requirements */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={password}
              onChangeText={setPassword}
              placeholder={t('auth.password')}
              placeholderTextColor="rgba(255,255,255,0.4)"
              secureTextEntry
              accessibilityLabel={t('auth.password')}
              maxLength={128}
            />

            {/* Password Requirements Checklist */}
            {password.length > 0 && (
              <View style={styles.passwordChecks}>
                <ScaledText style={styles.checklistTitle}>{t('auth.signUp.passwordRequirements')}</ScaledText>
                <PasswordCheckItem isValid={passwordChecks.minLength} text={t('resetPassword.minLength')} />
                <PasswordCheckItem isValid={passwordChecks.hasLowercase} text={t('resetPassword.lowercase')} />
                <PasswordCheckItem isValid={passwordChecks.hasUppercase} text={t('resetPassword.uppercase')} />
                <PasswordCheckItem isValid={passwordChecks.hasNumber} text={t('resetPassword.number')} />
              </View>
            )}
          </View>

          {/* Username */}
          <View style={styles.inputContainer}>
            <TextInput
              style={[styles.input, username && !isUsernameValid && styles.inputError]}
              value={username}
              onChangeText={setUserName}
              placeholder={t('auth.username')}
              placeholderTextColor="rgba(255,255,255,0.4)"
              autoCapitalize="none"
              accessibilityLabel={t('auth.username')}
              maxLength={30}
            />
            {username && !isUsernameValid && (
              <ScaledText style={styles.errorHint}>{t('signUp.usernameRequirements')}</ScaledText>
            )}
          </View>

          {/* Gender Picker */}
          <View style={styles.inputContainer}>
            <ScaledText style={styles.genderLabel}>{t('auth.signUp.gender')}</ScaledText>
            <View style={styles.genderRow}>
              {['male', 'female', 'other', 'prefer_not_to_say'].map((g) => (
                <TouchableOpacity
                  key={g}
                  onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    setGender(g);
                  }}
                  style={[
                    styles.genderChip,
                    gender === g && styles.genderChipActive,
                  ]}
                  accessibilityLabel={g === 'prefer_not_to_say' ? t('signUp.preferNotToSay') : g}
                  accessibilityRole="radio"
                  accessibilityState={{ selected: gender === g }}
                >
                  <ScaledText
                    style={[
                      styles.genderChipText,
                      gender === g && styles.genderChipTextActive,
                    ]}
                  >
                    {g === 'prefer_not_to_say' ? t('signUp.skip') : g.charAt(0).toUpperCase() + g.slice(1)}
                  </ScaledText>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Profile Image (Optional) */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={profileImage}
              onChangeText={setProfileImage}
              placeholder={t('signUp.profileImageUrlOptional')}
              placeholderTextColor="rgba(255,255,255,0.4)"
              autoCapitalize="none"
              accessibilityLabel={t('signUp.profileImageUrlOptional')}
              maxLength={500}
            />
          </View>

          {/* Sign Up Button */}
          <TouchableOpacity
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              handleSignUp();
            }}
            style={[styles.signUpButton, !canSubmit && styles.signUpButtonDisabled]}
            disabled={isLoading || !canSubmit}
            activeOpacity={0.8}
            accessibilityLabel={isLoading ? t('signUp.creatingAccount') : t('auth.signUp.signUp')}
            accessibilityRole="button"
          >
            <ScaledText style={styles.signUpButtonText}>
              {isLoading ? t('signUp.creatingAccount') : t("auth.signUp.signUp")}
            </ScaledText>
          </TouchableOpacity>

          {/* Sign In Link */}
          <TouchableOpacity
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              navigation.goBack();
            }}
            style={styles.signInLinkButton}
          >
            <ScaledText style={styles.signInLinkText}>
              <ScaledText style={styles.signInLinkTextMuted}>{t("auth.haveAccount")} </ScaledText>
              {t("auth.signIn")}
            </ScaledText>
          </TouchableOpacity>

        </ScrollView>
      </LinearGradient>
    </KeyboardAvoidingView>
  );
};

export default SignUpScreen;

const styles = StyleSheet.create({
  flex1: {
    flex: 1,
  },
  container: {
    flexGrow: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
    paddingVertical: 40,
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
  inputError: {
    borderColor: "#EF4444",
  },
  errorHint: {
    color: "#EF4444",
    fontSize: 12,
    marginTop: 6,
    marginLeft: 6,
  },
  passwordChecks: {
    marginTop: 12,
    paddingHorizontal: 6,
    backgroundColor: "rgba(255, 255, 255, 0.03)",
    padding: 12,
    borderRadius: 12,
  },
  checklistTitle: {
    fontSize: 12,
    fontWeight: "600",
    color: "rgba(255,255,255,0.6)",
    marginBottom: 8,
  },
  checkItem: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 6,
  },
  checkText: {
    fontSize: 13,
    color: "rgba(255,255,255,0.4)",
    marginLeft: 8,
  },
  checkTextValid: {
    color: "rgba(255,255,255,0.9)",
  },
  signUpButton: {
    backgroundColor: "#FFF",
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: "center",
    marginTop: 12,
    marginBottom: 20,
  },
  signUpButtonDisabled: {
    opacity: 0.5,
  },
  signUpButtonText: {
    fontSize: 16,
    fontWeight: "700",
    color: "#000",
  },
  signInLinkButton: {
    alignItems: "center",
    marginBottom: 32,
  },
  signInLinkText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#FFF",
  },
  signInLinkTextMuted: {
    color: "rgba(255,255,255,0.6)",
  },
  genderLabel: {
    fontSize: 13,
    fontWeight: "600",
    color: "rgba(255,255,255,0.5)",
    marginBottom: 10,
  },
  genderRow: {
    flexDirection: "row",
    gap: 8,
  },
  genderChip: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  genderChipActive: {
    backgroundColor: "rgba(255, 215, 0, 0.15)",
    borderColor: "#FFD700",
  },
  genderChipText: {
    fontSize: 13,
    fontWeight: "500",
    color: "rgba(255,255,255,0.5)",
  },
  genderChipTextActive: {
    color: "#FFD700",
    fontWeight: "600",
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
