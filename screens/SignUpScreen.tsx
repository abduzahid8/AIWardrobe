import {
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import React, { useState, useMemo } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";
import { Ionicons } from "@expo/vector-icons";

const SignUpScreen = () => {
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [gender, setGender] = useState("");
  const [username, setUserName] = useState("");
  const [profileImage, setProfileImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigation = useNavigation();
  const { register } = useAuthStore();

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
      Alert.alert("Missing Fields", "Please fill in all required fields");
      return;
    }

    if (!isEmailValid) {
      Alert.alert("Invalid Email", "Please enter a valid email address");
      return;
    }

    if (!isPasswordValid) {
      Alert.alert(
        "Password Requirements",
        "Your password must:\n• Be at least 8 characters\n• Include a lowercase letter (a-z)\n• Include an uppercase letter (A-Z)\n• Include a number (0-9)"
      );
      return;
    }

    if (!isUsernameValid) {
      Alert.alert(
        "Invalid Username",
        "Username must be 3-30 characters and can only contain letters, numbers, and underscores"
      );
      return;
    }

    if (!isGenderValid) {
      Alert.alert(
        "Invalid Gender",
        "Please enter: Male, Female, Other, or Prefer not to say"
      );
      return;
    }

    setIsLoading(true);
    try {
      await register(email, password, username, gender.toLowerCase(), profileImage);
    } catch (error: any) {
      const errorMessage = error.message || "Registration failed";
      console.log('Signup error:', errorMessage);

      // Parse specific error messages
      if (errorMessage.includes("User already registered") || errorMessage.includes("already registered")) {
        Alert.alert("Email Taken", "This email is already registered. Try signing in instead.");
      } else if (errorMessage.includes("Username already exists")) {
        // This comes from our custom trigger or RLS if we implemented checks there, 
        // otherwise Supabase might return a generic database error 23505 for unique violation.
        Alert.alert("Username Taken", "This username is already in use. Please choose another.");
      } else if (errorMessage.includes("Database error") && errorMessage.includes("username")) {
        Alert.alert("Username Taken", "This username is already in use. Please choose another.");
      } else if (errorMessage.includes("Network request failed")) {
        Alert.alert("Connection Failed", "Cannot connect to server. Please check your internet connection.");
      } else {
        Alert.alert("Registration Failed", errorMessage);
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
      <Text style={[styles.checkText, isValid && styles.checkTextValid]}>{text}</Text>
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
          <Text style={styles.title}>{t('auth.signUp')}</Text>

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
            />
            {email && !isEmailValid && (
              <Text style={styles.errorHint}>Please enter a valid email</Text>
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
            />

            {/* Password Requirements Checklist */}
            {password.length > 0 && (
              <View style={styles.passwordChecks}>
                <Text style={styles.checklistTitle}>Password requirements:</Text>
                <PasswordCheckItem isValid={passwordChecks.minLength} text="At least 8 characters" />
                <PasswordCheckItem isValid={passwordChecks.hasLowercase} text="One lowercase letter (a-z)" />
                <PasswordCheckItem isValid={passwordChecks.hasUppercase} text="One uppercase letter (A-Z)" />
                <PasswordCheckItem isValid={passwordChecks.hasNumber} text="One number (0-9)" />
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
            />
            {username && !isUsernameValid && (
              <Text style={styles.errorHint}>3-30 chars, letters/numbers/underscores only</Text>
            )}
          </View>

          {/* Gender */}
          <View style={styles.inputContainer}>
            <TextInput
              style={[styles.input, gender && !isGenderValid && styles.inputError]}
              value={gender}
              onChangeText={setGender}
              placeholder="Gender (Male, Female, Other)"
              placeholderTextColor="rgba(255,255,255,0.4)"
            />
            {gender && !isGenderValid && (
              <Text style={styles.errorHint}>Enter: Male, Female, Other, or Prefer not to say</Text>
            )}
          </View>

          {/* Profile Image (Optional) */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={profileImage}
              onChangeText={setProfileImage}
              placeholder="Profile Image URL (optional)"
              placeholderTextColor="rgba(255,255,255,0.4)"
              autoCapitalize="none"
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
          >
            <Text style={styles.signUpButtonText}>
              {isLoading ? "Creating Account..." : t("auth.signUp")}
            </Text>
          </TouchableOpacity>

          {/* Sign In Link */}
          <TouchableOpacity
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              navigation.goBack();
            }}
            style={styles.signInLinkButton}
          >
            <Text style={styles.signInLinkText}>
              <Text style={styles.signInLinkTextMuted}>{t("auth.haveAccount")} </Text>
              {t("auth.signIn")}
            </Text>
          </TouchableOpacity>

          {/* Trial Mode */}
          <View style={styles.dividerContainer}>
            <View style={styles.divider} />
            <Text style={styles.orText}>or</Text>
            <View style={styles.divider} />
          </View>

          <TouchableOpacity
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              const { startTrial } = useAuthStore.getState();
              startTrial();
            }}
            style={styles.trialButton}
            activeOpacity={0.8}
          >
            <Text style={styles.trialButtonText}>Try App First (3 free tries)</Text>
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
  dividerContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 24,
  },
  divider: {
    flex: 1,
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.1)",
  },
  orText: {
    color: "rgba(255, 255, 255, 0.4)",
    marginHorizontal: 16,
    fontSize: 14,
  },
  trialButton: {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: "center",
  },
  trialButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#FFF",
  },
});
