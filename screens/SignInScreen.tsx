import {
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  KeyboardAvoidingView,
  TouchableWithoutFeedback,
  Keyboard,
  ScrollView,
} from "react-native";
import React, { useState } from "react";
import { useNavigation } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";

const SignInScreen = () => {
  const navigation = useNavigation();
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuthStore();

  const handleSignIn = async () => {
    if (!email || !password) {
      Alert.alert("Error", "Email and password are required");
      return;
    }
    if (isLoading) return;
    setIsLoading(true);
    try {
      await login(email, password);
    } catch (error: any) {
      const errorMessage = error.message || "Login failed";
      if (errorMessage.includes("Invalid login credentials") || errorMessage.includes("invalid claim")) {
        Alert.alert("Login Failed", "Invalid email or password.");
      } else if (errorMessage.includes("Email not confirmed")) {
        Alert.alert("Email Not Verified", "Please verify your email address before logging in.");
      } else {
        Alert.alert("Login Failed", errorMessage);
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
              <Text style={styles.title}>{t('auth.signIn')}</Text>

              <View style={styles.inputContainer}>
                <TextInput
                  style={styles.input}
                  value={email}
                  onChangeText={setEmail}
                  placeholder={t('auth.email')}
                  placeholderTextColor="rgba(255,255,255,0.4)"
                  keyboardType="email-address"
                  autoCapitalize="none"
                  accessibilityLabel="Email address"
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
                  accessibilityLabel="Password"
                />
              </View>

              {/* Forgot Password */}
              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  (navigation.navigate as any)("ForgotPassword");
                }}
                style={styles.forgotButton}
                accessibilityLabel="Forgot password"
                accessibilityRole="button"
              >
                <Text style={styles.forgotText}>Forgot Password?</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  handleSignIn();
                }}
                style={[styles.primaryButton, isLoading && styles.primaryButtonDisabled]}
                disabled={isLoading}
                activeOpacity={0.8}
                accessibilityLabel={isLoading ? "Signing in" : "Sign in"}
                accessibilityRole="button"
              >
                <Text style={styles.primaryButtonText}>
                  {isLoading ? "Signing In..." : t('auth.signIn')}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                  (navigation.navigate as any)("SignUp");
                }}
                style={styles.linkButton}
              >
                <Text style={styles.linkText}>
                  <Text style={styles.linkTextMuted}>{t('auth.noAccount')} </Text>
                  {t('auth.signUp')}
                </Text>
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
});
