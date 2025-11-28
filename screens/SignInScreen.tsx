import {
  Alert,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useState } from "react";
import { useNavigation } from "@react-navigation/native";
import useAuthStore from "../store/auth";
export const API_URL = "https://aiwardrobe-ivh4.onrender.com";

const SignInScreen = () => {
  const navigation = useNavigation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { login } = useAuthStore();
  const handleSignIn = async () => {
    if (!email || !password) {
      Alert.alert("Error", "Email and password are required");
      return;
    }
    try {
      await login(email, password);
    } catch (error: any) {
      Alert.alert("Error", error.message);
    }
  };
  console.log("Мой текущий API URL:", API_URL);
  return (
    <View className="flex-1 bg-white justify-center p-4">
      <Text className="text-2xl font-bold text-center mb-6">SignIn</Text>
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={email}
        onChangeText={setEmail}
        placeholder="Email"
      />
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={password}
        onChangeText={setPassword}
        placeholder="Password"
        secureTextEntry
      />
      <TouchableOpacity onPress={handleSignIn} className="bg-blue-500 p-3 rounded-lg mb-4">
        <Text className="text-center text-white text-lg">Sign In</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => (navigation.navigate as any)("SignUp")}>
        <Text className="text-center text-blue-500 text-lg">
          Don't have an account? Sign Up
        </Text>
      </TouchableOpacity>
    </View>
  );
};

export default SignInScreen;

const styles = StyleSheet.create({});
