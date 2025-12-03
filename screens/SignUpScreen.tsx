import {
  Alert,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useState } from "react";
import { useNavigation } from "@react-navigation/native";
import useAuthStore from "../store/auth";
import { useTranslation } from "react-i18next";

const SignUpScreen = () => {
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [gender, setGender] = useState("");
  const [username, setUserName] = useState("");
  const [profileImage, setProfileImage] = useState("");
  const navigation = useNavigation();
  const { register } = useAuthStore();

  const handleSignUp = async () => {
    if (!email || !password || !username || !gender) {
      Alert.alert("Error", "All fields except profile image are req");
      return;
    }
    try {
      await register(email, password, username, gender, profileImage)
    } catch (error: any) {
      Alert.alert("Error", error.message)
    }
  }

  return (
    <View className="flex-1 bg-white justify-center p-4">
      <Text className="text-2xl font-bold text-center mb-6">{t('auth.signUp')}</Text>
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={email}
        onChangeText={setEmail}
        placeholder={t('auth.email')}
      />
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={password}
        onChangeText={setPassword}
        placeholder={t('auth.password')}
        secureTextEntry
      />
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={username}
        onChangeText={setUserName}
        placeholder={t('auth.username')}
      />
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={gender}
        onChangeText={setGender}
        placeholder="Gender"
      />
      <TextInput
        className="border border-gray-300 p-3 mb-4 rounded-lg"
        value={profileImage}
        onChangeText={setProfileImage}
        placeholder="Profile Image URL (optional)"
      />
      <TouchableOpacity onPress={handleSignUp} className="bg-blue-500 p-3 rounded-lg mb-4">
        <Text className="text-center text-white text-lg">{t('auth.signUp')}</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => navigation.goBack()}>
        <Text className="text-center text-blue-500 text-lg">
          {t('auth.haveAccount')} {t('auth.signIn')}
        </Text>
      </TouchableOpacity>
    </View>
  );
};

export default SignUpScreen;

const styles = StyleSheet.create({});
