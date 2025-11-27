import Constants from "expo-constants";
import { Platform } from "react-native";

// Эта функция автоматически достает IP вашего компьютера из настроек Expo
const getBackendUrl = () => {
  return "https://aiwardrobe-ivh4.onrender.com";
};

export const API_URL = getBackendUrl();

console.log("🌐 API URL:", API_URL);