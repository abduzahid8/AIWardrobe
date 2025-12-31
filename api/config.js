import Constants from "expo-constants";
import { Platform } from "react-native";

// 🔥 SOTA Detection - Using local AliceVision server
const getBackendUrl = () => {
  // For testing SOTA locally (Qwen2.5-VL, Florence-2, etc.)
  // Comment out the next line to use production
  return "http://172.20.10.5:5050";

  // Production server
  // return "https://aiwardrobe-ivh4.onrender.com";
};

export const API_URL = getBackendUrl();

console.log("🌐 API URL:", API_URL);