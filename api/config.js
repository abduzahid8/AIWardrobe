import Constants from "expo-constants";
import { Platform } from "react-native";

// Эта функция автоматически достает IP вашего компьютера из настроек Expo
const getBackendUrl = () => {
  // Для симулятора iOS
  if (Platform.OS === 'ios') {
    const debuggerHost = Constants.expoConfig?.hostUri || Constants.manifest?.debuggerHost;
    
    if (debuggerHost) {
      const ip = debuggerHost.split(":")[0]; // Берем только IP (например, 192.168.1.5)
      return `http://${ip}:3000`; // Возвращаем готовый адрес
    }
  }
  
  // Для Android эмулятора (он использует специальный IP)
  if (Platform.OS === 'android') {
    return "http://10.0.2.2:3000";
  }

  // Запасной вариант (если вдруг не сработало)
  return "http://localhost:3000";
};

export const API_URL = getBackendUrl();

console.log("🌐 API URL:", API_URL);