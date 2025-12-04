import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  Alert,
  ActionSheetIOS,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
import moment from "moment";
import { LinearGradient } from "expo-linear-gradient";

const { width } = Dimensions.get("window");

const HomeScreen = () => {
  const navigation = useNavigation();
  const [message, setMessage] = useState("");
  const [userName, setUserName] = useState("User");

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const token = await AsyncStorage.getItem("userToken");
        if (token) {
          const decoded: any = jwtDecode(token);
          setUserName(decoded.name || decoded.username || "User");
        }
      } catch (error) {
        console.log("Error fetching user:", error);
      }
    };
    fetchUser();
  }, []);

  const getGreeting = () => {
    const hour = moment().hour();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  // Show quick action menu
  const showQuickActions = () => {
    if (Platform.OS === 'ios') {
      ActionSheetIOS.showActionSheetWithOptions(
        {
          options: ['Cancel', 'Scan Wardrobe', 'AI Try-On', 'Create Outfit', 'Design Room'],
          cancelButtonIndex: 0,
        },
        (buttonIndex) => {
          if (buttonIndex === 1) (navigation as any).navigate('WardrobeVideo');
          if (buttonIndex === 2) (navigation as any).navigate('AITryOn');
          if (buttonIndex === 3) (navigation as any).navigate('NewOutfit');
          if (buttonIndex === 4) (navigation as any).navigate('DesignRoom');
        }
      );
    } else {
      Alert.alert(
        'Quick Actions',
        'Choose an action',
        [
          { text: 'Scan Wardrobe', onPress: () => (navigation as any).navigate('WardrobeVideo') },
          { text: 'AI Try-On', onPress: () => (navigation as any).navigate('AITryOn') },
          { text: 'Create Outfit', onPress: () => (navigation as any).navigate('NewOutfit') },
          { text: 'Design Room', onPress: () => (navigation as any).navigate('DesignRoom') },
          { text: 'Cancel', style: 'cancel' },
        ]
      );
    }
  };

  return (
    <View className="flex-1 bg-white">
      {/* Background Gradient Mesh (Subtle) */}
      <LinearGradient
        colors={['#E0E7FF', '#F3E8FF', '#FFFFFF']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView className="flex-1">
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          className="flex-1"
        >
          {/* Header */}
          <View className="px-6 pt-2 flex-row justify-between items-center">
            <TouchableOpacity
              onPress={() => (navigation as any).navigate("Profile")}
              className="w-10 h-10 rounded-full bg-white/50 items-center justify-center border border-white/60 shadow-sm"
            >
              <Ionicons name="person-outline" size={20} color="#4B5563" />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => (navigation as any).navigate("Profile")}
              className="w-10 h-10 rounded-full bg-white/50 items-center justify-center border border-white/60 shadow-sm"
            >
              <Ionicons name="settings-outline" size={20} color="#4B5563" />
            </TouchableOpacity>
          </View>

          {/* Main Content */}
          <View className="flex-1 justify-center px-8">
            <View className="mb-8">
              <Text className="text-center text-lg font-medium text-gray-500 mb-2 tracking-wide">
                {getGreeting()}, <Text className="text-purple-600 font-semibold">{userName}</Text>
              </Text>
              <Text className="text-center text-5xl font-bold text-gray-900 leading-tight">
                How can I{"\n"}
                <Text className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-500" style={{ color: '#7C3AED' }}>
                  style you?
                </Text>
              </Text>
            </View>

            {/* Suggestion Chips */}
            <View className="flex-row flex-wrap justify-center gap-3 mb-8">
              {[
                { text: "Outfit for work", icon: "briefcase-outline", color: ["#EFF6FF", "#DBEAFE"] },
                { text: "Date night", icon: "heart-outline", color: ["#FFF1F2", "#FFE4E6"] },
                { text: "Casual weekend", icon: "cafe-outline", color: ["#F0FDF4", "#DCFCE7"] },
                { text: "Summer wedding", icon: "rose-outline", color: ["#FFFBEB", "#FEF3C7"] }
              ].map((chip, index) => (
                <TouchableOpacity
                  key={index}
                  onPress={() => setMessage(chip.text)}
                  className="rounded-full shadow-sm"
                >
                  <LinearGradient
                    colors={chip.color as any}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    className="px-5 py-3 rounded-full flex-row items-center border border-white/50"
                  >
                    <Ionicons name={chip.icon as any} size={16} color="#374151" style={{ marginRight: 6 }} />
                    <Text className="text-gray-700 font-medium text-sm">{chip.text}</Text>
                  </LinearGradient>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Chat Input Area */}
          <View className="px-6 pb-6 pt-4">
            <View className="bg-white rounded-[24px] p-2 shadow-lg shadow-purple-100 border border-purple-50 flex-row items-end">
              <TouchableOpacity className="p-3 bg-gray-50 rounded-full" onPress={showQuickActions}>
                <Ionicons name="add" size={24} color="#6B7280" />
              </TouchableOpacity>

              <TextInput
                className="flex-1 text-base text-gray-900 px-3 py-3 max-h-24"
                placeholder="Describe an occasion or item..."
                placeholderTextColor="#9CA3AF"
                multiline
                value={message}
                onChangeText={setMessage}
              />

              <TouchableOpacity
                disabled={!message.trim()}
                onPress={() => {
                  if (message.trim()) {
                    (navigation as any).navigate("AIChat", { initialMessage: message });
                    setMessage("");
                  }
                }}
              >
                <LinearGradient
                  colors={message.trim() ? ['#8B5CF6', '#6366F1'] : ['#E5E7EB', '#D1D5DB']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  className="w-12 h-12 rounded-full items-center justify-center shadow-md"
                >
                  <Ionicons name="arrow-up" size={24} color="white" />
                </LinearGradient>
              </TouchableOpacity>
            </View>
            <Text className="text-center text-xs text-gray-400 mt-4 font-medium">
              Powered by AIWardrobe Intelligence
            </Text>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

export default HomeScreen;