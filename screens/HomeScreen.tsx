import {
  Dimensions,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Alert
} from "react-native";
import React, { useEffect, useState } from "react";
import { useIsFocused, useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import moment from "moment";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
import axios from "axios";
import * as Location from 'expo-location';
// @ts-ignore
import { API_URL } from "../api/config";
// 👇 Import from safe-area-context instead of react-native
import { SafeAreaView } from "react-native-safe-area-context";

const { width, height } = Dimensions.get("window");

// Weather API Key
const WEATHER_API_KEY = "acec1d31ef3e181c0ca471ac4db642ff"; 

const features = [
  {
    title: "AI Suggestions",
    image: "https://i.pinimg.com/736x/2e/3d/d1/2e3dd14ac81b207ee6d86bc99ef576eb.jpg",
    screen: "AIChat",
  },
  {
    title: "AI Outfit Maker",
    image: "https://i.pinimg.com/736x/50/83/0e/50830e372ee844c1f429b8ef89e26fd1.jpg",
    screen: "AIOutfit",
  },
  {
    title: "AI Try On",
    image: "https://i.pinimg.com/736x/c2/78/95/c2789530a2dc8c9dbfd4aa5e2e70d608.jpg",
    screen: "AITryOn",
  },
  {
    title: "Color Analysis",
    image: "https://i.pinimg.com/736x/84/bf/ce/84bfce1e46977d50631c4ef2f72f83b1.jpg",
    screen: "ColorAnalysis",
  },
  {
    title: "Scan Wardrobe",
    image: "https://cdn-icons-png.flaticon.com/512/3616/3616835.png",
    screen: "ScanWardrobe",
  },
];

const popularItems = [
  {
    username: "Trisha Wushres",
    profile: "https://randomuser.me/api/portraits/women/1.jpg",
    image:
      "https://res.cloudinary.com/db1ccefar/image/upload/v1753859289/skirt3_oanqxj.png",
    itemName: "Floral Skirt",
  },
  {
    username: "Anna Cris",
    profile: "https://randomuser.me/api/portraits/women/2.jpg",
    image:
      "https://res.cloudinary.com/db1ccefar/image/upload/v1753975629/Untitled_design_3_syip4x.png",
    itemName: "Mens Jeans",
  },
  {
    username: "Isabella",
    profile: "https://randomuser.me/api/portraits/women/3.jpg",
    image:
      "https://res.cloudinary.com/db1ccefar/image/upload/v1753975802/Untitled_design_11_p7t2us.png",
    itemName: "Shoes",
  },
];

const initialStories = [
  {
    username: "Your OOTD",
    avatar: "https://picsum.photos/100/100?random=8",
    isOwn: true,
    viewed: false,
  },
  {
    username: "_trishwushres",
    avatar: "https://picsum.photos/100/100?random=10",
    isOwn: false,
    viewed: false,
  },
  {
    username: "myglam",
    avatar: "https://picsum.photos/100/100?random=11",
    isOwn: false,
    viewed: false,
  },
  {
    username: "stylist",
    avatar: "https://picsum.photos/100/100?random=12",
    isOwn: false,
    viewed: false,
  },
];

const HomeScreen = () => {
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  
  const [savedOutfits, setSavedOutfits] = useState<any>({});
  const [stories, setStories] = useState(initialStories);
  const [userId, setUserId] = useState("");
  const [weather, setWeather] = useState<any>(null);
  const [locationLoading, setLocationLoading] = useState(true);

  const generateDates = () => {
    const today = moment().startOf("day");
    const dates = [];
    for (let i = -3; i <= 3; i++) {
      dates.push({
        label: today.clone().add(i, "days").format("ddd, Do MMM"),
        outfit: i === 1,
      });
    }
    return dates;
  };
  const dates = generateDates();

  // Weather Effect
  useEffect(() => {
    (async () => {
      setLocationLoading(true);
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission denied', 'Cannot access location');
          setLocationLoading(false);
          return;
        }

        let location = await Location.getCurrentPositionAsync({});
        const { latitude, longitude } = location.coords;

        const response = await axios.get(
            `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${WEATHER_API_KEY}`
        );
        
        setWeather({
            city: response.data.name,
            temp: Math.round(response.data.main.temp),
            description: response.data.weather[0].description
        });
        
      } catch (error: any) {
        console.log("❌ Weather error:", error.message);
        // Fallback weather
        setWeather({
            city: "Tashkent",
            temp: 24,
            description: "Sunny"
        });
      } finally {
        setLocationLoading(false);
      }
    })();
  }, []);

  // Auth & Outfits Effect
  useEffect(() => {
    const fetchToken = async () => {
      try {
        const token = await AsyncStorage.getItem("userToken");
        if (token) {
          const decoded = jwtDecode(token) as { id: string };
          setUserId(decoded.id);
        }
      } catch (error) {
        console.error("Failed to fetch token", error);
      }
    };

    const fetchSavedOutfits = async () => {
      if (!userId) return;
      try {
        const token = await AsyncStorage.getItem("userToken");
        const response = await axios.get(
          `${API_URL}/save-outfit/user/${userId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const outfits = response.data.reduce(
          (acc: any, outfit: any) => {
            acc[outfit.date] = outfit.items;
            return acc;
          },
          {}
        );
        setSavedOutfits(outfits);
      } catch (error) {
        console.error("Failed to fetch saved outfits", error);
      }
    };

    if (isFocused) {
      fetchToken().then(() => {
        if (userId) fetchSavedOutfits();
      });
    }
  }, [isFocused, navigation, userId]);

  return (
    <SafeAreaView className="flex-1 bg-white">
      <ScrollView className="flex-1 bg-white">
        <View className="flex-row items-center justify-between px-4 pt-4">
          <Text className="text-3xl font-bold">Fits</Text>
          <View className="flex-row items-center gap-3">
            <TouchableOpacity className="bg-black px-4 py-1 rounded-full">
              <Text className="text-white font-semibold text-sm">Upgrade</Text>
            </TouchableOpacity>
            <Ionicons name="notifications-outline" color={"black"} size={24} />
            <Ionicons name="search-outline" color={"black"} size={24} />
          </View>
        </View>

        {/* Weather Widget */}
        <View className="mx-4 mt-6 mb-2 bg-blue-50 p-4 rounded-2xl flex-row justify-between items-center border border-blue-100">
          <View>
            {locationLoading ? (
              <ActivityIndicator color="#3b82f6" />
            ) : weather ? (
              <>
                <Text className="text-gray-500 text-xs font-medium mb-1">
                📍 {weather.city}
                </Text>
                <View className="flex-row items-end">
                  <Text className="text-3xl font-bold text-gray-800">
                  {weather.temp}°C
                  </Text>
                  <Text className="text-gray-600 mb-1 ml-2 capitalize">
                  {weather.description}
                  </Text>
                </View>
              </>
            ) : (
              <Text className="text-gray-500">Weather unavailable</Text>
            )}
          </View>
          
          <TouchableOpacity 
            className="bg-blue-500 px-4 py-2 rounded-xl flex-row items-center shadow-sm"
            onPress={() => {
               (navigation as any).navigate("AIChat"); 
            }}
          >
            <Ionicons name="sparkles" size={16} color="white" style={{marginRight: 5}} />
            <Text className="text-white font-bold text-xs">Ask AI</Text>
          </TouchableOpacity>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          className="mt-4 pl-4"
        >
          {stories.map((story, idx) => (
            <Pressable key={idx} className="mr-4 items-center">
              <View
                className={`w-16 h-16 rounded-full items-center justify-center relative ${
                  story.viewed
                    ? "border-2 border-gray-200"
                    : "border-2 border-purple-400"
                }`}
              >
                <Image
                  className="w-16 h-16 rounded-full"
                  source={{ uri: story.avatar }}
                />
                {story.isOwn && (
                  <View className="absolute bottom-0 right-0 bg-black rounded-full w-5 h-5 items-center justify-center">
                    <Text className="text-white text-xs">+</Text>
                  </View>
                )}
              </View>
              <Text className="text-xs mt-1">{story.username}</Text>
            </Pressable>
          ))}
        </ScrollView>

        <View className="flex-row items-center justify-between mt-6 px-4">
          <Text className="text-lg font-semibold">Your Week</Text>
          <Text className="text-gray-500">Planner </Text>
        </View>

        <ScrollView
          className="mt-4 pl-4"
          horizontal
          showsHorizontalScrollIndicator={false}
        >
          {dates?.map((day, idx) => {
            const today = moment().format("ddd, Do MMM");
            const outfit =
              savedOutfits[day.label] ||
              (day.label == today && savedOutfits[today]
                ? savedOutfits[today]
                : null);

            return (
              <View key={idx} className="mr-3">
                <Pressable
                  onPress={() => {
                    (navigation as any).navigate("AddOutfit", {
                      date: day.label,
                      savedOutfits,
                    });
                  }}
                  className={`w-24 h-40 rounded-xl items-center justify-center overflow-hidden shadow-md ${
                    outfit ? "bg-white" : "bg-gray-50"
                  }`}
                >
                  {!outfit && (
                    <View className="w-full h-full flex items-center justify-center">
                      <Text className="text-3xl text-gray-400">+</Text>
                    </View>
                  )}
                  {outfit && (
                    <View>
                      {outfit.find((item: any) => item.type === "shirt") && (
                        <Image
                          source={{
                            uri: outfit.find((item: any) => item.type === "shirt")
                              ?.image,
                          }}
                          className="w-20 h-20"
                          resizeMode="contain"
                          style={{ maxWidth: "100%", maxHeight: "50%" }}
                        />
                      )}
                      {outfit.find(
                        (item: any) =>
                          item.type === "pants" || item.type == "skirts"
                      ) && (
                        <Image
                          source={{
                            uri: outfit.find(
                              (item: any) =>
                                item.type === "pants" || item.type == "skirts"
                            )?.image,
                          }}
                          className="w-20 h-20"
                          resizeMode="contain"
                          style={{ maxWidth: "100%", maxHeight: "50%" }}
                        />
                      )}
                    </View>
                  )}
                </Pressable>
                <Text className="text-xs text-center mt-1 text-gray-700">
                  {day.label}
                </Text>
              </View>
            );
          })}
        </ScrollView>

        <View className="flex-row flex-wrap justify-between px-4 mt-6">
          {features.map((feature, idx) => (
            <Pressable
              onPress={() => (navigation as any).navigate(feature.screen)}
              style={{
                backgroundColor: ["#FFF1F2", "#EFF6FF", "#F0FFF4", "#FFFBEB"][
                  idx % 4
                ],
                elevation: 3,
              }}
              key={idx}
              className="w-[48%] h-36 mb-4 rounded-2xl shadow-md overflow-hidden"
            >
              <View className="p-3">
                <Text className="font-bold text-[16px] text-gray-800">
                  {feature.title}
                </Text>
                <Text className="text-xs text-gray-500 mt-1">
                  {idx === 0
                    ? "Ask AI for style advice"
                    : idx == 1
                    ? "Create outfits"
                    : idx === 2
                    ? "Virtual Try-On"
                    : idx === 3
                    ? "Find best colors"
                    : "Digitize wardrobe"}
                </Text>
              </View>
              <Image
                source={{ uri: feature.image }}
                className="w-20 h-20 absolute bottom-[-3] right-[-1] rounded-lg"
                style={{ transform: [{ rotate: "12deg" }], opacity: 0.9 }}
                resizeMode="cover"
              />
            </Pressable>
          ))}
        </View>

        <View className="flex-row items-center justify-between mt-6 px-4">
          <Text className="text-lg font-semibold">Popular this week</Text>
          <Text className="text-gray-500">More</Text>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          className="mt-4 pl-4"
        >
          {popularItems.map((item, idx) => (
            <View key={idx} className="w-36 mr-4">
              <Image
                className="w-36 h-44 rounded-lg"
                source={{ uri: item?.image }}
              />
              <View className="flex-row items-center mt-2">
                <Image
                  className="w-6 h-6 rounded-full mr-2"
                  source={{ uri: item?.profile }}
                />
                <Text className="text-xs font-medium">{item.username}</Text>
              </View>
              <Text className="text-xs text-gray-500 mt-1">
                {item.itemName}
              </Text>
            </View>
          ))}
        </ScrollView>
      </ScrollView>
    </SafeAreaView>
  );
};

export default HomeScreen;

const styles = StyleSheet.create({});