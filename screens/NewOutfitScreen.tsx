import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useEffect, useState } from "react";
import { useNavigation, useRoute } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
import axios from "axios";

interface ClothingItem {
  id: number;
  image: string;
  x: number;
  y: number;
  type?: "pants" | "shoes" | "shirt" | "skirts";
  gender?: "m" | "f" | "unisex";
}

const NewOutfitScreen = () => {
  const route = useRoute();
  const { selectedItems, date, savedOutfits } = route.params as {
    selectedItems: ClothingItem[];
    date: string;
    savedOutfits: { [key: string]: any[] };
  };
  const navigation = useNavigation();
  const [caption, setCaption] = useState("");
  const [isOotd, setIsOotd] = useState(false);
  const [occasion, setOcassion] = useState("Work");
  const [visiblilty, setVisiblity] = useState("Everyone");
  const [loading, setLoading] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const BASE_URL = "https://aiwardrobe-ivh4.onrender.com";

  useEffect(() => {
    const fetchToken = async () => {
      try {
        const token = await AsyncStorage.getItem("userToken");
        if (token) {
          const decoded = jwtDecode(token) as { id: string };
          setUserId(decoded.id);
        } else {
          Alert.alert("Error", "No authentication token found");
        }
      } catch (error) {
        console.error("Failed to fetch token:", error);
        Alert.alert("Error", "Failed to authenticate");
      }
    };
    fetchToken();
  }, []);
  const convertToBase64 = async (image: string) => {
    return image; // Use URL directly
  };
  const handleSave = async () => {
    if (!userId) {
      Alert.alert("Error", "User not authenticated");
      return;
    }
    setLoading(true);
    try {
      const validateItems = await Promise.all(
        selectedItems.map(async (item) => {
          const base64Image = await convertToBase64(item?.image);
          return {
            id: item.id,
            type: item?.type || "Unknown",
            image: base64Image,
            x: item.x || 0,
            y: item.y || 0,
          };
        })
      );

      const validItems = validateItems.filter((item) => item !== null);

      if (validItems.length === 0) {
        throw new Error("No valid items to save");
      }

      const outfitData = {
        userId,
        date,
        items: validItems,
        caption,
        occasion,
        visiblilty,
        isOotd,
      };

      const token = await AsyncStorage.getItem("userToken");
      const response = await axios.post(`${BASE_URL}/save-outfit`, outfitData, {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      const updatedOutfits = {
        ...savedOutfits,
        [date]: response.data.outfit.items,
      };
      navigation.reset({
        index: 0,
        routes: [
          {
            name: "Home" as never,
            params: {
              screen: "Home",
              params: { savedOutfits: updatedOutfits },
            },
          },
        ],
      });
    } catch (error) {
      console.log("Save error", error instanceof Error ? error.message : error);
    } finally {
      setLoading(false);
    }
  };
  return (
    <SafeAreaView className="flex-1 bg-white">
      <View className="flex-row justify-between items-center p-4">
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text className="text-black">Back</Text>
        </TouchableOpacity>
        <Text className="text-lg font-semibold">New Outfit</Text>
      </View>
      <View className="flex-1 items-center justify-center">
        {selectedItems
          ?.sort((a, b) => {
            const order: { [key: string]: number } = { shirt: 1, skirts: 2, pants: 3, shoes: 4 };
            return (order[a.type || ""] || 5) - (order[b.type || ""] || 5);
          })
          .map((item, index) => (
            <Image
              resizeMode="contain"
              key={index}
              source={{ uri: item?.image }}
              style={{
                width: 240,
                height: item?.type == "shoes" ? 180 : 240,
                marginBottom: index < selectedItems.length - 1 ? -60 : 0,
              }}
            />
          ))}
      </View>
      <View className="p-4">
        <TextInput
          className="border-b border-gray-300 pb-2 text-gray-500"
          placeholder="Add a caption..."
          value={caption}
          onChangeText={setCaption}
        />
        <View className="mt-4">
          <View className="flex-row items-center justify-between">
            <Text className="text-gray-500">Date</Text>
            <Text className="text-black">{date || "Today"}</Text>
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Add to OOTD story</Text>
            <Switch value={isOotd} onValueChange={setIsOotd} />
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Ocassion</Text>
            <Text className="text-black">{occasion}</Text>
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Visibility</Text>
            <Text className="text-black">{visiblilty}</Text>
          </View>
        </View>
      </View>
      <TouchableOpacity className="bg-black py-3 mx-4 mb-4 rounded" onPress={handleSave} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="#ffffff" />
        ) : (
          <Text className="text-white text-center font-semibold">Save outfit</Text>
        )}
      </TouchableOpacity>
    </SafeAreaView>
  );
};

export default NewOutfitScreen;

const styles = StyleSheet.create({});
