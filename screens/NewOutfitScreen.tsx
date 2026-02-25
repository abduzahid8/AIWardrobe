import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import React, { useState } from "react";
import { useNavigation, useRoute } from "@react-navigation/native";
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
// Removed missing import: import { ClothingItem } from '../types';

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
  const params = (route.params || {}) as {
    selectedItems?: ClothingItem[];
    date?: string;
    savedOutfits?: { [key: string]: any[] };
  };
  const today = new Date();
  const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  const { selectedItems = [], date = todayStr } = params;
  const navigation = useNavigation();
  const { user } = useAuthStore();

  const [caption, setCaption] = useState("");
  const [isOotd, setIsOotd] = useState(false);
  const [occasion] = useState("Work");
  const [visiblilty] = useState("Everyone");
  const [loading, setLoading] = useState(false);


  const handleSave = async () => {
    if (!user?.id) {
      Alert.alert("Error", "User not authenticated");
      return;
    }

    if (selectedItems.length === 0) {
      Alert.alert("Error", "Please add at least one item to the outfit");
      return;
    }

    setLoading(true);
    try {
      const validItems = await Promise.all(
        selectedItems.map(async (item) => {
          // In Supabase migration, we rely on having image URLs (from storage) 
          // or base64 if still local. Assuming passed items have valid 'image' property.
          return {
            id: item.id,
            type: item?.type || "Unknown",
            image: item.image, // Use directly
            x: item.x || 0,
            y: item.y || 0,
          };
        })
      );

      const { error } = await supabase
        .from('saved_outfits')
        .insert({
          user_id: user.id,
          items: validItems,
          date: date,
          occasion: occasion,
          season: "All",
          name: `${occasion} Outfit`,
          caption: caption,
          visibility: visiblilty,
          is_ootd: isOotd,
        })
        .select();

      if (error) throw error;

      Alert.alert("Success", "Outfit saved successfully!");

      if (navigation.canGoBack()) {
        navigation.goBack();
      } else {
        (navigation as any).navigate('Home');
      }

    } catch (error: any) {
      console.error("Error saving outfit:", error);
      Alert.alert("Error", `Failed to save outfit: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  return (
    <SafeAreaView className="flex-1 bg-white">
      <View className="flex-row justify-between items-center p-4">
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text className="text-[#0A1931]">Back</Text>
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
                height: item?.type === "shoes" ? 180 : 240,
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
            <Text className="text-[#0A1931]">{date || "Today"}</Text>
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Add to OOTD story</Text>
            <Switch value={isOotd} onValueChange={setIsOotd} />
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Ocassion</Text>
            <Text className="text-[#0A1931]">{occasion}</Text>
          </View>
          <View className="flex-row items-center justify-between mt-2">
            <Text className="text-gray-500">Visibility</Text>
            <Text className="text-[#0A1931]">{visiblilty}</Text>
          </View>
        </View>
      </View>
      <TouchableOpacity className="bg-[#0A1931] py-3 mx-4 mb-4 rounded" onPress={handleSave} disabled={loading}>
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


