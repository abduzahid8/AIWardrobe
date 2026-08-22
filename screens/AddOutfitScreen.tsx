import { Image, SafeAreaView, ScrollView, StyleSheet, TextInput, TouchableOpacity, View, ActivityIndicator, Alert } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import React, { useState } from "react";
import { useNavigation, useRoute } from "@react-navigation/native";
import { mpants, mshirts, pants, shoes, skirts, tops } from "../data/legacyImageCatalog";
import { Ionicons } from "@expo/vector-icons";
import { useTranslation } from "react-i18next";
// Убедитесь, что путь верный!
import Config from "../src/config/env";
const API_URL = Config.api.url;

const AddOutfitScreen = () => {
  const route = useRoute();
  const { t } = useTranslation();
  // @ts-ignore
  const { date, savedOutfits } = route?.params || {};
  const navigation = useNavigation();

  // 1. Состояния (State)
  const [link, setLink] = useState("");
  const [loadingLink, setLoadingLink] = useState(false);
  const [selected, setSelected] = useState<number[]>([]);

  const [popularClothes, setPopularClothes] = useState([
    ...pants,
    ...mpants,
    ...shoes,
    ...tops,
    ...mshirts,
    ...skirts,
  ]
    .map((item, idx) => ({
      ...item,
      id: idx + 1,
    }))
    .filter((item) => item.image));

  // 2. Функция скачивания (ТЕПЕРЬ ОНА ВНУТРИ КОМПОНЕНТА)
  const handleLinkPaste = async () => {
    if (!link.trim()) return;
    setLoadingLink(true);

    try {

      const response = await fetch(`${API_URL}/scrape-item`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: link }),
      });

      const data = await response.json();

      if (data.image) {

        const newItem = {
          id: Date.now(),
          image: data.image,
          name: data.title,
          gender: "u",
          type: "imported",
        };

        // Добавляем в начало и выбираем
        setPopularClothes((prev) => [newItem, ...prev]);
        setSelected((prev) => [...prev, newItem.id]);
        setLink("");
        Alert.alert(t('common.success'), t('addOutfit.itemImported'));
      } else {
        Alert.alert(t('common.error'), t('addOutfit.errors.noImage'));
      }
    } catch (error) {

      Alert.alert(t('common.error'), t('addOutfit.errors.connectionFailed'));
    } finally {
      setLoadingLink(false);
    }
  };

  const toggleSelect = (id: number) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleNext = () => {
    const selectedItems = popularClothes.filter((item) =>
      selected.includes(item?.id)
    );
    // @ts-ignore
    navigation.navigate("DesignRoom", {
      selectedItems,
      date,
      savedOutfits,
    });
  };

  return (
    <SafeAreaView className="flex-1 bg-white">
      <View className="flex-row items-center justify-between px-4">
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={28} color="#0A1931" />
        </TouchableOpacity>
        <ScaledText className="text-lg font-semibold">{t('addOutfit.title')}</ScaledText>
        <ScaledText className="text-gray-500">{date}</ScaledText>
      </View>

      {/* 👇 НОВЫЙ БЛОК ИМПОРТА 👇 */}
      <View className="mx-4 mt-4 p-3 bg-gray-50 rounded-xl border border-gray-200">
        <ScaledText className="text-xs text-gray-500 mb-2 ml-1">{t('addOutfit.importFromWeb')}</ScaledText>
        <View className="flex-row items-center">
          <TextInput
            className="flex-1 bg-white p-3 rounded-lg border border-gray-300 mr-2"
            placeholder={t('addOutfit.pasteLink')}
            value={link}
            onChangeText={setLink}
            maxLength={2000}
          />
          <TouchableOpacity
            onPress={handleLinkPaste}
            disabled={loadingLink}
            className="bg-[#0A1931] w-12 h-12 rounded-lg items-center justify-center"
          >
            {loadingLink ? (
              <ActivityIndicator color="white" size="small" />
            ) : (
              <Ionicons name="download-outline" size={24} color="white" />
            )}
          </TouchableOpacity>
        </View>
      </View>
      {/* 👆 -------------------- 👆 */}

      <View className="flex-row justify-around mt-4 px-4">
        <TouchableOpacity className="bg-gray-100 w-[30%] py-3 rounded-lg items-center">
          <Ionicons name="camera-outline" size={22} color="#0A1931" />
          <ScaledText className="font-medium mt-1">{t('addOutfit.selfie')}</ScaledText>
        </TouchableOpacity>
        <TouchableOpacity className="bg-gray-100 w-[30%] py-3 rounded-lg items-center">
          <Ionicons name="sparkles-outline" size={22} color="#0A1931" />
          <ScaledText className="font-medium mt-1">{t('addOutfit.suggestions')}</ScaledText>
        </TouchableOpacity>
        <TouchableOpacity className="bg-gray-100 w-[30%] py-3 rounded-lg items-center">
          <Ionicons name="shirt-outline" size={22} color="#0A1931" />
          <ScaledText className="font-medium mt-1">{t('addOutfit.saved')}</ScaledText>
        </TouchableOpacity>
      </View>

      <ScrollView className="flex-1 mt-4">
        <ScaledText className="text-lg font-semibold px-4 mt-4">{t('addOutfit.popularClothes')}</ScaledText>
        <View className="flex-row flex-wrap px-4 mt-2 mb-20">
          {popularClothes?.map((item, index) => (
            <TouchableOpacity
              key={item.id || index}
              onPress={() => toggleSelect(item?.id)}
              className="w-1/3 p-1 relative"
            >
              <Image
                className="w-full h-32 rounded-md bg-gray-100"
                source={{ uri: item?.image }}
                resizeMode="contain"
              />
              <View className="absolute top-2 right-2 w-6 h-6 rounded-full border-2 items-center justify-center">
                <ScaledText className="text-xs">
                  {item.gender === "m" ? "♂" : item.gender === "f" ? "♀" : "⚪"}
                </ScaledText>
              </View>
              <View
                className={`absolute top-2 left-2 w-6 h-6 rounded-full border-2 ${selected.includes(item.id) ? "bg-[#0A1931]" : "border-gray-400"
                  } items-center justify-center`}
              >
                {selected.includes(item?.id) && (
                  <Ionicons name="checkmark" size={16} color="white" />
                )}
              </View>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>

      {selected.length > 0 && (
        <View className="absolute bottom-0 left-0 right-0 bg-white p-3 border-t border-gray-200">
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            className=""
          >
            {selected?.map((id) => {
              const item = popularClothes.find((c) => c.id === id);
              if (!item) return null;
              return (
                <Image
                  key={id}
                  source={{ uri: item?.image }}
                  className="w-16 h-16 mr-3 rounded-md bg-gray-100"
                />
              );
            })}
          </ScrollView>
          <TouchableOpacity
            onPress={handleNext}
            className="bg-[#0A1931] py-3 rounded-lg mt-3 mb-3 items-center self-end w-24"
          >
            <ScaledText className="text-white font-semibold">{t('common.next')}</ScaledText>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  );
};

export default AddOutfitScreen;

const styles = StyleSheet.create({});