import React, { useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, Alert, ActivityIndicator, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import { useNavigation, useRoute } from '@react-navigation/native';
import { API_URL } from '../api/config';
import useAuthStore from '../store/auth';

export default function ReviewScanScreen() {
    const navigation = useNavigation<any>();
    const route = useRoute<any>();
    const { items } = route.params; // Получаем данные от Gemini
    const { token } = useAuthStore(); // Токен юзера

    const [reviewedItems, setReviewedItems] = useState(items);
    const [isSaving, setIsSaving] = useState(false);

    // Функция сохранения (Запуск магии Replicate + Supabase)
    const handleSaveToWardrobe = async () => {
        setIsSaving(true);
        try {
            console.log(`📤 Отправляем ${reviewedItems.length} вещей на генерацию...`);

            const response = await axios.post(
                `${API_URL}/wardrobe/add-batch`,
                { items: reviewedItems },
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (response.data.success) {
                Alert.alert(
                    "Готово! 🎉",
                    `Добавлено ${response.data.count} вещей. Сейчас мы генерируем для них красивые фото.`,
                    [{ text: "ОК", onPress: () => navigation.navigate("Home") }]
                );
            }
        } catch (error) {
            console.error("Ошибка сохранения:", error);
            Alert.alert("Ошибка", "Не удалось сохранить вещи. Попробуйте позже.");
        } finally {
            setIsSaving(false);
        }
    };

    // Удаление лишней вещи из списка (если ИИ ошибся)
    const removeItem = (index: number) => {
        const newItems = [...reviewedItems];
        newItems.splice(index, 1);
        setReviewedItems(newItems);
    };

    if (isSaving) {
        return (
            <View className="flex-1 bg-black justify-center items-center">
                <ActivityIndicator size="large" color="#fff" />
                <Text className="text-white text-lg font-bold mt-4">Генерируем одежду...</Text>
                <Text className="text-gray-400 text-sm mt-2">Рисуем фото и сохраняем в облако</Text>
            </View>
        );
    }

    return (
        <SafeAreaView className="flex-1 bg-white">
            {/* Заголовок */}
            <View className="flex-row items-center p-4 border-b border-gray-100">
                <TouchableOpacity onPress={() => navigation.goBack()} className="mr-4">
                    <Ionicons name="arrow-back" size={24} color="black" />
                </TouchableOpacity>
                <Text className="text-xl font-bold">Найдено {reviewedItems.length} вещей</Text>
            </View>

            {/* Список найденного */}
            <FlatList
                data={reviewedItems}
                keyExtractor={(item, index) => index.toString()}
                contentContainerStyle={{ padding: 16 }}
                renderItem={({ item, index }) => (
                    <View className="flex-row bg-gray-50 p-4 rounded-xl mb-3 items-center border border-gray-100">
                        {/* Иконка типа вещи */}
                        <View className="w-12 h-12 bg-blue-100 rounded-full items-center justify-center mr-4">
                            <Ionicons name="shirt-outline" size={24} color="#3b82f6" />
                        </View>

                        <View className="flex-1">
                            <Text className="text-lg font-semibold text-gray-800">{item.itemType}</Text>
                            <Text className="text-sm text-gray-500 capitalize">
                                {item.color} • {item.style} • {item.season}
                            </Text>
                            <Text className="text-xs text-gray-400 mt-1" numberOfLines={1}>
                                {item.description}
                            </Text>
                        </View>

                        <TouchableOpacity onPress={() => removeItem(index)} className="p-2">
                            <Ionicons name="trash-outline" size={20} color="#ef4444" />
                        </TouchableOpacity>
                    </View>
                )}
            />

            {/* Кнопка действия */}
            <View className="p-4 border-t border-gray-100">
                <TouchableOpacity
                    onPress={handleSaveToWardrobe}
                    className="bg-black py-4 rounded-2xl items-center shadow-lg"
                >
                    <Text className="text-white font-bold text-lg">
                        Добавить в гардероб ({reviewedItems.length})
                    </Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}