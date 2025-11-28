import React, { useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import axios from 'axios';
import { API_URL } from '../api/config';
import useAuthStore from '../store/auth';

export default function ReviewScanScreen({ route, navigation }: any) {
    // Получаем вещи, которые нашел Gemini
    const { items } = route.params;
    const [reviewedItems, setReviewedItems] = useState(items);
    const { token } = useAuthStore();

    // Функция сохранения в реальную базу
// Функция сохранения в реальную базу
    const saveAllToWardrobe = async () => {
        try {
        console.log("📤 Сохраняем вещи...");
      
      // Отправляем весь массив разом на новый роут
        await axios.post(`${API_URL}/wardrobe/add-batch`, { 
            items: reviewedItems 
        }, { 
              headers: { Authorization: `Bearer ${token}` } // Не забудь токен!
        });
      
        Alert.alert("Успех", `Добавлено ${reviewedItems.length} вещей в ваш гардероб!`);
      
      // Возвращаемся домой или в гардероб
        navigation.navigate("Home"); 
      
        } catch (e) {
        console.error(e);
        Alert.alert("Ошибка", "Не удалось сохранить вещи. Проверьте соединение.");
        }
    };

    return (
        <SafeAreaView className="flex-1 bg-white p-4">
            <Text className="text-2xl font-bold mb-4">Найдено {reviewedItems.length} вещей</Text>

            <FlatList
                data={reviewedItems}
                keyExtractor={(item, index) => index.toString()}
                renderItem={({ item, index }) => (
                    <View className="bg-gray-100 p-4 rounded-xl mb-3 flex-row justify-between items-center">
                        <View>
                            <Text className="text-lg font-semibold">{item.itemType}</Text>
                            <Text className="text-gray-500">{item.color} • {item.style}</Text>
                            <Text className="text-gray-400 text-xs">{item.description}</Text>
                        </View>
                        <TouchableOpacity
                            onPress={() => {
                                // Удалить вещь из списка, если ИИ ошибся
                                const newItems = [...reviewedItems];
                                newItems.splice(index, 1);
                                setReviewedItems(newItems);
                            }}
                            className="bg-red-100 p-2 rounded-lg"
                        >
                            <Text className="text-red-500">✕</Text>
                        </TouchableOpacity>
                    </View>
                )}
            />

            <TouchableOpacity
                onPress={saveAllToWardrobe}
                className="bg-black p-4 rounded-2xl items-center mt-4"
            >
                <Text className="text-white font-bold text-lg">Добавить всё в гардероб</Text>
            </TouchableOpacity>
        </SafeAreaView>
    );
}