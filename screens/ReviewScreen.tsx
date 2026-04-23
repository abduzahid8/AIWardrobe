import React, { useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, Alert, ActivityIndicator, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import useAuthStore from '../store/auth';
import { supabase } from '../lib/supabase';
import { useTranslation } from 'react-i18next';

export default function ReviewScanScreen() {
    const { t } = useTranslation();
    const navigation = useNavigation<any>();
    const route = useRoute<any>();
    const { items } = route.params; // Получаем данные от Gemini

    const [reviewedItems, setReviewedItems] = useState(items);
    const [isSaving, setIsSaving] = useState(false);

    // Функция сохранения (Запуск магии Replicate + Supabase)
    const handleSaveToWardrobe = async () => {
        setIsSaving(true);
        try {


            const { user } = useAuthStore.getState();
            if (!user) {
                Alert.alert(t('review.error'), t('review.notAuthorized'));
                return;
            }

            const { data, error } = await supabase.functions.invoke('add-clothing-batch', {
                body: {
                    items: reviewedItems,
                    userId: user.id
                }
            });

            if (error) throw error;

            if (data && data.success) {
                Alert.alert(
                    t('review.done'),
                    t('review.itemsAdded', { count: data.count }),
                    [{ text: t('common.ok'), onPress: () => navigation.navigate("Home") }]
                );
            }
        } catch (error: any) {
            console.error("Ошибка сохранения:", error);
            Alert.alert(t('review.error'), t('review.saveFailed', { error: error.message || "Unknown error" }));
        } finally {
            setIsSaving(false);
        }
    };

    // Удаление лишней вещи из списка (если ИИ ошибся)
    const removeItem = (index: number) => {
        Alert.alert(
            t('review.removeItem'),
            t('review.confirmRemove'),
            [
                { text: t('common.cancel'), style: 'cancel' },
                {
                    text: t('common.remove'),
                    style: 'destructive',
                    onPress: () => {
                        const newItems = [...reviewedItems];
                        newItems.splice(index, 1);
                        setReviewedItems(newItems);
                    },
                },
            ]
        );
    };

    if (isSaving) {
        return (
            <View className="flex-1 bg-[#0A1931] justify-center items-center">
                <ActivityIndicator size="large" color="#fff" />
                <Text className="text-white text-lg font-bold mt-4">{t('review.generating')}</Text>
                <Text className="text-gray-400 text-sm mt-2">{t('review.savingToCloud')}</Text>
            </View>
        );
    }

    return (
        <SafeAreaView className="flex-1 bg-white">
            {/* Заголовок */}
            <View className="flex-row items-center p-4 border-b border-gray-100">
                <TouchableOpacity onPress={() => navigation.goBack()} className="mr-4">
                    <Ionicons name="arrow-back" size={24} color="#0A1931" />
                </TouchableOpacity>
                <Text className="text-xl font-bold">{t('review.itemsFound', { count: reviewedItems.length })}</Text>
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
                    className="bg-[#0A1931] py-4 rounded-2xl items-center shadow-lg"
                >
                    <Text className="text-white font-bold text-lg">
                        {t('review.addToWardrobe', { count: reviewedItems.length })}
                    </Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}