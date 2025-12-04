import React, { useState } from "react";
import {
    View,
    Text,
    ScrollView,
    Image,
    TouchableOpacity,
    Pressable,
    Dimensions,
    StyleSheet,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import { SafeAreaView } from "react-native-safe-area-context";
import { features, popularItems } from "../data";
import { LinearGradient } from "expo-linear-gradient";

const { width } = Dimensions.get("window");

const DiscoverScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [activeTab, setActiveTab] = useState("Closet");

    return (
        <View className="flex-1 bg-gray-50">
            {/* Header Background */}
            <View className="absolute top-0 left-0 right-0 h-32 bg-white shadow-sm z-0" />

            <SafeAreaView className="flex-1">
                {/* Header */}
                <View className="px-6 pt-2 pb-4 flex-row items-center justify-between z-10 bg-white">
                    <Text className="text-3xl font-bold text-gray-900 tracking-tight">Discover</Text>
                    <View className="flex-row gap-3">
                        <TouchableOpacity className="w-10 h-10 bg-gray-50 rounded-full items-center justify-center border border-gray-100 shadow-sm">
                            <Ionicons name="search-outline" size={20} color="#374151" />
                        </TouchableOpacity>
                        <TouchableOpacity className="w-10 h-10 bg-gray-50 rounded-full items-center justify-center border border-gray-100 shadow-sm">
                            <Ionicons name="notifications-outline" size={20} color="#374151" />
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Tabs */}
                <View className="bg-white pb-4 px-6 z-10 shadow-sm">
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 12 }}>
                        {["Closet", "Wishlist", "Inspiration", "Design Room"].map((tab) => (
                            <TouchableOpacity
                                key={tab}
                                onPress={() => setActiveTab(tab)}
                                className="rounded-full"
                            >
                                {activeTab === tab ? (
                                    <LinearGradient
                                        colors={['#8B5CF6', '#6366F1']}
                                        start={{ x: 0, y: 0 }}
                                        end={{ x: 1, y: 0 }}
                                        className="px-5 py-2 rounded-full"
                                    >
                                        <Text className="text-white font-bold text-sm">{tab}</Text>
                                    </LinearGradient>
                                ) : (
                                    <View className="px-5 py-2 rounded-full bg-gray-100 border border-gray-200">
                                        <Text className="text-gray-600 font-medium text-sm">{tab}</Text>
                                    </View>
                                )}
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                </View>

                <ScrollView className="flex-1" showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 100 }}>

                    {/* Welcome / Closet Section */}
                    <View className="px-6 py-6">
                        <View className="w-full bg-white rounded-[24px] overflow-hidden shadow-md shadow-purple-100 border border-purple-50">
                            <LinearGradient
                                colors={['#F3E8FF', '#FFFFFF']}
                                className="p-6 items-center"
                            >
                                <View className="w-24 h-24 bg-white rounded-full items-center justify-center mb-4 shadow-sm">
                                    <Ionicons name="shirt-outline" size={40} color="#8B5CF6" />
                                </View>
                                <Text className="text-xl font-bold text-gray-900 mb-2">Your Digital Closet</Text>
                                <Text className="text-center text-gray-500 mb-6 px-4 leading-5">
                                    Start building your virtual wardrobe. Add items to get personalized outfit ideas.
                                </Text>

                                <TouchableOpacity
                                    className="w-full shadow-md shadow-purple-200"
                                    onPress={() => (navigation as any).navigate("AddOutfit")}
                                >
                                    <LinearGradient
                                        colors={['#111827', '#374151']}
                                        className="w-full py-4 rounded-xl items-center"
                                    >
                                        <Text className="text-white font-bold text-base">+ Add New Item</Text>
                                    </LinearGradient>
                                </TouchableOpacity>
                            </LinearGradient>
                        </View>
                    </View>

                    {/* AI Tools Section */}
                    <View className="px-6 mb-8">
                        <Text className="text-lg font-bold text-gray-900 mb-4">AI Studio</Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false} className="-mx-6 px-6">
                            {features.map((feature, idx) => (
                                <TouchableOpacity
                                    key={idx}
                                    onPress={() => (navigation as any).navigate(feature.screen)}
                                    className="mr-4 w-36"
                                >
                                    <View
                                        className="w-36 h-40 rounded-2xl items-center justify-center mb-3 border border-white/50 shadow-sm"
                                        style={{ backgroundColor: ['#EFF6FF', '#FFF1F2', '#F0FDF4'][idx % 3] }}
                                    >
                                        <View className="w-14 h-14 bg-white rounded-full items-center justify-center mb-3 shadow-sm">
                                            <Ionicons
                                                name={idx === 0 ? "sparkles" : idx === 1 ? "shirt" : "camera"}
                                                size={28}
                                                color={['#3B82F6', '#EC4899', '#10B981'][idx % 3]}
                                            />
                                        </View>
                                        <Text className="text-center text-gray-900 font-bold text-sm px-2">{feature.title}</Text>
                                    </View>
                                </TouchableOpacity>
                            ))}
                        </ScrollView>
                    </View>

                    {/* Popular / Basics Section */}
                    <View className="px-6">
                        <View className="flex-row items-center justify-between mb-4">
                            <Text className="text-lg font-bold text-gray-900">Trending Now</Text>
                            <TouchableOpacity>
                                <Text className="text-purple-600 font-semibold text-sm">See All</Text>
                            </TouchableOpacity>
                        </View>

                        <View className="flex-row flex-wrap justify-between">
                            {popularItems.map((item, idx) => (
                                <Pressable key={idx} className="w-[48%] mb-6">
                                    <View className="bg-white rounded-2xl aspect-[3/4] mb-3 overflow-hidden shadow-sm border border-gray-100 relative">
                                        <Image
                                            source={{ uri: item.image }}
                                            className="w-full h-full"
                                            resizeMode="cover"
                                        />
                                        <LinearGradient
                                            colors={['transparent', 'rgba(0,0,0,0.3)']}
                                            className="absolute bottom-0 left-0 right-0 h-1/3"
                                        />
                                        <TouchableOpacity className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded-full p-2 shadow-sm">
                                            <Ionicons name="heart-outline" size={18} color="#EF4444" />
                                        </TouchableOpacity>
                                    </View>
                                    <Text className="font-bold text-gray-900 text-sm mb-1" numberOfLines={1}>{item.itemName}</Text>
                                    <View className="flex-row items-center">
                                        <Image
                                            source={{ uri: item.profile }}
                                            className="w-4 h-4 rounded-full mr-1.5"
                                        />
                                        <Text className="text-xs text-gray-500">{item.username}</Text>
                                    </View>
                                </Pressable>
                            ))}
                        </View>
                    </View>

                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

export default DiscoverScreen;
