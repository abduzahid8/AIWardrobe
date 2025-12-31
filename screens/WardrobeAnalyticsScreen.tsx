import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    FlatList
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { LinearGradient } from 'expo-linear-gradient';

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3000';

interface Analytics {
    totalInvested: string;
    totalWears: number;
    averageCPW: string;
    zombieItems: any[];
    bestROI: any[];
    mostWorn: any[];
    itemCount: number;
    stats: {
        itemsWithPrice: number;
        itemsNeverWorn: number;
        itemsWornOnce: number;
        itemsWorn5Plus: number;
    };
}

const WardrobeAnalyticsScreen = () => {
    const navigation = useNavigation();
    const [loading, setLoading] = useState(true);
    const [analytics, setAnalytics] = useState<Analytics | null>(null);
    const [userId, setUserId] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'roi' | 'zombie'>('overview');

    useEffect(() => {
        getUserIdAndFetchAnalytics();
    }, []);

    const getUserIdAndFetchAnalytics = async () => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                const response = await axios.get(`${API_URL}/me`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const id = response.data._id;
                setUserId(id);
                await fetchAnalytics(id);
            }
        } catch (error) {
            console.error('Error:', error);
            setLoading(false);
        }
    };

    const fetchAnalytics = async (id: string) => {
        try {
            setLoading(true);
            const response = await axios.get(`${API_URL}/api/wardrobe-analytics/${id}`);
            setAnalytics(response.data);
        } catch (error) {
            console.error('Analytics fetch error:', error);
        } finally {
            setLoading(false);
        }
    };

    const getCPWColor = (cpw: number) => {
        if (cpw < 5) return '#10B981'; // Green
        if (cpw < 10) return '#F59E0B'; // Yellow
        return '#EF4444'; // Red
    };

    const renderOverview = () => (
        <ScrollView className="flex-1 px-6">
            {/* Key Metrics */}
            <View className="flex-row flex-wrap gap-3 mb-6">
                <View className="flex-1 min-w-[45%] bg-white/10 rounded-2xl p-4 border border-white/20">
                    <Text className="text-white/70 text-sm mb-1">Total Invested</Text>
                    <Text className="text-white text-2xl font-bold">
                        ${analytics?.totalInvested || '0.00'}
                    </Text>
                </View>

                <View className="flex-1 min-w-[45%] bg-white/10 rounded-2xl p-4 border border-white/20">
                    <Text className="text-white/70 text-sm mb-1">Total Wears</Text>
                    <Text className="text-white text-2xl font-bold">
                        {analytics?.totalWears || 0}
                    </Text>
                </View>

                <View className="flex-1 min-w-[45%] bg-white/10 rounded-2xl p-4 border border-white/20">
                    <Text className="text-white/70 text-sm mb-1">Avg Cost/Wear</Text>
                    <Text className="text-white text-2xl font-bold">
                        ${analytics?.averageCPW || '0.00'}
                    </Text>
                </View>

                <View className="flex-1 min-w-[45%] bg-white/10 rounded-2xl p-4 border border-white/20">
                    <Text className="text-white/70 text-sm mb-1">Items</Text>
                    <Text className="text-white text-2xl font-bold">
                        {analytics?.itemCount || 0}
                    </Text>
                </View>
            </View>

            {/* Usage Stats */}
            <View className="bg-white/10 rounded-2xl p-5 mb-6 border border-white/20">
                <Text className="text-white text-lg font-bold mb-4">Usage Breakdown</Text>

                <View className="space-y-3">
                    <View className="flex-row justify-between items-center">
                        <Text className="text-white/70">Never Worn</Text>
                        <View className="flex-row items-center">
                            <Text className="text-red-400 font-bold mr-2">
                                {analytics?.stats.itemsNeverWorn || 0}
                            </Text>
                            <Text className="text-white/50">
                                ({analytics?.itemCount ?
                                    ((analytics.stats.itemsNeverWorn / analytics.itemCount) * 100).toFixed(0) : 0}%)
                            </Text>
                        </View>
                    </View>

                    <View className="flex-row justify-between items-center">
                        <Text className="text-white/70">Worn Once</Text>
                        <View className="flex-row items-center">
                            <Text className="text-yellow-400 font-bold mr-2">
                                {analytics?.stats.itemsWornOnce || 0}
                            </Text>
                            <Text className="text-white/50">
                                ({analytics?.itemCount ?
                                    ((analytics.stats.itemsWornOnce / analytics.itemCount) * 100).toFixed(0) : 0}%)
                            </Text>
                        </View>
                    </View>

                    <View className="flex-row justify-between items-center">
                        <Text className="text-white/70">Worn 5+ Times</Text>
                        <View className="flex-row items-center">
                            <Text className="text-green-400 font-bold mr-2">
                                {analytics?.stats.itemsWorn5Plus || 0}
                            </Text>
                            <Text className="text-white/50">
                                ({analytics?.itemCount ?
                                    ((analytics.stats.itemsWorn5Plus / analytics.itemCount) * 100).toFixed(0) : 0}%)
                            </Text>
                        </View>
                    </View>
                </View>
            </View>

            {/* Most Worn Items */}
            <Text className="text-white text-lg font-bold mb-3">⭐ Most Worn Items</Text>
            <View className="bg-white/10 rounded-2xl p-4 mb-6 border border-white/20">
                {analytics?.mostWorn.slice(0, 5).map((item, index) => (
                    <View
                        key={item.itemId}
                        className={`flex-row items-center py-3 ${index < 4 ? 'border-b border-white/10' : ''
                            }`}
                    >
                        <View className="w-10 h-10 rounded-xl bg-purple-500/30 items-center justify-center mr-3">
                            <Text className="text-white font-bold">#{index + 1}</Text>
                        </View>
                        <View className="flex-1">
                            <Text className="text-white font-medium">{item.name}</Text>
                            <Text className="text-white/60 text-sm">
                                Worn {item.wearCount} times
                            </Text>
                        </View>
                    </View>
                ))}
            </View>
        </ScrollView>
    );

    const renderBestROI = () => (
        <ScrollView className="flex-1 px-6">
            <Text className="text-white/70 text-sm mb-4">
                Items with the best return on investment (lowest $/wear)
            </Text>

            {analytics?.bestROI.length === 0 ? (
                <View className="bg-white/10 rounded-2xl p-8 items-center border border-white/20">
                    <Text className="text-white/70 text-center">
                        Start wearing your clothes to see ROI metrics!
                    </Text>
                </View>
            ) : (
                analytics?.bestROI.map((item, index) => (
                    <View
                        key={item.itemId}
                        className="bg-white/10 rounded-2xl p-4 mb-3 border border-white/20"
                    >
                        <View className="flex-row justify-between items-center mb-2">
                            <Text className="text-white font-bold text-lg">{item.name}</Text>
                            <View className="bg-green-500/30 px-3 py-1 rounded-full">
                                <Text className="text-green-300 font-bold">
                                    ${item.cpw}/wear
                                </Text>
                            </View>
                        </View>
                        <View className="flex-row justify-between">
                            <Text className="text-white/60">
                                Price: ${item.price?.toFixed(2)}
                            </Text>
                            <Text className="text-white/60">
                                Worn {item.wearCount} times
                            </Text>
                        </View>
                    </View>
                ))
            )}
        </ScrollView>
    );

    const renderZombieItems = () => (
        <ScrollView className="flex-1 px-6">
            <Text className="text-white/70 text-sm mb-4">
                Items never worn or not worn in 90+ days. Consider selling or donating!
            </Text>

            {analytics?.zombieItems.length === 0 ? (
                <View className="bg-white/10 rounded-2xl p-8 items-center border border-white/20">
                    <Text className="text-4xl mb-2">🎉</Text>
                    <Text className="text-white/70 text-center">
                        Great job! No zombie items in your wardrobe.
                    </Text>
                </View>
            ) : (
                analytics?.zombieItems.map(item => (
                    <View
                        key={item.itemId}
                        className="bg-white/10 rounded-2xl p-4 mb-3 border border-red-500/30"
                    >
                        <View className="flex-row justify-between items-center mb-2">
                            <Text className="text-white font-bold text-lg">{item.name}</Text>
                            {item.wearCount === 0 && (
                                <View className="bg-red-500/30 px-3 py-1 rounded-full">
                                    <Text className="text-red-300 font-bold text-xs">
                                        NEVER WORN
                                    </Text>
                                </View>
                            )}
                        </View>
                        <View className="flex-row justify-between text-sm">
                            <Text className="text-white/60">
                                {item.price ? `Wasted: $${item.price.toFixed(2)}` : 'No price'}
                            </Text>
                            <Text className="text-white/60">
                                {item.daysSincePurchase ? `${item.daysSincePurchase} days old` : ''}
                            </Text>
                        </View>
                    </View>
                ))
            )}

            {analytics && analytics.zombieItems.length > 0 && (
                <View className="bg-yellow-500/20 rounded-2xl p-4 mt-4 border border-yellow-500/30">
                    <Text className="text-yellow-200 text-sm">
                        💡 Tip: Wearing zombie items just 3 times each could save you $
                        {(analytics.zombieItems.reduce((sum, i) => sum + (i.price || 0), 0) * 0.3).toFixed(0)}
                        {' '}in future purchases!
                    </Text>
                </View>
            )}
        </ScrollView>
    );

    if (loading) {
        return (
            <View className="flex-1 bg-black items-center justify-center">
                <ActivityIndicator size="large" color="#8B5CF6" />
                <Text className="text-white mt-4">Loading analytics...</Text>
            </View>
        );
    }

    return (
        <View className="flex-1 bg-black">
            <LinearGradient
                colors={['#1E1B4B', '#312E81', '#1F2937']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                className="absolute inset-0"
            />

            <SafeAreaView className="flex-1">
                {/* Header */}
                <View className="px-6 pt-4 pb-2 flex-row justify-between items-center">
                    <TouchableOpacity onPress={() => navigation.goBack()}>
                        <Ionicons name="arrow-back" size={24} color="white" />
                    </TouchableOpacity>
                    <Text className="text-white text-xl font-bold">Wardrobe Analytics</Text>
                    <TouchableOpacity onPress={() => fetchAnalytics(userId!)}>
                        <Ionicons name="refresh" size={24} color="white" />
                    </TouchableOpacity>
                </View>

                {/* Tab Selector */}
                <View className="px-6 pt-4 pb-2 flex-row gap-2">
                    <TouchableOpacity
                        onPress={() => setActiveTab('overview')}
                        className={`flex-1 py-3 rounded-xl ${activeTab === 'overview' ? 'bg-purple-600' : 'bg-white/10'
                            }`}
                    >
                        <Text className={`text-center font-bold ${activeTab === 'overview' ? 'text-white' : 'text-white/60'
                            }`}>
                            Overview
                        </Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        onPress={() => setActiveTab('roi')}
                        className={`flex-1 py-3 rounded-xl ${activeTab === 'roi' ? 'bg-purple-600' : 'bg-white/10'
                            }`}
                    >
                        <Text className={`text-center font-bold ${activeTab === 'roi' ? 'text-white' : 'text-white/60'
                            }`}>
                            Best ROI
                        </Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        onPress={() => setActiveTab('zombie')}
                        className={`flex-1 py-3 rounded-xl ${activeTab === 'zombie' ? 'bg-purple-600' : 'bg-white/10'
                            }`}
                    >
                        <Text className={`text-center font-bold ${activeTab === 'zombie' ? 'text-white' : 'text-white/60'
                            }`}>
                            🧟 Zombie Items
                        </Text>
                    </TouchableOpacity>
                </View>

                {/* Content */}
                <View className="flex-1 pt-4">
                    {activeTab === 'overview' && renderOverview()}
                    {activeTab === 'roi' && renderBestROI()}
                    {activeTab === 'zombie' && renderZombieItems()}
                </View>
            </SafeAreaView>
        </View>
    );
};

export default WardrobeAnalyticsScreen;
