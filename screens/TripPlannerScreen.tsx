import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Alert,
    FlatList
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { LinearGradient } from 'expo-linear-gradient';
import DateTimePicker from '@react-native-community/datetimepicker';

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3000';

interface WeatherDay {
    date: string;
    tempHigh: number;
    tempLow: number;
    condition: string;
    description: string;
    icon: string;
}

interface PackingItem {
    _id: string;
    itemType: string;
    color: string;
    imageUrl?: string;
    uses: number;
}

interface TripPlan {
    destination: string;
    weather: WeatherDay[];
    packingList: PackingItem[];
    outfitsByDay: any[];
    stats: {
        totalItems: number;
        totalOutfits: number;
        daysPlanned: number;
    };
}

const TripPlannerScreen = () => {
    const navigation = useNavigation();
    const [step, setStep] = useState(1); // 1=input, 2=loading, 3=results

    // Form inputs
    const [destination, setDestination] = useState('');
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000));
    const [occasions, setOccasions] = useState<string[]>(['casual']);
    const [showStartPicker, setShowStartPicker] = useState(false);
    const [showEndPicker, setShowEndPicker] = useState(false);

    // Results
    const [loading, setLoading] = useState(false);
    const [tripPlan, setTripPlan] = useState<TripPlan | null>(null);
    const [userId, setUserId] = useState<string | null>(null);

    useEffect(() => {
        getUserId();
    }, []);

    const getUserId = async () => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                const response = await axios.get(`${API_URL}/me`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setUserId(response.data._id);
            }
        } catch (error) {
            console.error('Error getting user ID:', error);
        }
    };

    const occasionOptions = [
        { id: 'casual', emoji: '👕', label: 'Casual' },
        { id: 'business', emoji: '💼', label: 'Business' },
        { id: 'formal', emoji: '👔', label: 'Formal' },
        { id: 'beach', emoji: '🏖️', label: 'Beach' },
        { id: 'sport', emoji: '⚽', label: 'Sport' },
        { id: 'party', emoji: '🎉', label: 'Party' }
    ];

    const toggleOccasion = (occasionId: string) => {
        setOccasions(prev =>
            prev.includes(occasionId)
                ? prev.filter(o => o !== occasionId)
                : [...prev, occasionId]
        );
    };

    const handleCreatePlan = async () => {
        if (!destination.trim()) {
            Alert.alert('Error', 'Please enter a destination');
            return;
        }

        if (!userId) {
            Alert.alert('Error', 'Please log in to create a trip plan');
            return;
        }

        setLoading(true);
        setStep(2);

        try {
            const response = await axios.post(
                `${API_URL}/api/trip-planner/create`,
                {
                    userId,
                    destination: destination.trim(),
                    startDate: startDate.toISOString().split('T')[0],
                    endDate: endDate.toISOString().split('T')[0],
                    occasions
                },
                {
                    timeout: 30000 // 30 seconds
                }
            );

            setTripPlan(response.data);
            setStep(3);
        } catch (error: any) {
            console.error('Trip planning error:', error);
            Alert.alert(
                'Error',
                error.response?.data?.message || 'Failed to create trip plan. Please try again.'
            );
            setStep(1);
        } finally {
            setLoading(false);
        }
    };

    const renderStepIndicator = () => (
        <View className="flex-row justify-center items-center mb-6">
            {[1, 2, 3].map(num => (
                <View key={num} className="flex-row items-center">
                    <View className={`w-8 h-8 rounded-full items-center justify-center ${step >= num ? 'bg-purple-600' : 'bg-gray-300'
                        }`}>
                        <Text className={`font-bold ${step >= num ? 'text-white' : 'text-gray-600'}`}>
                            {num}
                        </Text>
                    </View>
                    {num < 3 && (
                        <View className={`w-12 h-1 ${step > num ? 'bg-purple-600' : 'bg-gray-300'}`} />
                    )}
                </View>
            ))}
        </View>
    );

    const renderInputStep = () => (
        <ScrollView className="flex-1 px-6">
            {/* Destination */}
            <Text className="text-white text-lg font-bold mb-2">📍 Destination</Text>
            <TextInput
                className="bg-white/10 border border-white/20 rounded-2xl px-4 py-3 text-white mb-4"
                placeholder="e.g. Miami, Paris, Tokyo"
                placeholderTextColor="rgba(255,255,255,0.5)"
                value={destination}
                onChangeText={setDestination}
            />

            {/* Dates */}
            <Text className="text-white text-lg font-bold mb-2">📅 Travel Dates</Text>
            <View className="flex-row gap-2 mb-4">
                <TouchableOpacity
                    onPress={() => setShowStartPicker(true)}
                    className="flex-1 bg-white/10 border border-white/20 rounded-2xl px-4 py-3"
                >
                    <Text className="text-white/70 text-xs">Start Date</Text>
                    <Text className="text-white text-base">{startDate.toLocaleDateString()}</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    onPress={() => setShowEndPicker(true)}
                    className="flex-1 bg-white/10 border border-white/20 rounded-2xl px-4 py-3"
                >
                    <Text className="text-white/70 text-xs">End Date</Text>
                    <Text className="text-white text-base">{endDate.toLocaleDateString()}</Text>
                </TouchableOpacity>
            </View>

            {showStartPicker && (
                <DateTimePicker
                    value={startDate}
                    mode="date"
                    display="default"
                    onChange={(event, date) => {
                        setShowStartPicker(false);
                        if (date) setStartDate(date);
                    }}
                />
            )}

            {showEndPicker && (
                <DateTimePicker
                    value={endDate}
                    mode="date"
                    display="default"
                    minimumDate={startDate}
                    onChange={(event, date) => {
                        setShowEndPicker(false);
                        if (date) setEndDate(date);
                    }}
                />
            )}

            {/* Occasions */}
            <Text className="text-white text-lg font-bold mb-2">🎯 Occasions</Text>
            <View className="flex-row flex-wrap gap-2 mb-6">
                {occasionOptions.map(option => (
                    <TouchableOpacity
                        key={option.id}
                        onPress={() => toggleOccasion(option.id)}
                        className={`px-4 py-2 rounded-full flex-row items-center ${occasions.includes(option.id)
                                ? 'bg-purple-600'
                                : 'bg-white/10 border border-white/20'
                            }`}
                    >
                        <Text className="text-lg mr-2">{option.emoji}</Text>
                        <Text className="text-white">{option.label}</Text>
                    </TouchableOpacity>
                ))}
            </View>

            {/* Create Button */}
            <TouchableOpacity
                onPress={handleCreatePlan}
                disabled={loading}
                className="bg-white rounded-2xl py-4 mb-8"
            >
                {loading ? (
                    <ActivityIndicator color="#8B5CF6" />
                ) : (
                    <Text className="text-purple-600 font-bold text-lg text-center">
                        ✨ Create Trip Plan
                    </Text>
                )}
            </TouchableOpacity>
        </ScrollView>
    );

    const renderLoadingStep = () => (
        <View className="flex-1 justify-center items-center px-6">
            <ActivityIndicator size="large" color="white" />
            <Text className="text-white text-xl font-bold mt-4 mb-2">
                Planning your trip...
            </Text>
            <Text className="text-white/70 text-center">
                Checking weather forecast and selecting perfect outfits
            </Text>
        </View>
    );

    const renderResultsStep = () => {
        if (!tripPlan) return null;

        return (
            <ScrollView className="flex-1 px-6">
                {/* Stats Summary */}
                <View className="bg-white/10 rounded-3xl p-6 mb-6 border border-white/20">
                    <Text className="text-white text-2xl font-bold mb-4">
                        📦 Your Packing List
                    </Text>
                    <View className="flex-row justify-around">
                        <View className="items-center">
                            <Text className="text-white/70 text-sm">Items to Pack</Text>
                            <Text className="text-white text-3xl font-bold">
                                {tripPlan.stats.totalItems}
                            </Text>
                        </View>
                        <View className="items-center">
                            <Text className="text-white/70 text-sm">Total Outfits</Text>
                            <Text className="text-white text-3xl font-bold">
                                {tripPlan.stats.totalOutfits}
                            </Text>
                        </View>
                        <View className="items-center">
                            <Text className="text-white/70 text-sm">Days</Text>
                            <Text className="text-white text-3xl font-bold">
                                {tripPlan.stats.daysPlanned}
                            </Text>
                        </View>
                    </View>
                </View>

                {/* Packing List */}
                <Text className="text-white text-xl font-bold mb-3">Items to Pack</Text>
                <View className="bg-white/10 rounded-2xl p-4 mb-6 border border-white/20">
                    {tripPlan.packingList.map((item, index) => (
                        <View
                            key={item._id}
                            className={`flex-row items-center py-3 ${index < tripPlan.packingList.length - 1 ? 'border-b border-white/10' : ''
                                }`}
                        >
                            <View className="w-12 h-12 rounded-xl bg-white/20 items-center justify-center mr-3">
                                <Text className="text-2xl">👕</Text>
                            </View>
                            <View className="flex-1">
                                <Text className="text-white font-medium">
                                    {item.color} {item.itemType}
                                </Text>
                                <Text className="text-white/60 text-sm">
                                    Wear {item.uses} time{item.uses > 1 ? 's' : ''}
                                </Text>
                            </View>
                        </View>
                    ))}
                </View>

                {/* Day-by-Day Outfits */}
                <Text className="text-white text-xl font-bold mb-3">Daily Outfits</Text>
                {tripPlan.outfitsByDay.map((day, index) => (
                    <View
                        key={day.date}
                        className="bg-white/10 rounded-2xl p-4 mb-4 border border-white/20"
                    >
                        <View className="flex-row justify-between items-center mb-3">
                            <Text className="text-white font-bold text-lg">
                                Day {index + 1} - {new Date(day.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                            </Text>
                            <View className="bg-white/20 px-3 py-1 rounded-full">
                                <Text className="text-white text-sm">
                                    {day.weather.tempHigh}°C {day.weather.condition}
                                </Text>
                            </View>
                        </View>

                        {day.outfits.map((outfit: any, oIndex: number) => (
                            <View key={oIndex} className="mt-2">
                                <Text className="text-white/70 text-sm mb-1">
                                    {outfit.occasion.charAt(0).toUpperCase() + outfit.occasion.slice(1)}
                                </Text>
                                <Text className="text-white">
                                    {outfit.items.map((item: any) => `${item.color} ${item.itemType}`).join(' + ')}
                                </Text>
                            </View>
                        ))}
                    </View>
                ))}

                {/* Actions */}
                <TouchableOpacity
                    onPress={() => {
                        setStep(1);
                        setTripPlan(null);
                    }}
                    className="bg-white/20 rounded-2xl py-4 mb-8 border border-white/30"
                >
                    <Text className="text-white font-bold text-lg text-center">
                        Plan Another Trip
                    </Text>
                </TouchableOpacity>
            </ScrollView>
        );
    };

    return (
        <View className="flex-1 bg-black">
            <LinearGradient
                colors={['#6366F1', '#8B5CF6', '#EC4899']}
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
                    <Text className="text-white text-xl font-bold">Trip Planner</Text>
                    <View style={{ width: 24 }} />
                </View>

                {/* Step Indicator */}
                <View className="px-6 pt-4">
                    {renderStepIndicator()}
                </View>

                {/* Content */}
                {step === 1 && renderInputStep()}
                {step === 2 && renderLoadingStep()}
                {step === 3 && renderResultsStep()}
            </SafeAreaView>
        </View>
    );
};

export default TripPlannerScreen;
