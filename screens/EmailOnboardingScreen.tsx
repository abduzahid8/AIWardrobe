import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    ActivityIndicator,
    ScrollView,
    Alert,
    Linking
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { LinearGradient } from 'expo-linear-gradient';

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:3000';

interface ScanResult {
    receiptsScanned: number;
    receiptsFound: number;
    itemsDetected: number;
    items: any[];
}

/**
 * Email Onboarding Screen
 * Allows users to connect Gmail for automatic wardrobe ingestion
 * Competitive advantage vs Alta Daily
 */
const EmailOnboardingScreen = () => {
    const navigation = useNavigation();
    const [loading, setLoading] = useState(false);
    const [scanning, setScanning] = useState(false);
    const [connected, setConnected] = useState(false);
    const [scanResults, setScanResults] = useState<ScanResult | null>(null);
    const [userId, setUserId] = useState<string | null>(null);

    useEffect(() => {
        checkEmailStatus();
        getUserId();
    }, []);

    const getUserId = async () => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                // Decode JWT to get userId (or fetch from /me endpoint)
                const response = await axios.get(`${API_URL}/me`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setUserId(response.data._id);
            }
        } catch (error) {
            console.error('Error getting user ID:', error);
        }
    };

    const checkEmailStatus = async () => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            const response = await axios.get(`${API_URL}/api/email/status`, {
                headers: { Authorization: `Bearer ${token}` },
                params: { userId }
            });
            setConnected(response.data.connected);
        } catch (error) {
            console.error('Error checking email status:', error);
        }
    };

    const handleConnectEmail = async () => {
        setLoading(true);
        try {
            // Get OAuth URL from backend
            const response = await axios.get(`${API_URL}/api/email/auth-url`, {
                params: { userId }
            });

            const { authUrl } = response.data;

            // Open OAuth URL in browser
            const supported = await Linking.canOpenURL(authUrl);
            if (supported) {
                await Linking.openURL(authUrl);

                // Show message that user needs to authorize
                Alert.alert(
                    'Complete Authorization',
                    'Please complete the authorization in your browser, then come back to the app.',
                    [
                        {
                            text: 'I\'ve Authorized',
                            onPress: () => {
                                checkEmailStatus();
                                navigation.goBack();
                            }
                        }
                    ]
                );
            } else {
                Alert.alert('Error', 'Cannot open authorization URL');
            }
        } catch (error) {
            console.error('Error connecting email:', error);
            Alert.alert('Error', 'Failed to connect email. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleScanReceipts = async () => {
        if (!connected) {
            Alert.alert('Not Connected', 'Please connect your email first.');
            return;
        }

        setScanning(true);
        setScanResults(null);

        try {
            const token = await AsyncStorage.getItem('userToken');

            // Scan for receipts
            const response = await axios.post(
                `${API_URL}/api/email/scan-receipts`,
                {
                    userId,
                    maxResults: 100,
                    maxAge: '1y' // Scan past year
                },
                {
                    headers: { Authorization: `Bearer ${token}` },
                    timeout: 120000 // 2 minutes
                }
            );

            setScanResults(response.data);

            Alert.alert(
                'Scan Complete!',
                `Found ${response.data.itemsDetected} clothing items from ${response.data.receiptsFound} receipts.`,
                [
                    {
                        text: 'Import to Wardrobe',
                        onPress: () => handleImportItems(response.data.items)
                    },
                    {
                        text: 'Cancel',
                        style: 'cancel'
                    }
                ]
            );
        } catch (error) {
            console.error('Error scanning receipts:', error);
            Alert.alert('Error', 'Failed to scan receipts. Please try again.');
        } finally {
            setScanning(false);
        }
    };

    const handleImportItems = async (items: any[]) => {
        try {
            const token = await AsyncStorage.getItem('userToken');

            const response = await axios.post(
                `${API_URL}/api/email/import-items`,
                { userId, items },
                { headers: { Authorization: `Bearer ${token}` } }
            );

            Alert.alert(
                'Success!',
                `Imported ${response.data.itemsImported} items to your wardrobe.`,
                [
                    {
                        text: 'View Wardrobe',
                        onPress: () => (navigation as any).navigate('Home', { screen: 'Closet' })
                    }
                ]
            );
        } catch (error) {
            console.error('Error importing items:', error);
            Alert.alert('Error', 'Failed to import items. Please try again.');
        }
    };

    const handleSkip = () => {
        navigation.goBack();
    };

    return (
        <View className="flex-1 bg-white">
            <LinearGradient
                colors={['#8B5CF6', '#6366F1']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                className="absolute inset-0"
            />

            <SafeAreaView className="flex-1">
                <ScrollView className="flex-1 px-6">
                    {/* Header */}
                    <View className="mt-8 mb-6">
                        <View className="w-20 h-20 rounded-full bg-white/20 items-center justify-center mb-6">
                            <Ionicons name="mail-outline" size={40} color="white" />
                        </View>

                        <Text className="text-4xl font-bold text-white mb-3">
                            Auto-Fill Your Wardrobe
                        </Text>

                        <Text className="text-lg text-white/90">
                            Connect your email and we'll automatically find your clothing purchases in seconds.
                        </Text>
                    </View>

                    {/* Features List */}
                    <View className="bg-white/10 rounded-3xl p-6 mb-6 border border-white/20">
                        <Text className="text-xl font-bold text-white mb-4">How it works:</Text>

                        {[
                            { icon: '🔒', text: 'We only read purchase receipts (100% safe)' },
                            { icon: '⚡', text: 'Scan 1 year of purchases in 30 seconds' },
                            { icon: '📸', text: 'Auto-fetch product photos from retailers' },
                            { icon: '🎯', text: 'Skip manual upload - save hours of time' }
                        ].map((item, index) => (
                            <View key={index} className="flex-row items-start mb-3">
                                <Text className="text-2xl mr-3">{item.icon}</Text>
                                <Text className="text-white/90 text-base flex-1 pt-1">
                                    {item.text}
                                </Text>
                            </View>
                        ))}
                    </View>

                    {/* Connection Status */}
                    {connected && (
                        <View className="bg-green-500/20 border border-green-300/30 rounded-2xl p-4 mb-6 flex-row items-center">
                            <Ionicons name="checkmark-circle" size={24} color="#10B981" />
                            <Text className="text-green-100 ml-3 flex-1">
                                Gmail Connected Successfully
                            </Text>
                        </View>
                    )}

                    {/* Scan Results */}
                    {scanResults && (
                        <View className="bg-white/10 rounded-2xl p-5 mb-6 border border-white/20">
                            <Text className="text-white font-bold text-lg mb-3">
                                Scan Results
                            </Text>
                            <Text className="text-white/90 text-base mb-2">
                                📧 Emails scanned: {scanResults.receiptsScanned}
                            </Text>
                            <Text className="text-white/90 text-base mb-2">
                                🧾 Receipts found: {scanResults.receiptsFound}
                            </Text>
                            <Text className="text-white/90 text-base">
                                👔 Items detected: {scanResults.itemsDetected}
                            </Text>
                        </View>
                    )}

                    {/* Action Buttons */}
                    <View className="space-y-4 mb-8">
                        {!connected ? (
                            <TouchableOpacity
                                onPress={handleConnectEmail}
                                disabled={loading}
                                className="bg-white rounded-2xl py-4 px-6 shadow-lg"
                            >
                                {loading ? (
                                    <ActivityIndicator color="#8B5CF6" />
                                ) : (
                                    <View className="flex-row items-center justify-center">
                                        <Ionicons name="logo-google" size={20} color="#8B5CF6" />
                                        <Text className="text-purple-600 font-bold text-lg ml-3">
                                            Connect Gmail
                                        </Text>
                                    </View>
                                )}
                            </TouchableOpacity>
                        ) : (
                            <TouchableOpacity
                                onPress={handleScanReceipts}
                                disabled={scanning}
                                className="bg-white rounded-2xl py-4 px-6 shadow-lg"
                            >
                                {scanning ? (
                                    <View>
                                        <ActivityIndicator color="#8B5CF6" size="small" />
                                        <Text className="text-purple-600 text-center mt-2 font-medium">
                                            Scanning receipts...
                                        </Text>
                                    </View>
                                ) : (
                                    <View className="flex-row items-center justify-center">
                                        <Ionicons name="scan-outline" size={20} color="#8B5CF6" />
                                        <Text className="text-purple-600 font-bold text-lg ml-3">
                                            Scan My Receipts
                                        </Text>
                                    </View>
                                )}
                            </TouchableOpacity>
                        )}

                        <TouchableOpacity
                            onPress={handleSkip}
                            className="py-4"
                        >
                            <Text className="text-white text-center text-base font-medium">
                                I'll upload manually instead
                            </Text>
                        </TouchableOpacity>
                    </View>

                    {/* Privacy Notice */}
                    <View className="bg-white/5 rounded-xl p-4 mb-8 border border-white/10">
                        <Text className="text-white/70 text-xs leading-5">
                            🔒 Privacy: We only read emails containing clothing purchases.
                            Your data is encrypted and never shared. You can disconnect anytime.
                        </Text>
                    </View>
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

export default EmailOnboardingScreen;
