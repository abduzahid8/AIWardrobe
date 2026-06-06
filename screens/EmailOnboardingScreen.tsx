import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, ActivityIndicator, ScrollView, Alert, Linking } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import useAuthStore from '../store/auth';
import { useTranslation } from 'react-i18next';

// API_URL removed

// Scanned email item type
interface ScannedEmailItem {
    id?: string;
    name: string;
    brand?: string;
    category?: string;
    color?: string;
    price?: number;
    imageUrl?: string;
    purchaseDate?: string;
}

interface ScanResult {
    receiptsScanned: number;
    receiptsFound: number;
    itemsDetected: number;
    items: ScannedEmailItem[];
}

/**
 * Email Onboarding Screen
 * Allows users to connect Gmail for automatic wardrobe ingestion
 * Competitive advantage vs Alta Daily
 */
const EmailOnboardingScreen = () => {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const { user } = useAuthStore();
    const [loading, setLoading] = useState(false);
    const [scanning, setScanning] = useState(false);
    const [connected, setConnected] = useState(false);
    const [scanResults, setScanResults] = useState<ScanResult | null>(null);

    useEffect(() => {
        // user is already available from store
    }, []);

    const checkEmailStatus = async () => {
        // Mock status check
        // In future, this would check a Supabase 'integrations' table
    };

    const handleConnectEmail = async () => {
        setLoading(true);
        try {
            // Mock OAuth flow
            await new Promise(resolve => setTimeout(resolve, 1500));
            setConnected(true);
            Alert.alert(t('common.success'), t('emailOnboarding.gmailConnected'));
        } catch (error) {
            console.error('Error connecting email:', error);
            Alert.alert(t('common.error'), t('emailOnboarding.failedConnectEmail'));
        } finally {
            setLoading(false);
        }
    };

    const handleScanReceipts = async () => {
        if (!connected) {
            Alert.alert(t('emailOnboarding.notConnected'), t('emailOnboarding.connectEmailFirst'));
            return;
        }

        setScanning(true);
        setScanResults(null);

        try {
            // Mock scanning process
            await new Promise(resolve => setTimeout(resolve, 3000));

            const results = {
                receiptsScanned: 154,
                receiptsFound: 12,
                itemsDetected: 5,
                items: [
                    { name: "Mock Item 1", category: "Shirt" },
                    { name: "Mock Item 2", category: "Pants" }
                ]
            };

            setScanResults(results as any);

            Alert.alert(
                t('emailOnboarding.scanComplete'),
                t('emailOnboarding.scanCompleteMessage', { itemsDetected: results.itemsDetected, receiptsFound: results.receiptsFound }),
                [
                    {
                        text: t('emailOnboarding.importToWardrobe'),
                        onPress: () => handleImportItems(results.items as any[])
                    },
                    {
                        text: t('common.cancel'),
                        style: 'cancel'
                    }
                ]
            );
        } catch (error) {
            console.error('Error scanning receipts:', error);
            Alert.alert(t('common.error'), t('emailOnboarding.failedScanReceipts'));
        } finally {
            setScanning(false);
        }
    };

    const handleImportItems = async (items: ScannedEmailItem[]) => {
        try {
            // Mock import
            await new Promise(resolve => setTimeout(resolve, 1000));

            Alert.alert(
                'Success!',
                `Imported ${items.length} items to your wardrobe.`,
                [
                    {
                        text: t('emailOnboarding.viewWardrobe'),
                        onPress: () => (navigation as any).navigate('Home', { screen: 'Closet' })
                    }
                ]
            );
        } catch (error) {
            console.error('Error importing items:', error);
            Alert.alert(t('common.error'), t('emailOnboarding.failedImportItems'));
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

                        <ScaledText className="text-4xl font-bold text-white mb-3">
                            Auto-Fill Your Wardrobe
                        </ScaledText>

                        <ScaledText className="text-lg text-white/90">
                            Connect your email and we'll automatically find your clothing purchases in seconds.
                        </ScaledText>
                    </View>

                    {/* Features List */}
                    <View className="bg-white/10 rounded-3xl p-6 mb-6 border border-white/20">
                        <ScaledText className="text-xl font-bold text-white mb-4">{t('emailOnboarding.howItWorks')}</ScaledText>

                        {[
                            { icon: '🔒', text: 'We only read purchase receipts (100% safe)' },
                            { icon: '⚡', text: 'Scan 1 year of purchases in 30 seconds' },
                            { icon: '📸', text: 'Auto-fetch product photos from retailers' },
                            { icon: '🎯', text: 'Skip manual upload - save hours of time' }
                        ].map((item, index) => (
                            <View key={index} className="flex-row items-start mb-3">
                                <ScaledText className="text-2xl mr-3">{item.icon}</ScaledText>
                                <ScaledText className="text-white/90 text-base flex-1 pt-1">
                                    {item.text}
                                </ScaledText>
                            </View>
                        ))}
                    </View>

                    {/* Connection Status */}
                    {connected && (
                        <View className="bg-green-500/20 border border-green-300/30 rounded-2xl p-4 mb-6 flex-row items-center">
                            <Ionicons name="checkmark-circle" size={24} color="#10B981" />
                            <ScaledText className="text-green-100 ml-3 flex-1">
                                Gmail Connected Successfully
                            </ScaledText>
                        </View>
                    )}

                    {/* Scan Results */}
                    {scanResults && (
                        <View className="bg-white/10 rounded-2xl p-5 mb-6 border border-white/20">
                            <ScaledText className="text-white font-bold text-lg mb-3">
                                Scan Results
                            </ScaledText>
                            <ScaledText className="text-white/90 text-base mb-2">
                                📧 Emails scanned: {scanResults.receiptsScanned}
                            </ScaledText>
                            <ScaledText className="text-white/90 text-base mb-2">
                                🧾 Receipts found: {scanResults.receiptsFound}
                            </ScaledText>
                            <ScaledText className="text-white/90 text-base">
                                👔 Items detected: {scanResults.itemsDetected}
                            </ScaledText>
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
                                        <ScaledText className="text-purple-600 font-bold text-lg ml-3">
                                            Connect Gmail
                                        </ScaledText>
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
                                        <ScaledText className="text-purple-600 text-center mt-2 font-medium">
                                            Scanning receipts...
                                        </ScaledText>
                                    </View>
                                ) : (
                                    <View className="flex-row items-center justify-center">
                                        <Ionicons name="scan-outline" size={20} color="#8B5CF6" />
                                        <ScaledText className="text-purple-600 font-bold text-lg ml-3">
                                            Scan My Receipts
                                        </ScaledText>
                                    </View>
                                )}
                            </TouchableOpacity>
                        )}

                        <TouchableOpacity
                            onPress={handleSkip}
                            className="py-4"
                        >
                            <ScaledText className="text-white text-center text-base font-medium">
                                I'll upload manually instead
                            </ScaledText>
                        </TouchableOpacity>
                    </View>

                    {/* Privacy Notice */}
                    <View className="bg-white/5 rounded-xl p-4 mb-8 border border-white/10">
                        <ScaledText className="text-white/70 text-xs leading-5">
                            🔒 Privacy: We only read emails containing clothing purchases.
                            Your data is encrypted and never shared. You can disconnect anytime.
                        </ScaledText>
                    </View>
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

export default EmailOnboardingScreen;
