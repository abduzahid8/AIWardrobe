import React, { useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    Modal,
    StyleSheet,
    Pressable,
    Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';

export type Language = 'en' | 'ru' | 'uz';

interface LanguageOption {
    code: Language;
    name: string;
    nativeName: string;
    flag: string;
}

const languages: LanguageOption[] = [
    { code: 'en', name: 'English', nativeName: 'English', flag: '🇬🇧' },
    { code: 'ru', name: 'Russian', nativeName: 'Русский', flag: '🇷🇺' },
    { code: 'uz', name: 'Uzbek', nativeName: 'O\'zbek', flag: '🇺🇿' },
];

const LanguageSelector = () => {
    const { i18n, t } = useTranslation();
    const [modalVisible, setModalVisible] = useState(false);
    const [fadeAnim] = useState(new Animated.Value(0));

    const selectedLanguage = i18n.language as Language;

    const openModal = () => {
        setModalVisible(true);
        Animated.timing(fadeAnim, {
            toValue: 1,
            duration: 200,
            useNativeDriver: true,
        }).start();
    };

    const closeModal = () => {
        Animated.timing(fadeAnim, {
            toValue: 0,
            duration: 200,
            useNativeDriver: true,
        }).start(() => {
            setModalVisible(false);
        });
    };

    const handleLanguageSelect = (language: Language) => {
        i18n.changeLanguage(language);
        closeModal();
    };

    const currentLanguage = languages.find((lang) => lang.code === selectedLanguage);

    return (
        <View>
            {/* Language Button */}
            <TouchableOpacity
                onPress={openModal}
                className="flex-row items-center bg-gray-100 px-3 py-2 rounded-full"
            >
                <Text className="text-lg mr-1">{currentLanguage?.flag}</Text>
                <Text className="text-xs font-semibold text-gray-700 mr-1">
                    {currentLanguage?.code.toUpperCase()}
                </Text>
                <Ionicons name="chevron-down" size={12} color="#374151" />
            </TouchableOpacity>

            {/* Language Selection Modal */}
            <Modal
                transparent
                visible={modalVisible}
                animationType="none"
                onRequestClose={closeModal}
            >
                <Pressable
                    style={styles.modalOverlay}
                    onPress={closeModal}
                >
                    <Animated.View
                        style={[
                            styles.modalContent,
                            {
                                opacity: fadeAnim,
                                transform: [
                                    {
                                        scale: fadeAnim.interpolate({
                                            inputRange: [0, 1],
                                            outputRange: [0.9, 1],
                                        }),
                                    },
                                ],
                            },
                        ]}
                    >
                        <Pressable>
                            <View className="bg-white rounded-2xl p-4 shadow-lg">
                                <View className="flex-row items-center justify-between mb-4">
                                    <Text className="text-lg font-bold text-gray-800">
                                        {t('language.selectLanguage')}
                                    </Text>
                                    <TouchableOpacity onPress={closeModal}>
                                        <Ionicons name="close" size={24} color="#6B7280" />
                                    </TouchableOpacity>
                                </View>

                                {languages.map((language) => (
                                    <TouchableOpacity
                                        key={language.code}
                                        onPress={() => handleLanguageSelect(language.code)}
                                        className={`flex-row items-center justify-between p-4 rounded-xl mb-2 ${selectedLanguage === language.code
                                            ? 'bg-blue-50 border-2 border-blue-500'
                                            : 'bg-gray-50'
                                            }`}
                                    >
                                        <View className="flex-row items-center">
                                            <Text className="text-2xl mr-3">{language.flag}</Text>
                                            <View>
                                                <Text className="text-base font-semibold text-gray-800">
                                                    {language.nativeName}
                                                </Text>
                                                <Text className="text-xs text-gray-500">
                                                    {t(`language.${language.code === 'en' ? 'english' : language.code === 'ru' ? 'russian' : 'uzbek'}`)}
                                                </Text>
                                            </View>
                                        </View>
                                        {selectedLanguage === language.code && (
                                            <Ionicons name="checkmark-circle" size={24} color="#3B82F6" />
                                        )}
                                    </TouchableOpacity>
                                ))}
                            </View>
                        </Pressable>
                    </Animated.View>
                </Pressable>
            </Modal>
        </View>
    );
};

const styles = StyleSheet.create({
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    modalContent: {
        width: '100%',
        maxWidth: 400,
    },
});

export default LanguageSelector;
