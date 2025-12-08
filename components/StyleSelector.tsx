import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    Modal,
    StyleSheet,
    Animated,
    Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';
import { DESIGNER_STYLES, DesignerStyle } from '../src/styles/designerStyles';
import { colors, shadows, spacing } from '../src/theme';

const STYLE_STORAGE_KEY = '@selected_style';

interface StyleSelectorProps {
    onStyleChange?: () => void;
}

const StyleSelector = ({ onStyleChange }: StyleSelectorProps) => {
    const { t } = useTranslation();
    const [modalVisible, setModalVisible] = useState(false);
    const [fadeAnim] = useState(new Animated.Value(0));
    const [selectedStyle, setSelectedStyle] = useState<string | null>(null);

    // Load saved style on mount
    useEffect(() => {
        loadSavedStyle();
    }, []);

    const loadSavedStyle = async () => {
        try {
            const saved = await AsyncStorage.getItem(STYLE_STORAGE_KEY);
            if (saved) {
                setSelectedStyle(saved);
            }
        } catch (error) {
            console.error('Error loading style:', error);
        }
    };

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
        }).start(() => setModalVisible(false));
    };

    const handleStyleSelect = async (styleId: string) => {
        try {
            await AsyncStorage.setItem(STYLE_STORAGE_KEY, styleId);
            setSelectedStyle(styleId);
            closeModal();
            // Notify parent component of style change
            if (onStyleChange) {
                onStyleChange();
            }
        } catch (error) {
            console.error('Error saving style:', error);
        }
    };

    const getSelectedStyleName = () => {
        if (!selectedStyle) return t('styles.selectStyle');
        const style = DESIGNER_STYLES.find(s => s.id === selectedStyle);
        return style ? t(`${style.translationKey}.name`) : t('styles.selectStyle');
    };

    return (
        <>
            <TouchableOpacity
                style={styles.selectorButton}
                onPress={openModal}
                activeOpacity={0.7}
            >
                <View style={styles.buttonContent}>
                    <Ionicons name="sparkles" size={20} color="#4f46e5" />
                    <View style={styles.textContainer}>
                        <Text style={styles.label}>{t('styles.yourStyle')}</Text>
                        <Text style={styles.selectedValue}>{getSelectedStyleName()}</Text>
                    </View>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#94a3b8" />
            </TouchableOpacity>

            <Modal
                transparent
                visible={modalVisible}
                animationType="none"
                onRequestClose={closeModal}
            >
                <Animated.View style={[styles.modalOverlay, { opacity: fadeAnim }]}>
                    <TouchableOpacity
                        style={StyleSheet.absoluteFill}
                        onPress={closeModal}
                        activeOpacity={1}
                    />
                    <Animated.View
                        style={[
                            styles.modalContent,
                            {
                                transform: [
                                    {
                                        translateY: fadeAnim.interpolate({
                                            inputRange: [0, 1],
                                            outputRange: [300, 0],
                                        }),
                                    },
                                ],
                            },
                        ]}
                    >
                        <View style={styles.modalHeader}>
                            <Text style={styles.modalTitle}>{t('styles.selectStyle')}</Text>
                            <TouchableOpacity onPress={closeModal} style={styles.closeButton}>
                                <Ionicons name="close" size={24} color="#64748b" />
                            </TouchableOpacity>
                        </View>

                        <View style={styles.stylesList}>
                            {DESIGNER_STYLES.map((style) => (
                                <TouchableOpacity
                                    key={style.id}
                                    style={[
                                        styles.styleCard,
                                        selectedStyle === style.id && styles.selectedCard,
                                    ]}
                                    onPress={() => handleStyleSelect(style.id)}
                                    activeOpacity={0.7}
                                >
                                    <LinearGradient
                                        colors={[style.colors[0] + '20', style.colors[1] + '10']}
                                        style={styles.cardGradient}
                                    >
                                        <View style={styles.cardHeader}>
                                            <Text style={styles.styleIcon}>{style.icon}</Text>
                                            <View style={styles.colorsPreview}>
                                                {style.colors.map((color, idx) => (
                                                    <View
                                                        key={idx}
                                                        style={[styles.colorDot, { backgroundColor: color }]}
                                                    />
                                                ))}
                                            </View>
                                        </View>
                                        <Text style={styles.styleName}>
                                            {t(`${style.translationKey}.name`)}
                                        </Text>
                                        <Text style={styles.styleDescription}>
                                            {t(`${style.translationKey}.description`)}
                                        </Text>
                                        {selectedStyle === style.id && (
                                            <View style={styles.checkmark}>
                                                <Ionicons name="checkmark-circle" size={24} color="#4f46e5" />
                                            </View>
                                        )}
                                    </LinearGradient>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </Animated.View>
                </Animated.View>
            </Modal>
        </>
    );
};

const styles = StyleSheet.create({
    selectorButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#fff',
        paddingHorizontal: 16,
        paddingVertical: 14,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: '#e2e8f0',
        ...shadows.soft,
    },
    buttonContent: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    textContainer: {
        gap: 2,
    },
    label: {
        fontSize: 12,
        color: '#94a3b8',
        fontWeight: '500',
    },
    selectedValue: {
        fontSize: 15,
        color: '#1e293b',
        fontWeight: '600',
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'flex-end',
    },
    modalContent: {
        backgroundColor: '#fff',
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        paddingBottom: 40,
        maxHeight: '80%',
    },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingVertical: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#f1f5f9',
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#1e293b',
    },
    closeButton: {
        padding: 4,
    },
    stylesList: {
        padding: 16,
        gap: 12,
    },
    styleCard: {
        borderRadius: 16,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    selectedCard: {
        borderColor: '#4f46e5',
    },
    cardGradient: {
        padding: 16,
        position: 'relative',
    },
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    styleIcon: {
        fontSize: 28,
    },
    colorsPreview: {
        flexDirection: 'row',
        gap: 6,
    },
    colorDot: {
        width: 16,
        height: 16,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.1)',
    },
    styleName: {
        fontSize: 18,
        fontWeight: '700',
        color: '#1e293b',
        marginBottom: 4,
    },
    styleDescription: {
        fontSize: 14,
        color: '#64748b',
        lineHeight: 20,
    },
    checkmark: {
        position: 'absolute',
        top: 16,
        right: 16,
    },
});

export default StyleSelector;
