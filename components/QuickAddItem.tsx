/**
 * QuickAddItem — Manual item entry component
 *
 * Bypasses the full camera/video scan flow.
 * Photo picker + category selector + color picker.
 * Adds items directly to the wardrobe store.
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Image,
    ScrollView,
    TextInput,
    Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import type { ClothingCategory, Season, Occasion } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// ============================================
// COMPONENT
// ============================================

interface QuickAddItemProps {
    onItemAdded?: () => void;
    onCancel?: () => void;
}

const QuickAddItem: React.FC<QuickAddItemProps> = ({ onItemAdded, onCancel }) => {
    const { t } = useTranslation();

    const CATEGORIES: { value: ClothingCategory; label: string; icon: string }[] = [
        { value: 'top', label: t('quickAdd.categories.top'), icon: 'shirt-outline' },
        { value: 'bottom', label: t('quickAdd.categories.bottom'), icon: 'resize-outline' },
        { value: 'dress', label: t('quickAdd.categories.dress'), icon: 'body-outline' },
        { value: 'shoes', label: t('quickAdd.categories.shoes'), icon: 'footsteps-outline' },
        { value: 'outerwear', label: t('quickAdd.categories.outerwear'), icon: 'cloudy-outline' },
        { value: 'accessory', label: t('quickAdd.categories.accessory'), icon: 'watch-outline' },
        { value: 'other', label: t('quickAdd.categories.other'), icon: 'ellipsis-horizontal-outline' },
    ];

    const BASIC_COLORS: { name: string; hex: string }[] = [
        { name: t('quickAdd.colors.black'), hex: '#000000' },
        { name: t('quickAdd.colors.white'), hex: '#FFFFFF' },
        { name: t('quickAdd.colors.navy'), hex: '#1B2A4A' },
        { name: t('quickAdd.colors.gray'), hex: '#808080' },
        { name: t('quickAdd.colors.beige'), hex: '#D4C5A9' },
        { name: t('quickAdd.colors.brown'), hex: '#8B4513' },
        { name: t('quickAdd.colors.red'), hex: '#DC2626' },
        { name: t('quickAdd.colors.blue'), hex: '#2563EB' },
        { name: t('quickAdd.colors.green'), hex: '#16A34A' },
        { name: t('quickAdd.colors.pink'), hex: '#EC4899' },
        { name: t('quickAdd.colors.yellow'), hex: '#EAB308' },
        { name: t('quickAdd.colors.purple'), hex: '#7C3AED' },
    ];

    const [imageUri, setImageUri] = useState<string | null>(null);
    const [category, setCategory] = useState<ClothingCategory>('top');
    const [selectedColor, setSelectedColor] = useState(BASIC_COLORS[0]);
    const [itemName, setItemName] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const addItem = useWardrobeStore((state) => state.addItem);

    const pickImage = useCallback(async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ['images'],
                allowsEditing: true,
                aspect: [1, 1],
                quality: 0.8,
            });

            if (!result.canceled && result.assets[0]) {
                setImageUri(result.assets[0].uri);
            }
        } catch (error) {
            console.error('Image picker error:', error);
        }
    }, []);

    const takePhoto = useCallback(async () => {
        try {
            const { status } = await ImagePicker.requestCameraPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert(t('quickAdd.permissionNeeded'), t('quickAdd.cameraPermissionRequired'));
                return;
            }

            const result = await ImagePicker.launchCameraAsync({
                allowsEditing: true,
                aspect: [1, 1],
                quality: 0.8,
            });

            if (!result.canceled && result.assets[0]) {
                setImageUri(result.assets[0].uri);
            }
        } catch (error) {
            console.error('Camera error:', error);
        }
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!imageUri) {
            Alert.alert(t('quickAdd.photoRequired'), t('quickAdd.photoRequiredMessage'));
            return;
        }

        setIsSubmitting(true);
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        try {
            await addItem({
                userId: '', // Will be set by store/auth
                imageUrl: imageUri,
                category,
                subCategory: itemName || category,
                primaryColor: selectedColor.name,
                colorHex: selectedColor.hex,
                pattern: 'solid',
                material: '',
                seasons: ['spring', 'summer', 'fall', 'winter'] as Season[],
                occasions: ['casual'] as Occasion[],
            });

            await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            onItemAdded?.();
        } catch (error) {
            console.error('Failed to add item:', error);
            Alert.alert(t('quickAdd.error'), t('quickAdd.addItemFailed'));
        } finally {
            setIsSubmitting(false);
        }
    }, [imageUri, category, itemName, selectedColor, addItem, onItemAdded, t]);

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <Text style={styles.title}>{t('quickAdd.title')}</Text>
            <Text style={styles.subtitle}>{t('quickAdd.subtitle')}</Text>

            {/* Photo Section */}
            <View style={styles.photoSection}>
                {imageUri ? (
                    <TouchableOpacity onPress={pickImage} activeOpacity={0.8}>
                        <Image source={{ uri: imageUri }} style={styles.photoPreview} />
                        <View style={styles.changePhotoOverlay}>
                            <Ionicons name="camera-outline" size={20} color="#FFF" />
                            <Text style={styles.changePhotoText}>{t('quickAdd.change')}</Text>
                        </View>
                    </TouchableOpacity>
                ) : (
                    <View style={styles.photoButtons}>
                        <TouchableOpacity style={styles.photoButton} onPress={takePhoto}>
                            <Ionicons name="camera-outline" size={28} color={colors.text.primary} />
                            <Text style={styles.photoButtonText}>{t('quickAdd.camera')}</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.photoButton} onPress={pickImage}>
                            <Ionicons name="images-outline" size={28} color={colors.text.primary} />
                            <Text style={styles.photoButtonText}>{t('quickAdd.gallery')}</Text>
                        </TouchableOpacity>
                    </View>
                )}
            </View>

            {/* Name (optional) */}
            <View style={styles.section}>
                <Text style={styles.sectionLabel}>{t('quickAdd.nameOptional')}</Text>
                <TextInput
                    style={styles.textInput}
                    value={itemName}
                    onChangeText={setItemName}
                    placeholder={t('quickAdd.namePlaceholder')}
                    placeholderTextColor={colors.text.tertiary}
                />
            </View>

            {/* Category */}
            <View style={styles.section}>
                <Text style={styles.sectionLabel}>{t('quickAdd.category')}</Text>
                <View style={styles.categoryRow}>
                    {CATEGORIES.map((cat) => (
                        <TouchableOpacity
                            key={cat.value}
                            style={[
                                styles.categoryChip,
                                category === cat.value && styles.categoryChipActive,
                            ]}
                            onPress={() => {
                                setCategory(cat.value);
                                Haptics.selectionAsync();
                            }}
                        >
                            <Ionicons
                                name={cat.icon as any}
                                size={18}
                                color={category === cat.value ? '#FFF' : colors.text.secondary}
                            />
                            <Text
                                style={[
                                    styles.categoryLabel,
                                    category === cat.value && styles.categoryLabelActive,
                                ]}
                            >
                                {cat.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </View>

            {/* Color */}
            <View style={styles.section}>
                <Text style={styles.sectionLabel}>{t('quickAdd.primaryColor')}</Text>
                <View style={styles.colorRow}>
                    {BASIC_COLORS.map((c) => (
                        <TouchableOpacity
                            key={c.hex}
                            style={[
                                styles.colorDot,
                                { backgroundColor: c.hex },
                                c.hex === '#FFFFFF' && styles.whiteDot,
                                selectedColor.hex === c.hex && styles.colorDotActive,
                            ]}
                            onPress={() => {
                                setSelectedColor(c);
                                Haptics.selectionAsync();
                            }}
                        >
                            {selectedColor.hex === c.hex && (
                                <Ionicons
                                    name="checkmark"
                                    size={14}
                                    color={c.hex === '#FFFFFF' || c.hex === '#EAB308' ? '#000' : '#FFF'}
                                />
                            )}
                        </TouchableOpacity>
                    ))}
                </View>
            </View>

            {/* Submit */}
            <TouchableOpacity
                style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]}
                onPress={handleSubmit}
                disabled={isSubmitting}
                activeOpacity={0.8}
            >
                <Ionicons name="add-circle-outline" size={20} color="#FFF" />
                <Text style={styles.submitText}>
                    {isSubmitting ? t('quickAdd.adding') : t('quickAdd.addToCloset')}
                </Text>
            </TouchableOpacity>

            {onCancel && (
                <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
                    <Text style={styles.cancelText}>{t('quickAdd.cancel')}</Text>
                </TouchableOpacity>
            )}
        </ScrollView>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    content: {
        padding: spacing.lg,
        gap: spacing.lg,
    },
    title: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    subtitle: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        marginTop: -spacing.sm,
    },

    // Photo
    photoSection: {
        alignItems: 'center',
    },
    photoPreview: {
        width: 200,
        height: 200,
        borderRadius: radius.lg,
    },
    changePhotoOverlay: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: spacing.sm,
        borderBottomLeftRadius: radius.lg,
        borderBottomRightRadius: radius.lg,
        gap: spacing.xs,
    },
    changePhotoText: {
        ...typography.scale.labelSmall,
        color: '#FFF',
        fontWeight: '600',
    },
    photoButtons: {
        flexDirection: 'row',
        gap: spacing.lg,
    },
    photoButton: {
        width: 100,
        height: 100,
        borderRadius: radius.lg,
        backgroundColor: colors.glass.frosted,
        borderWidth: 1,
        borderColor: colors.border.glass,
        alignItems: 'center',
        justifyContent: 'center',
        gap: spacing.xs,
    },
    photoButtonText: {
        ...typography.scale.labelSmall,
        color: colors.text.secondary,
    },

    // Sections
    section: {
        gap: spacing.sm,
    },
    sectionLabel: {
        ...typography.scale.labelLarge,
        color: colors.text.primary,
        fontWeight: '600',
    },

    // Text input
    textInput: {
        ...typography.scale.bodyMedium,
        color: colors.text.primary,
        backgroundColor: colors.glass.frosted,
        borderWidth: 1,
        borderColor: colors.border.glass,
        borderRadius: radius.md,
        padding: spacing.md,
    },

    // Category
    categoryRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.sm,
    },
    categoryChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: spacing.sm,
        paddingHorizontal: spacing.md,
        borderRadius: radius.pill,
        backgroundColor: colors.glass.frosted,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.xs,
    },
    categoryChipActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    categoryLabel: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
    },
    categoryLabelActive: {
        color: '#FFF',
    },

    // Color
    colorRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.sm,
    },
    colorDot: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    whiteDot: {
        borderWidth: 1,
        borderColor: colors.border.subtle,
    },
    colorDotActive: {
        borderWidth: 2.5,
        borderColor: colors.text.primary,
    },

    // Submit
    submitButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.text.primary,
        paddingVertical: spacing.md,
        borderRadius: radius.pill,
        gap: spacing.sm,
    },
    submitButtonDisabled: {
        opacity: 0.6,
    },
    submitText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },
    cancelButton: {
        alignItems: 'center',
        paddingVertical: spacing.md,
    },
    cancelText: {
        ...typography.scale.labelLarge,
        color: colors.text.secondary,
    },
});

export default QuickAddItem;
