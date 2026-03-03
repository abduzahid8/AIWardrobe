/**
 * OutfitLogForm — Modal for selecting wardrobe items and logging an outfit.
 */

import React from 'react';
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    Image,
    ScrollView,
    StyleSheet,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { colors, spacing, borderRadius } from '../../../src/theme';
import { OCCASIONS } from '../hooks/useOutfitCalendar';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface OutfitLogFormProps {
    visible: boolean;
    wardrobeItems: any[];
    selectedItems: any[];
    selectedOccasion: string;
    onClose: () => void;
    onToggleItem: (item: any) => void;
    onSelectOccasion: (id: string) => void;
    onSave: () => void;
}

export const OutfitLogForm: React.FC<OutfitLogFormProps> = ({
    visible,
    wardrobeItems,
    selectedItems,
    selectedOccasion,
    onClose,
    onToggleItem,
    onSelectOccasion,
    onSave,
}) => {
    return (
        <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
            <View style={styles.overlay}>
                <View style={styles.modal}>
                    <View style={styles.handle} />

                    <View style={styles.header}>
                        <Text style={styles.title}>Log Outfit</Text>
                        <TouchableOpacity onPress={onClose}>
                            <Ionicons name="close" size={24} color={colors.text.primary} />
                        </TouchableOpacity>
                    </View>

                    {/* Occasion Selection */}
                    <Text style={styles.label}>Occasion</Text>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.occasionScroll}>
                        {OCCASIONS.map(occ => (
                            <TouchableOpacity
                                key={occ.id}
                                style={[
                                    styles.occasionChip,
                                    selectedOccasion === occ.id && { backgroundColor: occ.color },
                                ]}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                    onSelectOccasion(occ.id);
                                }}
                            >
                                <Text style={styles.occasionEmoji}>{occ.icon}</Text>
                                <Text style={[
                                    styles.occasionLabel,
                                    selectedOccasion === occ.id && { color: '#FFF' },
                                ]}>
                                    {occ.label}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>

                    {/* Selected Items Preview */}
                    {selectedItems.length > 0 && (
                        <View style={styles.selectedRow}>
                            {selectedItems.map((item, idx) => (
                                <View key={idx} style={styles.selectedItem}>
                                    <Image source={{ uri: item.image || item.imageUrl }} style={styles.selectedImage} />
                                    <TouchableOpacity style={styles.removeBtn} onPress={() => onToggleItem(item)}>
                                        <Ionicons name="close-circle" size={18} color="#EF4444" />
                                    </TouchableOpacity>
                                </View>
                            ))}
                        </View>
                    )}

                    {/* Wardrobe Grid */}
                    <Text style={styles.label}>Select Items ({selectedItems.length}/6)</Text>
                    <ScrollView style={styles.wardrobeScroll}>
                        <View style={styles.wardrobeGrid}>
                            {wardrobeItems.map((item, idx) => {
                                const isSelected = selectedItems.find(i => i.id === item.id);
                                return (
                                    <TouchableOpacity
                                        key={item.id || idx}
                                        style={[styles.wardrobeItem, isSelected && styles.wardrobeItemSelected]}
                                        onPress={() => onToggleItem(item)}
                                    >
                                        <Image
                                            source={{ uri: item.image || item.imageUrl }}
                                            style={styles.wardrobeImage}
                                        />
                                        {isSelected && (
                                            <View style={styles.check}>
                                                <Ionicons name="checkmark" size={16} color="#FFF" />
                                            </View>
                                        )}
                                    </TouchableOpacity>
                                );
                            })}
                        </View>
                    </ScrollView>

                    {/* Save */}
                    <TouchableOpacity
                        style={[styles.saveBtn, selectedItems.length === 0 && styles.saveBtnDisabled]}
                        onPress={onSave}
                        disabled={selectedItems.length === 0}
                    >
                        <Text style={styles.saveBtnText}>
                            Save Outfit ({selectedItems.length} items)
                        </Text>
                    </TouchableOpacity>
                </View>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
    modal: {
        backgroundColor: colors.background,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        padding: spacing.l,
        maxHeight: '85%',
    },
    handle: {
        width: 40, height: 4,
        backgroundColor: colors.border,
        borderRadius: 2,
        alignSelf: 'center',
        marginBottom: spacing.m,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.m,
    },
    title: { fontSize: 20, fontWeight: '700', color: colors.text.primary },
    label: {
        fontSize: 14, fontWeight: '600',
        color: colors.text.secondary,
        marginBottom: spacing.s,
        marginTop: spacing.m,
    },
    occasionScroll: { marginBottom: spacing.m },
    occasionChip: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.surfaceHighlight,
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.full,
        marginRight: spacing.s,
        gap: 6,
    },
    occasionEmoji: { fontSize: 16 },
    occasionLabel: { fontSize: 14, fontWeight: '600', color: colors.text.primary },
    selectedRow: {
        flexDirection: 'row', flexWrap: 'wrap', gap: spacing.s, marginBottom: spacing.m,
    },
    selectedItem: { position: 'relative' },
    selectedImage: {
        width: 56, height: 56, borderRadius: 12, backgroundColor: colors.surfaceHighlight,
    },
    removeBtn: { position: 'absolute', top: -6, right: -6 },
    wardrobeScroll: { maxHeight: 240 },
    wardrobeGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.s },
    wardrobeItem: {
        width: (SCREEN_WIDTH - spacing.l * 2 - spacing.s * 3) / 4,
        aspectRatio: 1,
        backgroundColor: colors.surfaceHighlight,
        borderRadius: 12,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    wardrobeItemSelected: { borderColor: colors.text.accent },
    wardrobeImage: { width: '100%', height: '100%' },
    check: {
        position: 'absolute', top: 4, right: 4,
        width: 22, height: 22, borderRadius: 11,
        backgroundColor: colors.text.accent,
        justifyContent: 'center', alignItems: 'center',
    },
    saveBtn: {
        backgroundColor: colors.button.primary,
        paddingVertical: spacing.m,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        marginTop: spacing.m,
    },
    saveBtnDisabled: { backgroundColor: colors.border },
    saveBtnText: { color: '#FFF', fontSize: 16, fontWeight: '700' },
});

export default OutfitLogForm;
