/**
 * DayDetailModal — Shows outfit details for a logged day.
 */

import React from 'react';
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    Image,
    StyleSheet,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../../../src/theme';
import { OCCASIONS, type OutfitLog, getOccasionColor } from '../types';

interface DayDetailModalProps {
    visible: boolean;
    selectedDate: string | null;
    outfitLogs: Record<string, OutfitLog>;
    onClose: () => void;
    onDelete: (dateStr: string) => void;
}

export const DayDetailModal: React.FC<DayDetailModalProps> = ({
    visible,
    selectedDate,
    outfitLogs,
    onClose,
    onDelete,
}) => {
    const log = selectedDate ? outfitLogs[selectedDate] : null;
    const occasion = log ? OCCASIONS.find(o => o.id === log.occasion) : null;

    return (
        <Modal visible={visible} animationType="fade" transparent onRequestClose={onClose}>
            <TouchableOpacity style={styles.overlay} activeOpacity={1} onPress={onClose}>
                <View style={styles.modal}>
                    {selectedDate && log && (
                        <>
                            <View style={styles.header}>
                                <Text style={styles.title}>
                                    {new Date(selectedDate).toLocaleDateString('en-US', {
                                        weekday: 'long',
                                        month: 'long',
                                        day: 'numeric',
                                    })}
                                </Text>
                                <View style={[styles.tag, { backgroundColor: getOccasionColor(log.occasion) }]}>
                                    <Text style={styles.tagText}>
                                        {occasion?.icon} {occasion?.label}
                                    </Text>
                                </View>
                            </View>

                            <View style={styles.items}>
                                {log.items.map((item, idx) => (
                                    <View key={idx} style={styles.item}>
                                        <Image source={{ uri: item.image }} style={styles.itemImage} />
                                        <Text style={styles.itemType}>{item.type}</Text>
                                    </View>
                                ))}
                            </View>

                            <TouchableOpacity style={styles.deleteBtn} onPress={() => onDelete(selectedDate)}>
                                <Ionicons name="trash-outline" size={18} color="#EF4444" />
                                <Text style={styles.deleteText}>Delete</Text>
                            </TouchableOpacity>
                        </>
                    )}
                </View>
            </TouchableOpacity>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
    },
    modal: {
        backgroundColor: colors.surface,
        marginHorizontal: spacing.l,
        borderRadius: borderRadius.xl,
        padding: spacing.l,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.m,
    },
    title: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
    tag: { paddingHorizontal: spacing.s, paddingVertical: 4, borderRadius: 12 },
    tagText: { fontSize: 12, fontWeight: '600', color: '#FFF' },
    items: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.s,
        marginBottom: spacing.m,
    },
    item: { alignItems: 'center' },
    itemImage: {
        width: 64,
        height: 64,
        borderRadius: 12,
        backgroundColor: colors.surfaceHighlight,
        marginBottom: 4,
    },
    itemType: { fontSize: 11, color: colors.text.secondary },
    deleteBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
        paddingVertical: spacing.s,
    },
    deleteText: { fontSize: 14, color: '#EF4444', fontWeight: '600' },
});

export default DayDetailModal;
