/**
 * AdminOccasionsTab — List / edit / delete home daily outfit occasions and styles
 */
import React, { useState, useCallback, useEffect } from 'react';
import { ActivityIndicator, Alert, FlatList, Modal, ScrollView, StyleSheet, Switch, TextInput, TouchableOpacity, View,  } from 'react-native'
import { ScaledText } from '../../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../lib/supabase';
import { useTranslation } from 'react-i18next';
import { createLogger } from '../../src/utils/logger';

const logger = createLogger('AdminOccasionsTab');

interface HomeOccasion {
    id: string;
    occasion: string;
    style: string;
    is_active: boolean;
    sort_order: number;
    created_at?: string;
}

export const AdminOccasionsTab = () => {
    const { t } = useTranslation();
    const [occasions, setOccasions] = useState<HomeOccasion[]>([]);
    const [loading, setLoading] = useState(true);
    const [editItem, setEditItem] = useState<HomeOccasion | null>(null);
    const [isAddMode, setIsAddMode] = useState(false);
    const [modalVisible, setModalVisible] = useState(false);

    // List of styles supported by useDailyAIOutfit / edge functions
    const STYLE_OPTIONS = [
        { value: 'business_casual', label: 'Business Casual (Деловой стиль)' },
        { value: 'old_money', label: 'Old Money (Элегантная классика)' },
        { value: 'semi_classic', label: 'Semi Classic (Полуклассика)' },
        { value: 'minimalist', label: 'Minimalist (Минимализм)' },
        { value: 'casual', label: 'Casual (Повседневный)' },
    ];

    const fetchOccasions = useCallback(async () => {
        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('home_occasions')
                .select('*')
                .order('sort_order', { ascending: true });

            if (error) {
                logger.error('Fetch occasions failed', error);
                // If table doesn't exist yet, we degrade gracefully with empty state
                setOccasions([]);
            } else {
                setOccasions((data as HomeOccasion[]) || []);
            }
        } catch (err) {
            logger.error('Fetch occasions error', err);
            setOccasions([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchOccasions();
    }, [fetchOccasions]);

    const handleDelete = (item: HomeOccasion) => {
        Alert.alert(
            t('admin.occasions.deleteTitle', 'Delete Occasion'),
            t('admin.occasions.deleteConfirm', 'Are you sure you want to delete «{{name}}»?', { name: item.occasion }),
            [
                { text: t('common.cancel'), style: 'cancel' },
                {
                    text: t('common.delete'),
                    style: 'destructive',
                    onPress: async () => {
                        const { error } = await supabase
                            .from('home_occasions')
                            .delete()
                            .eq('id', item.id);

                        if (error) {
                            Alert.alert(t('common.error'), error.message);
                        } else {
                            setOccasions((prev) => prev.filter((i) => i.id !== item.id));
                        }
                    },
                },
            ]
        );
    };

    const handleToggleActive = async (item: HomeOccasion) => {
        const newActive = !item.is_active;
        const { error } = await supabase
            .from('home_occasions')
            .update({ is_active: newActive })
            .eq('id', item.id);

        if (error) {
            Alert.alert(t('common.error'), error.message);
        } else {
            setOccasions((prev) =>
                prev.map((i) => (i.id === item.id ? { ...i, is_active: newActive } : i))
            );
        }
    };

    const openEdit = (item: HomeOccasion) => {
        setEditItem({ ...item });
        setIsAddMode(false);
        setModalVisible(true);
    };

    const openAdd = () => {
        setEditItem({
            id: '',
            occasion: '',
            style: 'business_casual',
            is_active: true,
            sort_order: (occasions.length + 1) * 10,
        });
        setIsAddMode(true);
        setModalVisible(true);
    };

    const saveChanges = async () => {
        if (!editItem || !editItem.occasion.trim()) {
            Alert.alert(t('common.error'), t('admin.validation.required', 'Occasion name is required'));
            return;
        }

        if (isAddMode) {
            const { id, ...newObj } = editItem;
            const { data, error } = await supabase
                .from('home_occasions')
                .insert([newObj])
                .select();

            if (error) {
                Alert.alert(t('common.error'), error.message);
            } else {
                if (data && data.length > 0) {
                    setOccasions((prev) => [...prev, data[0] as HomeOccasion].sort((a, b) => a.sort_order - b.sort_order));
                } else {
                    fetchOccasions();
                }
                setModalVisible(false);
                setEditItem(null);
            }
        } else {
            const { id, created_at, ...updates } = editItem;
            const { error } = await supabase
                .from('home_occasions')
                .update(updates)
                .eq('id', id);

            if (error) {
                Alert.alert(t('common.error'), error.message);
            } else {
                setOccasions((prev) =>
                    prev
                        .map((i) => (i.id === editItem.id ? editItem : i))
                        .sort((a, b) => a.sort_order - b.sort_order)
                );
                setModalVisible(false);
                setEditItem(null);
            }
        }
    };

    const renderItem = ({ item }: { item: HomeOccasion }) => (
        <View style={[s.card, !item.is_active && s.cardInactive]}>
            <View style={s.cardInfo}>
                <ScaledText style={s.cardOccasion}>{item.occasion}</ScaledText>
                <ScaledText style={s.cardStyle}>
                    Style Type: <ScaledText style={s.boldText}>{item.style}</ScaledText> · Order: {item.sort_order}
                </ScaledText>
            </View>
            <View style={s.cardActions}>
                <TouchableOpacity onPress={() => handleToggleActive(item)} style={s.iconBtn}>
                    <Ionicons
                        name={item.is_active ? 'eye' : 'eye-off'}
                        size={20}
                        color={item.is_active ? '#34C759' : '#8E8E93'}
                    />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => openEdit(item)} style={s.iconBtn}>
                    <Ionicons name="create" size={20} color="#007AFF" />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => handleDelete(item)} style={s.iconBtn}>
                    <Ionicons name="trash" size={20} color="#FF3B30" />
                </TouchableOpacity>
            </View>
        </View>
    );

    return (
        <View style={s.container}>
            <View style={s.headerRow}>
                <ScaledText style={s.sectionTitle}>Home Suggestions Configuration</ScaledText>
                <TouchableOpacity style={s.addButton} onPress={openAdd}>
                    <Ionicons name="add" size={16} color="#FFF" />
                    <ScaledText style={s.addButtonText}>Add Occasion</ScaledText>
                </TouchableOpacity>
            </View>

            {loading ? (
                <View style={s.center}>
                    <ActivityIndicator size="large" color="#007AFF" />
                </View>
            ) : occasions.length === 0 ? (
                <View style={s.center}>
                    <Ionicons name="sparkles-outline" size={48} color="#8E8E93" />
                    <ScaledText style={s.emptyText}>No custom occasions set in Supabase.</ScaledText>
                    <ScaledText style={s.emptySubText}>The app will fall back to hardcoded default configurations.</ScaledText>
                </View>
            ) : (
                <FlatList
                    data={occasions}
                    keyExtractor={(item) => item.id}
                    renderItem={renderItem}
                    contentContainerStyle={{ paddingBottom: 40 }}
                    refreshing={loading}
                    onRefresh={fetchOccasions}
                />
            )}

            {/* Edit / Add Modal */}
            <Modal visible={modalVisible} animationType="slide" presentationStyle="pageSheet">
                <SafeAreaView style={s.modalContainer}>
                    <View style={s.modalHeader}>
                        <TouchableOpacity onPress={() => setModalVisible(false)}>
                            <ScaledText style={s.modalCancel}>{t('common.cancel')}</ScaledText>
                        </TouchableOpacity>
                        <ScaledText style={s.modalTitle}>
                            {isAddMode ? 'Add Home Occasion' : 'Edit Home Occasion'}
                        </ScaledText>
                        <TouchableOpacity onPress={saveChanges}>
                            <ScaledText style={s.modalSave}>{t('common.save')}</ScaledText>
                        </TouchableOpacity>
                    </View>
                    {editItem && (
                        <ScrollView style={s.modalBody} keyboardShouldPersistTaps="handled">
                            <ScaledText style={s.label}>Occasion Name (English/Key or Custom)</ScaledText>
                            <TextInput
                                style={s.input}
                                value={editItem.occasion}
                                onChangeText={(v) => setEditItem({ ...editItem, occasion: v })}
                                placeholder="e.g. Team Collaboration"
                            />
                            <ScaledText style={s.helperText}>
                                If the value matches keys in i18n locales, it will be automatically translated (e.g. 'Team Collaboration' / 'Night-Time Dinner').
                            </ScaledText>

                            <ScaledText style={s.label}>Style Type (Used by AI Outfit Generator)</ScaledText>
                            <View style={s.chipRow}>
                                {STYLE_OPTIONS.map((styleOpt) => (
                                    <TouchableOpacity
                                        key={styleOpt.value}
                                        style={[s.chip, editItem.style === styleOpt.value && s.chipActive]}
                                        onPress={() => setEditItem({ ...editItem, style: styleOpt.value })}
                                    >
                                        <ScaledText style={[s.chipText, editItem.style === styleOpt.value && s.chipTextActive]}>
                                            {styleOpt.label}
                                        </ScaledText>
                                    </TouchableOpacity>
                                ))}
                            </View>

                            <ScaledText style={s.label}>{t('admin.add.sortOrder')}</ScaledText>
                            <TextInput
                                style={s.input}
                                value={String(editItem.sort_order)}
                                onChangeText={(v) => setEditItem({ ...editItem, sort_order: parseInt(v) || 0 })}
                                keyboardType="number-pad"
                            />

                            <View style={s.switchRow}>
                                <ScaledText style={s.label}>{t('admin.add.isActive')}</ScaledText>
                                <Switch
                                    value={editItem.is_active}
                                    onValueChange={(v) => setEditItem({ ...editItem, is_active: v })}
                                />
                            </View>
                            <View style={{ height: 40 }} />
                        </ScrollView>
                    )}
                </SafeAreaView>
            </Modal>
        </View>
    );
};

const s = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F2F2F7' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
    emptyText: { fontSize: 16, fontWeight: '600', color: '#1C1C1E', marginTop: 12 },
    emptySubText: { fontSize: 14, color: '#8E8E93', marginTop: 4, textAlign: 'center' },
    headerRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
    },
    sectionTitle: { fontSize: 15, fontWeight: '600', color: '#636366' },
    addButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#007AFF',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 8,
    },
    addButtonText: { color: '#FFF', fontSize: 13, fontWeight: '600', marginLeft: 4 },
    card: {
        flexDirection: 'row',
        backgroundColor: '#FFF',
        marginHorizontal: 16,
        marginVertical: 4,
        borderRadius: 14,
        padding: 14,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOpacity: 0.04,
        shadowRadius: 6,
        elevation: 2,
    },
    cardInactive: { opacity: 0.55 },
    cardInfo: { flex: 1 },
    cardOccasion: { fontSize: 16, fontWeight: '600', color: '#1C1C1E' },
    cardStyle: { fontSize: 13, color: '#8E8E93', marginTop: 3 },
    boldText: { fontWeight: '700', color: '#007AFF' },
    cardActions: { flexDirection: 'row', gap: 4 },
    iconBtn: { padding: 8 },
    modalContainer: { flex: 1, backgroundColor: '#F2F2F7' },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: '#E5E5EA',
    },
    modalCancel: { fontSize: 16, color: '#007AFF' },
    modalTitle: { fontSize: 17, fontWeight: '600', color: '#1C1C1E' },
    modalSave: { fontSize: 16, fontWeight: '600', color: '#007AFF' },
    modalBody: { flex: 1, paddingHorizontal: 20 },
    label: { fontSize: 13, fontWeight: '600', color: '#636366', marginTop: 16, marginBottom: 4 },
    helperText: { fontSize: 12, color: '#8E8E93', marginTop: 2, fontStyle: 'italic' },
    input: {
        backgroundColor: '#FFF',
        borderRadius: 10,
        paddingHorizontal: 14,
        paddingVertical: 10,
        fontSize: 15,
        color: '#1C1C1E',
        borderWidth: 1,
        borderColor: '#E5E5EA',
        marginTop: 4,
    },
    chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 6 },
    chip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 20, backgroundColor: '#E5E5EA' },
    chipActive: { backgroundColor: '#007AFF' },
    chipText: { fontSize: 13, fontWeight: '500', color: '#636366' },
    chipTextActive: { color: '#FFF' },
    switchRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 20 },
});
