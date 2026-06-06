/**
 * AdminManageTab — List / edit / delete shop catalog items
 */
import React, { useState, useCallback, useEffect } from 'react';
import { ActivityIndicator, Alert, FlatList, Image, Modal, ScrollView, StyleSheet, Switch, TextInput, TouchableOpacity, View,  } from 'react-native'
import { ScaledText } from '../../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../lib/supabase';
import { useTranslation } from 'react-i18next';
import { createLogger } from '../../src/utils/logger';

const logger = createLogger('AdminManageTab');

interface ShopItem {
    id: string; brand: string; name: string; price: number; currency: string;
    category: string; garment_type: string; description: string; image_url: string;
    is_active: boolean; sort_order: number; source: string;
}

export const AdminManageTab = () => {
    const { t } = useTranslation();
    const [items, setItems] = useState<ShopItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [filterCategory, setFilterCategory] = useState('all');
    const [editItem, setEditItem] = useState<ShopItem | null>(null);
    const [editModalVisible, setEditModalVisible] = useState(false);

    const GARMENT_TYPES = [
        { value: 'upper_body', label: t('admin.garmentTypes.upperBody') },
        { value: 'lower_body', label: t('admin.garmentTypes.lowerBody') },
        { value: 'dresses', label: t('admin.garmentTypes.dresses') },
        { value: 'shoes', label: t('admin.garmentTypes.shoes') },
        { value: 'outfit', label: t('admin.garmentTypes.outfit') },
    ];

    const CATEGORIES = [
        { value: 'tops', label: t('admin.categories.tops') },
        { value: 'bottoms', label: t('admin.categories.bottoms') },
        { value: 'shoes', label: t('admin.categories.shoes') },
        { value: 'dresses', label: t('admin.categories.dresses') },
        { value: 'outerwear', label: t('admin.categories.outerwear') },
    ];

    const fetchItems = useCallback(async () => {
        setLoading(true);
        try {
            let query = supabase.from('shop_catalog').select('id,brand,name,price,currency,category,garment_type,description,image_url,is_active,sort_order,source').order('sort_order', { ascending: true });
            if (filterCategory !== 'all') query = query.eq('garment_type', filterCategory);
            const { data, error } = await query;
            if (error) logger.error('Fetch failed', error);
            else setItems((data as ShopItem[]) || []);
        } catch (err) { logger.error('Fetch error', err); }
        finally { setLoading(false); }
    }, [filterCategory]);

    useEffect(() => { fetchItems(); }, [fetchItems]);

    const handleDelete = (item: ShopItem) => {
        Alert.alert(t('admin.manage.deleteTitle'), t('admin.manage.deleteConfirm', { name: item.name }), [
            { text: t('common.cancel'), style: 'cancel' },
            { text: t('common.delete'), style: 'destructive', onPress: async () => {
                const { error } = await supabase.from('shop_catalog').delete().eq('id', item.id);
                if (error) Alert.alert(t('common.error'), error.message);
                else setItems((prev) => prev.filter((i) => i.id !== item.id));
            }},
        ]);
    };

    const handleToggleActive = async (item: ShopItem) => {
        const newActive = !item.is_active;
        const { error } = await supabase.from('shop_catalog').update({ is_active: newActive }).eq('id', item.id);
        if (error) Alert.alert(t('common.error'), error.message);
        else setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, is_active: newActive } : i)));
    };

    const openEdit = (item: ShopItem) => { setEditItem({ ...item }); setEditModalVisible(true); };

    const saveEdit = async () => {
        if (!editItem) return;
        const { id, ...updates } = editItem;
        const { error } = await supabase.from('shop_catalog').update(updates).eq('id', id);
        if (error) Alert.alert(t('common.error'), error.message);
        else { setItems((prev) => prev.map((i) => (i.id === editItem.id ? editItem : i))); setEditModalVisible(false); setEditItem(null); }
    };

    const renderItem = ({ item }: { item: ShopItem }) => (
        <View style={[s.card, !item.is_active && s.cardInactive]}>
            <Image source={{ uri: item.image_url }} style={s.cardImage} />
            <View style={s.cardInfo}>
                <ScaledText style={s.cardBrand}>{item.brand}</ScaledText>
                <ScaledText style={s.cardName} numberOfLines={1}>{item.name}</ScaledText>
                <ScaledText style={s.cardMeta}>{item.garment_type} · ${item.price} · {item.source}</ScaledText>
            </View>
            <View style={s.cardActions}>
                <TouchableOpacity onPress={() => handleToggleActive(item)} style={s.iconBtn}>
                    <Ionicons name={item.is_active ? 'eye' : 'eye-off'} size={20} color={item.is_active ? '#34C759' : '#8E8E93'} />
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

    const FILTER_OPTIONS = [{ value: 'all', label: t('common.all') }, ...GARMENT_TYPES];

    return (
        <View style={s.container}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.filterRow}>
                {FILTER_OPTIONS.map((opt) => (
                    <TouchableOpacity key={opt.value} style={[s.filterChip, filterCategory === opt.value && s.filterChipActive]} onPress={() => setFilterCategory(opt.value)}>
                        <ScaledText style={[s.filterChipText, filterCategory === opt.value && s.filterChipTextActive]}>{opt.label}</ScaledText>
                    </TouchableOpacity>
                ))}
            </ScrollView>

            {loading ? (
                <View style={s.center}><ActivityIndicator size="large" color="#007AFF" /></View>
            ) : items.length === 0 ? (
                <View style={s.center}>
                    <Ionicons name="shirt-outline" size={48} color="#8E8E93" />
                    <ScaledText style={s.emptyText}>{t('admin.manage.empty')}</ScaledText>
                </View>
            ) : (
                <FlatList data={items} keyExtractor={(item) => item.id} renderItem={renderItem} contentContainerStyle={{ paddingBottom: 40 }} refreshing={loading} onRefresh={fetchItems} />
            )}

            {/* Edit Modal */}
            <Modal visible={editModalVisible} animationType="slide" presentationStyle="pageSheet">
                <SafeAreaView style={s.modalContainer}>
                    <View style={s.modalHeader}>
                        <TouchableOpacity onPress={() => setEditModalVisible(false)}><ScaledText style={s.modalCancel}>{t('common.cancel')}</ScaledText></TouchableOpacity>
                        <ScaledText style={s.modalTitle}>{t('admin.manage.editItem')}</ScaledText>
                        <TouchableOpacity onPress={saveEdit}><ScaledText style={s.modalSave}>{t('common.save')}</ScaledText></TouchableOpacity>
                    </View>
                    {editItem && (
                        <ScrollView style={s.modalBody} keyboardShouldPersistTaps="handled">
                            <ScaledText style={s.label}>{t('admin.add.brand')}</ScaledText>
                            <TextInput style={s.input} value={editItem.brand} onChangeText={(v) => setEditItem({ ...editItem, brand: v })} />
                            <ScaledText style={s.label}>{t('admin.add.name')}</ScaledText>
                            <TextInput style={s.input} value={editItem.name} onChangeText={(v) => setEditItem({ ...editItem, name: v })} />
                            <View style={s.row}>
                                <View style={{ flex: 2 }}>
                                    <ScaledText style={s.label}>{t('admin.add.price')}</ScaledText>
                                    <TextInput style={s.input} value={String(editItem.price)} onChangeText={(v) => setEditItem({ ...editItem, price: parseFloat(v) || 0 })} keyboardType="decimal-pad" />
                                </View>
                                <View style={{ flex: 1, marginLeft: 10 }}>
                                    <ScaledText style={s.label}>{t('admin.add.currency')}</ScaledText>
                                    <TextInput style={s.input} value={editItem.currency} onChangeText={(v) => setEditItem({ ...editItem, currency: v })} />
                                </View>
                            </View>
                            <ScaledText style={s.label}>{t('admin.add.garmentType')}</ScaledText>
                            <View style={s.chipRow}>
                                {GARMENT_TYPES.map((gt) => (
                                    <TouchableOpacity key={gt.value} style={[s.chip, editItem.garment_type === gt.value && s.chipActive]} onPress={() => setEditItem({ ...editItem, garment_type: gt.value })}>
                                        <ScaledText style={[s.chipText, editItem.garment_type === gt.value && s.chipTextActive]}>{gt.label}</ScaledText>
                                    </TouchableOpacity>
                                ))}
                            </View>
                            <ScaledText style={s.label}>{t('admin.add.category')}</ScaledText>
                            <View style={s.chipRow}>
                                {CATEGORIES.map((c) => (
                                    <TouchableOpacity key={c.value} style={[s.chip, editItem.category === c.value && s.chipActive]} onPress={() => setEditItem({ ...editItem, category: c.value })}>
                                        <ScaledText style={[s.chipText, editItem.category === c.value && s.chipTextActive]}>{c.label}</ScaledText>
                                    </TouchableOpacity>
                                ))}
                            </View>
                            <ScaledText style={s.label}>{t('admin.add.description')}</ScaledText>
                            <TextInput style={[s.input, s.textArea]} value={editItem.description} onChangeText={(v) => setEditItem({ ...editItem, description: v })} multiline numberOfLines={3} />
                            <ScaledText style={s.label}>{t('admin.add.orImageUrl')}</ScaledText>
                            <TextInput style={s.input} value={editItem.image_url} onChangeText={(v) => setEditItem({ ...editItem, image_url: v })} autoCapitalize="none" keyboardType="url" />
                            <ScaledText style={s.label}>{t('admin.add.sortOrder')}</ScaledText>
                            <TextInput style={s.input} value={String(editItem.sort_order)} onChangeText={(v) => setEditItem({ ...editItem, sort_order: parseInt(v) || 0 })} keyboardType="number-pad" />
                            <View style={s.switchRow}>
                                <ScaledText style={s.label}>{t('admin.add.isActive')}</ScaledText>
                                <Switch value={editItem.is_active} onValueChange={(v) => setEditItem({ ...editItem, is_active: v })} />
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
    container: { flex: 1 },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    emptyText: { fontSize: 16, color: '#8E8E93', marginTop: 12 },
    filterRow: { paddingHorizontal: 16, paddingVertical: 8, maxHeight: 48 },
    filterChip: { paddingHorizontal: 14, paddingVertical: 7, borderRadius: 20, backgroundColor: '#E5E5EA', marginRight: 8 },
    filterChipActive: { backgroundColor: '#007AFF' },
    filterChipText: { fontSize: 13, fontWeight: '500', color: '#636366' },
    filterChipTextActive: { color: '#FFF' },
    card: { flexDirection: 'row', backgroundColor: '#FFF', marginHorizontal: 16, marginVertical: 4, borderRadius: 14, padding: 10, alignItems: 'center', shadowColor: '#000', shadowOpacity: 0.04, shadowRadius: 6, elevation: 2 },
    cardInactive: { opacity: 0.55 },
    cardImage: { width: 56, height: 72, borderRadius: 8, backgroundColor: '#E5E5EA' },
    cardInfo: { flex: 1, marginLeft: 10 },
    cardBrand: { fontSize: 12, fontWeight: '600', color: '#8E8E93', textTransform: 'uppercase' },
    cardName: { fontSize: 15, fontWeight: '500', color: '#1C1C1E', marginTop: 1 },
    cardMeta: { fontSize: 12, color: '#8E8E93', marginTop: 2 },
    cardActions: { flexDirection: 'row', gap: 4 },
    iconBtn: { padding: 8 },
    modalContainer: { flex: 1, backgroundColor: '#F2F2F7' },
    modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#E5E5EA' },
    modalCancel: { fontSize: 16, color: '#007AFF' },
    modalTitle: { fontSize: 17, fontWeight: '600', color: '#1C1C1E' },
    modalSave: { fontSize: 16, fontWeight: '600', color: '#007AFF' },
    modalBody: { flex: 1, paddingHorizontal: 20 },
    label: { fontSize: 13, fontWeight: '600', color: '#636366', marginTop: 12, marginBottom: 4 },
    input: { backgroundColor: '#FFF', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10, fontSize: 15, color: '#1C1C1E', borderWidth: 1, borderColor: '#E5E5EA' },
    textArea: { minHeight: 70, textAlignVertical: 'top' },
    row: { flexDirection: 'row' },
    chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4 },
    chip: { paddingHorizontal: 14, paddingVertical: 7, borderRadius: 20, backgroundColor: '#E5E5EA' },
    chipActive: { backgroundColor: '#007AFF' },
    chipText: { fontSize: 13, fontWeight: '500', color: '#636366' },
    chipTextActive: { color: '#FFF' },
    switchRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 },
});
