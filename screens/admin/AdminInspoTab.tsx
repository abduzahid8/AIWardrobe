/**
 * AdminInspoTab — Full edit access for the Inspiration page
 *  - Featured Capsules: full CRUD (title, subtitle, link_url, image, sort, active)
 *  - Shop / Inspo items: edit text (brand, name, price, description), change image, promote, delete
 */
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, FlatList, Image, KeyboardAvoidingView, Modal, Platform, ScrollView, StyleSheet, Switch, TextInput, TouchableOpacity, View,  } from 'react-native'
import { ScaledText } from '../../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { supabase } from '../../lib/supabase';
import { useTranslation } from 'react-i18next';
import { INSPO_MENS_SHOP_ITEMS } from '../../data/inspoMensShopItems';
import type { ShopCatalogItem } from '../../features/try-on/types';
import { createLogger } from '../../src/utils/logger';

const logger = createLogger('AdminInspoTab');
const SUPPORTED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];

const addCacheBust = (url?: string | null) => {
    if (!url) return url;
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}v=${Date.now()}`;
};

interface CapsuleExtras { id: string; title: string; subtitle: string; link_url: string; sort_order: number; is_active: boolean; image_url: string }
interface ShopExtras { id: string; brand: string; name: string; price: number; description: string; image_url: string; sort_order: number; is_active: boolean; category: string; garment_type: string }
type EditTarget =
    | { kind: 'capsule'; row: CapsuleExtras }
    | { kind: 'shop'; row: ShopExtras }
    | null;

export const AdminInspoTab = () => {
    const { t } = useTranslation();
    const [inspoItems, setInspoItems] = useState<ShopCatalogItem[]>(INSPO_MENS_SHOP_ITEMS);
    const [loading, setLoading] = useState(false);
    const [filterType, setFilterType] = useState('all');
    const [promoting, setPromoting] = useState<string | null>(null);
    const [capsuleMap, setCapsuleMap] = useState<Record<string, CapsuleExtras>>({});
    const [shopMap, setShopMap] = useState<Record<string, ShopExtras>>({});
    const [editTarget, setEditTarget] = useState<EditTarget>(null);
    const [editLocalImage, setEditLocalImage] = useState<string | null>(null);
    const [savingEdit, setSavingEdit] = useState(false);

    // Fetch items from shop_catalog (includes capsule items from DB)
    useEffect(() => {
        loadDbItems();
    }, []);

    const loadDbItems = async () => {
        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('shop_catalog')
                .select('id, brand, name, price, currency, category, garment_type, description, image_url, sort_order, is_active')
                .order('sort_order', { ascending: true });

            if (error) throw error;

            const dbItems: ShopCatalogItem[] = (data || []).filter((i: any) => i.is_active).map((item: any) => ({
                id: item.id,
                brand: item.brand,
                name: item.name,
                price: item.price,
                currency: item.currency,
                garmentType: item.garment_type as ShopCatalogItem['garmentType'],
                description: item.description,
                imageUrl: addCacheBust(item.image_url),
            }));
            const nextShopMap: Record<string, ShopExtras> = {};
            (data || []).forEach((row: any) => {
                nextShopMap[row.id] = {
                    id: row.id,
                    brand: row.brand || '',
                    name: row.name || '',
                    price: row.price || 0,
                    description: row.description || '',
                    image_url: row.image_url || '',
                    sort_order: row.sort_order || 0,
                    is_active: !!row.is_active,
                    category: row.category || '',
                    garment_type: row.garment_type || '',
                };
            });
            setShopMap(nextShopMap);

            // Merge: DB items take precedence, add hardcoded items not in DB
            const dbIds = new Set(dbItems.map(i => i.id));
            const localItems = INSPO_MENS_SHOP_ITEMS.filter(i => !dbIds.has(i.id));

            const { data: capsulesData } = await supabase
                .from('featured_capsules')
                .select('*')
                .order('sort_order', { ascending: true });

            const nextCapsuleMap: Record<string, CapsuleExtras> = {};
            (capsulesData || []).forEach((c: any) => {
                nextCapsuleMap[`capsule_real_${c.id}`] = {
                    id: c.id,
                    title: c.title || '',
                    subtitle: c.subtitle || '',
                    link_url: c.link_url || '',
                    sort_order: c.sort_order || 0,
                    is_active: !!c.is_active,
                    image_url: c.image_url || '',
                };
            });
            setCapsuleMap(nextCapsuleMap);

            const mappedCapsules: ShopCatalogItem[] = (capsulesData || [])
                .filter((c: any) => c.is_active)
                .map((c: any) => ({
                    id: `capsule_real_${c.id}`,
                    brand: c.subtitle || 'Capsule',
                    name: c.title,
                    price: 0,
                    currency: 'USD',
                    garmentType: 'capsule' as ShopCatalogItem['garmentType'],
                    description: '',
                    imageUrl: addCacheBust(c.image_url),
                }));

            setInspoItems([...dbItems, ...localItems, ...mappedCapsules]);
        } catch (err) {
            logger.error('Failed to load DB items', err);
        } finally {
            setLoading(false);
        }
    };

    const openEdit = (item: ShopCatalogItem) => {
        setEditLocalImage(null);
        if (item.id.startsWith('capsule_real_')) {
            const row = capsuleMap[item.id];
            if (!row) { Alert.alert(t('common.error'), 'Capsule not found'); return; }
            setEditTarget({ kind: 'capsule', row: { ...row } });
            return;
        }
        const dbRow = shopMap[item.id];
        if (dbRow) {
            setEditTarget({ kind: 'shop', row: { ...dbRow } });
            return;
        }
        // Hardcoded inspo item — seed a new shop_catalog row using the inspo- prefix
        setEditTarget({
            kind: 'shop',
            row: {
                id: `inspo-${item.id}`,
                brand: item.brand,
                name: item.name,
                price: item.price,
                description: item.description || '',
                image_url: typeof item.imageUrl === 'string' ? item.imageUrl : '',
                sort_order: 0,
                is_active: true,
                category: item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : item.garmentType,
                garment_type: item.garmentType,
            },
        });
    };

    const addNewCapsule = () => {
        setEditLocalImage(null);
        setEditTarget({
            kind: 'capsule',
            row: { id: '', title: '', subtitle: '', link_url: '', sort_order: 0, is_active: true, image_url: '' },
        });
    };

    const uploadEditImage = async (uri: string, prefix: string): Promise<string | null> => {
        const fileExt = uri.split('.').pop()?.toLowerCase() || 'jpg';
        const safeExt = SUPPORTED_IMAGE_EXTENSIONS.includes(fileExt) ? fileExt : 'jpg';
        const fileName = `${prefix}-${Date.now()}.${safeExt}`;
        const filePath = `admin/${fileName}`;
        const contentType = `image/${safeExt === 'jpg' ? 'jpeg' : safeExt}`;
        const { data: { session } } = await supabase.auth.getSession();
        const token = session?.access_token || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
        const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
        const uploadResult = await FileSystem.uploadAsync(
            `${supabaseUrl}/storage/v1/object/shop-catalog/${filePath}`,
            uri,
            { httpMethod: 'POST', headers: { Authorization: `Bearer ${token}`, apikey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!, 'Content-Type': contentType } }
        );
        if (uploadResult.status !== 200) {
            try { const p = JSON.parse(uploadResult.body); throw new Error(p.message || p.error || t('admin.inspo.uploadFailed')); }
            catch (e: any) { throw new Error(e.message || t('admin.inspo.uploadFailed')); }
        }
        return supabase.storage.from('shop-catalog').getPublicUrl(filePath).data.publicUrl;
    };

    const pickEditImage = async () => {
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: 'images' as ImagePicker.MediaType, quality: 0.85, allowsEditing: true,
        });
        if (!result.canceled && result.assets[0]?.uri) setEditLocalImage(result.assets[0].uri);
    };

    const saveEdit = async () => {
        if (!editTarget) return;
        setSavingEdit(true);
        try {
            let nextImageUrl: string | null = null;
            if (editLocalImage) {
                try { nextImageUrl = await uploadEditImage(editLocalImage, editTarget.kind); }
                catch (err: any) { Alert.alert(t('common.error'), err.message || 'Upload failed'); setSavingEdit(false); return; }
            }

            if (editTarget.kind === 'capsule') {
                const r = editTarget.row;
                if (!r.title.trim()) { Alert.alert(t('common.error'), t('admin.add.name') + ' *'); setSavingEdit(false); return; }
                const finalImage = nextImageUrl || r.image_url;
                if (!finalImage) { Alert.alert(t('common.error'), t('admin.validation.imageRequired', 'Image required')); setSavingEdit(false); return; }
                const payload = {
                    title: r.title.trim(),
                    subtitle: r.subtitle.trim() || null,
                    link_url: r.link_url.trim() || null,
                    sort_order: r.sort_order,
                    is_active: r.is_active,
                    image_url: finalImage,
                };
                if (r.id) {
                    const { error } = await supabase.from('featured_capsules').update(payload).eq('id', r.id);
                    if (error) throw error;
                } else {
                    const { error } = await supabase.from('featured_capsules').insert(payload);
                    if (error) throw error;
                }
            } else {
                const r = editTarget.row;
                if (!r.name.trim() || !r.brand.trim()) { Alert.alert(t('common.error'), t('admin.validation.nameBrand')); setSavingEdit(false); return; }
                const finalImage = nextImageUrl || r.image_url;
                if (!finalImage) { Alert.alert(t('common.error'), t('admin.validation.imageRequired', 'Image required')); setSavingEdit(false); return; }
                const existing = shopMap[r.id];
                const { error } = await supabase.from('shop_catalog').upsert({
                    id: r.id,
                    brand: r.brand.trim(),
                    name: r.name.trim(),
                    price: r.price || 0,
                    currency: 'USD',
                    category: r.category || 'tops',
                    garment_type: r.garment_type || 'upper_body',
                    description: r.description.trim(),
                    image_url: finalImage,
                    is_active: r.is_active,
                    sort_order: r.sort_order,
                    source: existing ? 'admin' : 'inspo',
                }, { onConflict: 'id' });
                if (error) throw error;
            }

            setEditTarget(null);
            setEditLocalImage(null);
            await loadDbItems();
            Alert.alert(t('common.success'), t('admin.guide.saved', 'Saved'));
        } catch (err: any) {
            logger.error('saveEdit failed', err);
            Alert.alert(t('common.error'), err.message || 'Save failed');
        } finally {
            setSavingEdit(false);
        }
    };

    const deleteItem = (item: ShopCatalogItem) => {
        const isCapsule = item.id.startsWith('capsule_real_');
        const inDb = isCapsule || !!shopMap[item.id];
        if (!inDb) { Alert.alert(t('common.info'), 'Hardcoded item — promote first to delete'); return; }
        Alert.alert(t('admin.manage.deleteTitle', 'Delete'), t('admin.manage.deleteConfirm', { name: item.name }), [
            { text: t('common.cancel'), style: 'cancel' },
            {
                text: t('common.delete'), style: 'destructive', onPress: async () => {
                    if (isCapsule) {
                        const realId = item.id.replace('capsule_real_', '');
                        const { error } = await supabase.from('featured_capsules').delete().eq('id', realId);
                        if (error) { Alert.alert(t('common.error'), error.message); return; }
                    } else {
                        const { error } = await supabase.from('shop_catalog').delete().eq('id', item.id);
                        if (error) { Alert.alert(t('common.error'), error.message); return; }
                    }
                    await loadDbItems();
                },
            },
        ]);
    };

    const GARMENT_TYPES = [
        { value: 'upper_body', label: t('admin.garmentTypes.upperBody', 'Tops') },
        { value: 'lower_body', label: t('admin.garmentTypes.lowerBody', 'Bottoms') },
        { value: 'dresses', label: t('admin.garmentTypes.dresses', 'Dresses') },
        { value: 'shoes', label: t('admin.garmentTypes.shoes', 'Shoes') },
        { value: 'outfit', label: t('admin.garmentTypes.outfit', 'Outfit') },
        { value: 'capsule', label: t('admin.garmentTypes.capsule', 'Capsules') },
    ];

    const filtered = filterType === 'all'
        ? inspoItems
        : inspoItems.filter((i) => i.garmentType === filterType);

    const promoteToShop = async (item: ShopCatalogItem) => {
        setPromoting(item.id);
        try {
            // Check if item already exists in DB (starts with inspo- or is capsule)
            const isAlreadyInDb = item.id.startsWith('inspo-') || item.id.startsWith('inspo-capsule-');
            
            if (isAlreadyInDb) {
                Alert.alert(t('common.info'), t('admin.inspo.alreadyInShop', 'Item already in shop catalog'));
                return;
            }

            const id = `inspo-${item.id}`;
            const category = item.garmentType === 'upper_body' ? 'tops'
                : item.garmentType === 'lower_body' ? 'bottoms'
                : item.garmentType;
            const { error } = await supabase.from('shop_catalog').upsert({
                id, brand: item.brand, name: item.name, price: item.price,
                currency: item.currency || 'USD', category, garment_type: item.garmentType,
                description: item.description || '', image_url: item.imageUrl,
                is_active: true, sort_order: 0, source: 'inspo',
            }, { onConflict: 'id' });
            if (error) Alert.alert(t('common.error'), error.message);
            else {
                Alert.alert(t('common.success'), t('admin.inspo.promoted'));
                loadDbItems(); // Refresh to show the promoted item
            }
        } catch (err) { logger.error('Promote failed', err); Alert.alert(t('common.error'), t('admin.inspo.promoteFailed'));
        } finally { setPromoting(null); }
    };

    const changePhoto = async (item: ShopCatalogItem) => {
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: 'images' as ImagePicker.MediaType, quality: 0.85, allowsEditing: true,
        });
        if (result.canceled || !result.assets[0] || !result.assets[0].uri) return;

        const asset = result.assets[0];
        const uri = asset.uri;
        const mimeFromAsset = asset.mimeType?.toLowerCase() || '';
        const extFromMime = mimeFromAsset.startsWith('image/') ? mimeFromAsset.replace('image/', '') : '';
        const fileExt = uri.split('.').pop()?.toLowerCase() || '';
        const safeExt = SUPPORTED_IMAGE_EXTENSIONS.includes(fileExt)
            ? fileExt
            : (SUPPORTED_IMAGE_EXTENSIONS.includes(extFromMime) ? extFromMime : 'jpg');
        const fileName = `inspo-${item.id.replace('capsule_real_', '')}-${Date.now()}.${safeExt}`;
        const filePath = `admin/${fileName}`;
        const contentType = `image/${safeExt === 'jpg' ? 'jpeg' : safeExt}`;
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
            const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;

            const uploadResult = await FileSystem.uploadAsync(
                `${supabaseUrl}/storage/v1/object/shop-catalog/${filePath}`,
                uri,
                {
                    httpMethod: 'POST',
                    headers: {
                        Authorization: `Bearer ${token}`,
                        apikey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!,
                        'Content-Type': contentType,
                    },
                }
            );

            if (uploadResult.status !== 200) {
                let errorMsg = t('admin.inspo.uploadFailed');
                try {
                    const parsed = JSON.parse(uploadResult.body);
                    errorMsg = parsed.message || parsed.error || errorMsg;
                } catch (e) {}
                throw new Error(errorMsg);
            }
        } catch (err: any) {
            logger.error('Upload exception', err);
            Alert.alert(t('common.error'), err.message || t('admin.inspo.uploadError'));
            return;
        }

        const { data: urlData } = supabase.storage.from('shop-catalog').getPublicUrl(filePath);
        const nextPublicUrl = urlData.publicUrl;
        const nextDisplayUrl = addCacheBust(nextPublicUrl);
        
        if (item.id.startsWith('capsule_real_')) {
            const capsuleId = item.id.replace('capsule_real_', '');
            const { data: updatedCapsule, error: updateError } = await supabase
                .from('featured_capsules')
                .update({ image_url: nextPublicUrl })
                .eq('id', capsuleId)
                .select('id')
                .maybeSingle();
            
            if (updateError) {
                Alert.alert(t('common.error'), updateError.message);
            } else if (!updatedCapsule) {
                Alert.alert(t('common.error'), t('admin.inspo.photoUpdateNoAccess', 'Update blocked: no permission to modify this capsule'));
            }
            else {
                setInspoItems(prev => prev.map(i => i.id === item.id ? { ...i, imageUrl: nextDisplayUrl } : i));
                await loadDbItems();
                Alert.alert(t('common.success'), t('admin.inspo.photoChanged'));
            }
            return;
        }

        const isAlreadyInDb = item.id.startsWith('inspo-') || item.id.startsWith('inspo-capsule-');
        const id = isAlreadyInDb ? item.id : `inspo-${item.id}`;

        const category = item.garmentType === 'upper_body' ? 'tops'
            : item.garmentType === 'lower_body' ? 'bottoms'
            : item.garmentType;
        const existingShop = shopMap[id];
        const { error: upsertError } = await supabase.from('shop_catalog').upsert({
            id, brand: item.brand, name: item.name, price: item.price,
            currency: item.currency || 'USD', category, garment_type: item.garmentType,
            description: item.description || '', image_url: nextPublicUrl,
            is_active: existingShop?.is_active ?? true,
            sort_order: existingShop?.sort_order ?? 0,
            source: 'inspo',
        }, { onConflict: 'id' });

        if (upsertError) Alert.alert(t('common.error'), upsertError.message);
        else {
            setInspoItems(prev => prev.map(i => i.id === item.id ? { ...i, id, imageUrl: nextDisplayUrl } : i));
            await loadDbItems();
            Alert.alert(t('common.success'), t('admin.inspo.photoChanged'));
        }
    };

    const renderInspoItem = ({ item }: { item: ShopCatalogItem }) => (
        <View style={s.card}>
            <Image
                key={typeof item.imageUrl === 'string' ? item.imageUrl : 'default'}
                source={{ uri: typeof item.imageUrl === 'string' ? item.imageUrl : undefined }}
                style={s.cardImage}
                defaultSource={require('../../assets/images/basic_cardigan.png')}
            />
            <View style={s.cardInfo}>
                <ScaledText style={s.cardBrand}>{item.brand}</ScaledText>
                <ScaledText style={s.cardName} numberOfLines={1}>{item.name}</ScaledText>
                <ScaledText style={s.cardMeta}>{item.garmentType} · ${item.price}</ScaledText>
            </View>
            <View style={s.cardActions}>
                <TouchableOpacity style={s.actionBtn} onPress={() => openEdit(item)}>
                    <Ionicons name="create-outline" size={18} color="#007AFF" />
                    <ScaledText style={s.actionLabel}>{t('common.edit', 'Edit')}</ScaledText>
                </TouchableOpacity>
                <TouchableOpacity style={s.actionBtn} onPress={() => changePhoto(item)}>
                    <Ionicons name="image" size={18} color="#007AFF" />
                    <ScaledText style={s.actionLabel}>{t('admin.inspo.changePhoto', 'Photo')}</ScaledText>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[s.actionBtn, (promoting === item.id || item.id.startsWith('capsule_real_')) && s.actionBtnDisabled]}
                    onPress={() => promoteToShop(item)}
                    disabled={promoting === item.id || item.id.startsWith('capsule_real_')}
                >
                    {promoting === item.id ? (
                        <ActivityIndicator size="small" color="#34C759" />
                    ) : (
                        <Ionicons name="arrow-up-circle" size={18} color="#34C759" />
                    )}
                    <ScaledText style={[s.actionLabel, { color: '#34C759' }]}>{t('admin.inspo.promote')}</ScaledText>
                </TouchableOpacity>
                <TouchableOpacity style={s.actionBtn} onPress={() => deleteItem(item)}>
                    <Ionicons name="trash-outline" size={18} color="#FF3B30" />
                    <ScaledText style={[s.actionLabel, { color: '#FF3B30' }]}>{t('common.delete', 'Delete')}</ScaledText>
                </TouchableOpacity>
            </View>
        </View>
    );

    const FILTER_OPTIONS = [{ value: 'all', label: t('common.all') }, ...GARMENT_TYPES];

    const isCapsuleEdit = editTarget?.kind === 'capsule';
    const editPreviewUri = editLocalImage
        || (editTarget?.kind === 'capsule' ? editTarget.row.image_url : editTarget?.kind === 'shop' ? editTarget.row.image_url : '');

    return (
        <View style={s.container}>
            <View style={s.toolbar}>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.filterRow}>
                    {FILTER_OPTIONS.map((opt) => (
                        <TouchableOpacity key={opt.value} style={[s.filterChip, filterType === opt.value && s.filterChipActive]} onPress={() => setFilterType(opt.value)}>
                            <ScaledText style={[s.filterChipText, filterType === opt.value && s.filterChipTextActive]}>{opt.label}</ScaledText>
                        </TouchableOpacity>
                    ))}
                </ScrollView>
                <TouchableOpacity style={s.addBtn} onPress={addNewCapsule}>
                    <Ionicons name="add-circle" size={18} color="#FFF" />
                    <ScaledText style={s.addBtnText}>{t('admin.inspo.addCapsule', 'Add Capsule')}</ScaledText>
                </TouchableOpacity>
            </View>
            {loading ? (
                <View style={s.center}>
                    <ActivityIndicator size="large" color="#007AFF" />
                </View>
            ) : filtered.length === 0 ? (
                <View style={s.center}>
                    <Ionicons name="images-outline" size={48} color="#8E8E93" />
                    <ScaledText style={s.emptyText}>{t('admin.inspo.empty', 'No inspiration items')}</ScaledText>
                </View>
            ) : (
                <FlatList data={filtered} keyExtractor={(item) => item.id} renderItem={renderInspoItem} contentContainerStyle={{ paddingBottom: 40 }} />
            )}

            <Modal visible={!!editTarget} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setEditTarget(null)}>
                <SafeAreaView style={s.modalContainer}>
                    <View style={s.modalHeader}>
                        <TouchableOpacity onPress={() => { setEditTarget(null); setEditLocalImage(null); }} disabled={savingEdit}>
                            <ScaledText style={s.modalCancel}>{t('common.cancel')}</ScaledText>
                        </TouchableOpacity>
                        <ScaledText style={s.modalTitle}>
                            {isCapsuleEdit
                                ? (editTarget?.kind === 'capsule' && editTarget.row.id ? t('admin.inspo.editCapsule', 'Edit Capsule') : t('admin.inspo.addCapsule', 'Add Capsule'))
                                : t('admin.manage.editItem', 'Edit Item')}
                        </ScaledText>
                        <TouchableOpacity onPress={saveEdit} disabled={savingEdit}>
                            {savingEdit ? <ActivityIndicator color="#007AFF" /> : <ScaledText style={s.modalSave}>{t('common.save')}</ScaledText>}
                        </TouchableOpacity>
                    </View>

                    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
                        <ScrollView style={s.modalBody} keyboardShouldPersistTaps="handled">
                            <TouchableOpacity style={s.imagePicker} onPress={pickEditImage}>
                                {editPreviewUri ? (
                                    <Image source={{ uri: editPreviewUri }} style={s.imagePreview} />
                                ) : (
                                    <View style={s.imagePlaceholder}>
                                        <Ionicons name="camera" size={32} color="#8E8E93" />
                                        <ScaledText style={s.imagePlaceholderText}>{t('admin.add.pickImage')}</ScaledText>
                                    </View>
                                )}
                            </TouchableOpacity>

                            {editTarget?.kind === 'capsule' && (
                                <>
                                    <ScaledText style={s.label}>{t('admin.guide.title')} *</ScaledText>
                                    <TextInput style={s.input} value={editTarget.row.title} onChangeText={(v) => setEditTarget({ kind: 'capsule', row: { ...editTarget.row, title: v } })} placeholder="Winter Dressing Guide" />
                                    <ScaledText style={s.label}>{t('admin.guide.subtitle')}</ScaledText>
                                    <TextInput style={s.input} value={editTarget.row.subtitle} onChangeText={(v) => setEditTarget({ kind: 'capsule', row: { ...editTarget.row, subtitle: v } })} placeholder="Optional subtitle" />
                                    <ScaledText style={s.label}>{t('admin.inspo.linkUrl', 'Link URL')}</ScaledText>
                                    <TextInput style={s.input} value={editTarget.row.link_url} onChangeText={(v) => setEditTarget({ kind: 'capsule', row: { ...editTarget.row, link_url: v } })} placeholder="https://..." autoCapitalize="none" autoCorrect={false} keyboardType="url" />
                                    <ScaledText style={s.label}>{t('admin.add.sortOrder')}</ScaledText>
                                    <TextInput style={s.input} value={String(editTarget.row.sort_order)} onChangeText={(v) => setEditTarget({ kind: 'capsule', row: { ...editTarget.row, sort_order: parseInt(v) || 0 } })} keyboardType="number-pad" />
                                    <View style={s.switchRow}>
                                        <ScaledText style={s.label}>{t('admin.add.isActive')}</ScaledText>
                                        <Switch value={editTarget.row.is_active} onValueChange={(v) => setEditTarget({ kind: 'capsule', row: { ...editTarget.row, is_active: v } })} />
                                    </View>
                                </>
                            )}

                            {editTarget?.kind === 'shop' && (
                                <>
                                    <ScaledText style={s.label}>{t('admin.add.brand')} *</ScaledText>
                                    <TextInput style={s.input} value={editTarget.row.brand} onChangeText={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, brand: v } })} />
                                    <ScaledText style={s.label}>{t('admin.add.name')} *</ScaledText>
                                    <TextInput style={s.input} value={editTarget.row.name} onChangeText={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, name: v } })} />
                                    <ScaledText style={s.label}>{t('admin.add.price')}</ScaledText>
                                    <TextInput style={s.input} value={String(editTarget.row.price)} onChangeText={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, price: parseFloat(v) || 0 } })} keyboardType="decimal-pad" />
                                    <ScaledText style={s.label}>{t('admin.add.description')}</ScaledText>
                                    <TextInput style={[s.input, s.textArea]} value={editTarget.row.description} onChangeText={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, description: v } })} multiline numberOfLines={3} />
                                    <ScaledText style={s.label}>{t('admin.add.sortOrder')}</ScaledText>
                                    <TextInput style={s.input} value={String(editTarget.row.sort_order)} onChangeText={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, sort_order: parseInt(v) || 0 } })} keyboardType="number-pad" />
                                    <View style={s.switchRow}>
                                        <ScaledText style={s.label}>{t('admin.add.isActive')}</ScaledText>
                                        <Switch value={editTarget.row.is_active} onValueChange={(v) => setEditTarget({ kind: 'shop', row: { ...editTarget.row, is_active: v } })} />
                                    </View>
                                </>
                            )}
                            <View style={{ height: 60 }} />
                        </ScrollView>
                    </KeyboardAvoidingView>
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
    cardImage: { width: 56, height: 72, borderRadius: 8, backgroundColor: '#E5E5EA' },
    cardInfo: { flex: 1, marginLeft: 10 },
    cardBrand: { fontSize: 12, fontWeight: '600', color: '#8E8E93', textTransform: 'uppercase' },
    cardName: { fontSize: 15, fontWeight: '500', color: '#1C1C1E', marginTop: 1 },
    cardMeta: { fontSize: 12, color: '#8E8E93', marginTop: 2 },
    cardActions: { gap: 6 },
    actionBtn: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6, paddingHorizontal: 8, borderRadius: 8, backgroundColor: '#F2F2F7' },
    actionBtnDisabled: { opacity: 0.5 },
    actionLabel: { fontSize: 12, fontWeight: '500', marginLeft: 4, color: '#007AFF' },
    toolbar: { flexDirection: 'row', alignItems: 'center', paddingRight: 12 },
    addBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#007AFF', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 18, marginLeft: 8 },
    addBtnText: { color: '#FFF', fontSize: 13, fontWeight: '600', marginLeft: 4 },
    modalContainer: { flex: 1, backgroundColor: '#F2F2F7' },
    modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#E5E5EA' },
    modalCancel: { fontSize: 16, color: '#007AFF' },
    modalTitle: { fontSize: 17, fontWeight: '600', color: '#1C1C1E' },
    modalSave: { fontSize: 16, fontWeight: '600', color: '#007AFF' },
    modalBody: { flex: 1, paddingHorizontal: 20 },
    imagePicker: { height: 180, borderRadius: 16, backgroundColor: '#E5E5EA', marginTop: 12, marginBottom: 8, overflow: 'hidden' },
    imagePreview: { width: '100%', height: '100%', resizeMode: 'cover' },
    imagePlaceholder: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    imagePlaceholderText: { fontSize: 14, color: '#8E8E93', marginTop: 6 },
    label: { fontSize: 13, fontWeight: '600', color: '#636366', marginTop: 12, marginBottom: 4 },
    input: { backgroundColor: '#FFF', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10, fontSize: 15, color: '#1C1C1E', borderWidth: 1, borderColor: '#E5E5EA' },
    textArea: { minHeight: 70, textAlignVertical: 'top' },
    switchRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 },
});
