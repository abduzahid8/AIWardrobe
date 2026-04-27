/**
 * AdminInspoTab — Browse inspiration items, change photos, promote to shop
 */
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    FlatList,
    Image,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
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

export const AdminInspoTab = () => {
    const { t } = useTranslation();
    const [inspoItems, setInspoItems] = useState<ShopCatalogItem[]>(INSPO_MENS_SHOP_ITEMS);
    const [loading, setLoading] = useState(false);
    const [filterType, setFilterType] = useState('all');
    const [promoting, setPromoting] = useState<string | null>(null);

    // Fetch items from shop_catalog (includes capsule items from DB)
    useEffect(() => {
        loadDbItems();
    }, []);

    const loadDbItems = async () => {
        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('shop_catalog')
                .select('id, brand, name, price, currency, category, garment_type, description, image_url')
                .eq('is_active', true)
                .order('sort_order', { ascending: true });
            
            if (error) throw error;
            
            const dbItems: ShopCatalogItem[] = (data || []).map((item) => ({
                id: item.id,
                brand: item.brand,
                name: item.name,
                price: item.price,
                currency: item.currency,
                garmentType: item.garment_type as ShopCatalogItem['garmentType'],
                description: item.description,
                imageUrl: addCacheBust(item.image_url),
            }));

            // Merge: DB items take precedence, add hardcoded items not in DB
            const dbIds = new Set(dbItems.map(i => i.id));
            const localItems = INSPO_MENS_SHOP_ITEMS.filter(i => !dbIds.has(i.id));
            
            const { data: capsulesData } = await supabase
                .from('featured_capsules')
                .select('*')
                .eq('is_active', true)
                .order('sort_order', { ascending: true });

            const mappedCapsules: ShopCatalogItem[] = (capsulesData || []).map((c) => ({
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
                let errorMsg = 'Upload failed';
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
        const { error: upsertError } = await supabase.from('shop_catalog').upsert({
            id, brand: item.brand, name: item.name, price: item.price,
            currency: item.currency || 'USD', category, garment_type: item.garmentType,
            description: item.description || '', image_url: nextPublicUrl,
            is_active: true, sort_order: 0, source: 'inspo',
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
                <Text style={s.cardBrand}>{item.brand}</Text>
                <Text style={s.cardName} numberOfLines={1}>{item.name}</Text>
                <Text style={s.cardMeta}>{item.garmentType} · ${item.price}</Text>
            </View>
            <View style={s.cardActions}>
                <TouchableOpacity style={s.actionBtn} onPress={() => changePhoto(item)}>
                    <Ionicons name="image" size={18} color="#007AFF" />
                    <Text style={s.actionLabel}>{t('admin.inspo.changePhoto', 'Photo')}</Text>
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
                    <Text style={[s.actionLabel, { color: '#34C759' }]}>{t('admin.inspo.promote')}</Text>
                </TouchableOpacity>
            </View>
        </View>
    );

    const FILTER_OPTIONS = [{ value: 'all', label: t('common.all') }, ...GARMENT_TYPES];

    return (
        <View style={s.container}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.filterRow}>
                {FILTER_OPTIONS.map((opt) => (
                    <TouchableOpacity key={opt.value} style={[s.filterChip, filterType === opt.value && s.filterChipActive]} onPress={() => setFilterType(opt.value)}>
                        <Text style={[s.filterChipText, filterType === opt.value && s.filterChipTextActive]}>{opt.label}</Text>
                    </TouchableOpacity>
                ))}
            </ScrollView>
            {loading ? (
                <View style={s.center}>
                    <ActivityIndicator size="large" color="#007AFF" />
                </View>
            ) : filtered.length === 0 ? (
                <View style={s.center}>
                    <Ionicons name="images-outline" size={48} color="#8E8E93" />
                    <Text style={s.emptyText}>{t('admin.inspo.empty', 'No inspiration items')}</Text>
                </View>
            ) : (
                <FlatList data={filtered} keyExtractor={(item) => item.id} renderItem={renderInspoItem} contentContainerStyle={{ paddingBottom: 40 }} />
            )}
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
});
