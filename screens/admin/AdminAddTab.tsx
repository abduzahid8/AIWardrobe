/**
 * AdminAddTab — Form to add new clothing items to shop_catalog
 */
import React, { useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    Image,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
    StyleSheet,
    Switch,
    Text,
    TextInput,
    TouchableOpacity,
    View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { supabase } from '../../lib/supabase';
import { useTranslation } from 'react-i18next';
import { createLogger } from '../../src/utils/logger';

const logger = createLogger('AdminAddTab');

export const AdminAddTab = () => {
    const { t } = useTranslation();
    const [brand, setBrand] = useState('');
    const [name, setName] = useState('');
    const [price, setPrice] = useState('');
    const [currency, setCurrency] = useState('USD');
    const [garmentType, setGarmentType] = useState('upper_body');
    const [category, setCategory] = useState('tops');

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
    const [description, setDescription] = useState('');
    const [imageUrl, setImageUrl] = useState('');
    const [localImage, setLocalImage] = useState<string | null>(null);
    const [isActive, setIsActive] = useState(true);
    const [sortOrder, setSortOrder] = useState('0');
    const [submitting, setSubmitting] = useState(false);

    const pickImage = async () => {
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: 'images' as ImagePicker.MediaType,
            quality: 0.85,
            allowsEditing: true,
        });
        if (!result.canceled && result.assets[0] && result.assets[0].uri) {
            setLocalImage(result.assets[0].uri);
            setImageUrl('');
        }
    };

    const uploadImage = async (uri: string): Promise<string | null> => {
        try {
            const fileExt = uri.split('.').pop()?.toLowerCase() || 'jpg';
            const safeExt = ['jpg', 'jpeg', 'png', 'webp'].includes(fileExt) ? fileExt : 'jpg';
            const fileName = `admin-${Date.now()}.${safeExt}`;
            const filePath = `admin/${fileName}`;
            const contentType = `image/${safeExt === 'jpg' ? 'jpeg' : safeExt}`;

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

            const { data } = supabase.storage.from('shop-catalog').getPublicUrl(filePath);
            return data.publicUrl;
        } catch (err: any) { 
            logger.error('Upload exception', err); 
            Alert.alert(t('admin.add.uploadError'), err.message || 'Exception occurred');
            return null; 
        }
    };

    const handleSubmit = async () => {
        if (!name.trim() || !brand.trim()) {
            Alert.alert(t('admin.validation.required', 'Required'), t('admin.validation.nameBrand', 'Name and brand are required'));
            return;
        }
        setSubmitting(true);
        try {
            let finalImageUrl = imageUrl.trim();
            if (localImage && !finalImageUrl) {
                const uploaded = await uploadImage(localImage);
                if (!uploaded) { setSubmitting(false); return; }
                finalImageUrl = uploaded;
            }
            if (!finalImageUrl) {
                Alert.alert(t('admin.validation.required', 'Required'), t('admin.validation.imageRequired', 'Image URL or photo is required'));
                setSubmitting(false); return;
            }
            const id = `admin-${Date.now()}`;
            const { error } = await supabase.from('shop_catalog').insert({
                id, brand: brand.trim(), name: name.trim(), price: parseFloat(price) || 0,
                currency: currency.trim() || 'USD', category, garment_type: garmentType,
                description: description.trim(), image_url: finalImageUrl, is_active: isActive,
                sort_order: parseInt(sortOrder) || 0, source: 'admin',
            });
            if (error) { Alert.alert(t('common.error'), error.message); return; }
            Alert.alert(t('common.success'), t('admin.add.success'));
            setBrand(''); setName(''); setPrice(''); setDescription('');
            setImageUrl(''); setLocalImage(null); setIsActive(true); setSortOrder('0');
        } catch (err) { logger.error('Submit error', err); Alert.alert(t('common.error'), t('admin.add.failedToAddItem'));
        } finally { setSubmitting(false); }
    };

    return (
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
            <ScrollView style={s.scroll} keyboardShouldPersistTaps="handled">
                <TouchableOpacity style={s.imagePicker} onPress={pickImage}>
                    {localImage ? (
                        <Image source={{ uri: localImage }} style={s.imagePreview} />
                    ) : (
                        <View style={s.imagePlaceholder}>
                            <Ionicons name="camera" size={32} color="#8E8E93" />
                            <Text style={s.imagePlaceholderText}>{t('admin.add.pickImage')}</Text>
                        </View>
                    )}
                </TouchableOpacity>

                <Text style={s.label}>{t('admin.add.orImageUrl')}</Text>
                <TextInput style={s.input} value={imageUrl} onChangeText={setImageUrl} placeholder={t('admin.add.imageUrlPlaceholder')} autoCapitalize="none" autoCorrect={false} keyboardType="url" />

                <Text style={s.label}>{t('admin.add.brand')} *</Text>
                <TextInput style={s.input} value={brand} onChangeText={setBrand} placeholder={t('admin.add.brandPlaceholder')} />

                <Text style={s.label}>{t('admin.add.name')} *</Text>
                <TextInput style={s.input} value={name} onChangeText={setName} placeholder={t('admin.add.namePlaceholder')} />

                <View style={s.row}>
                    <View style={{ flex: 2 }}>
                        <Text style={s.label}>{t('admin.add.price')}</Text>
                        <TextInput style={s.input} value={price} onChangeText={setPrice} placeholder={t('admin.add.pricePlaceholder')} keyboardType="decimal-pad" />
                    </View>
                    <View style={{ flex: 1, marginLeft: 10 }}>
                        <Text style={s.label}>{t('admin.add.currency')}</Text>
                        <TextInput style={s.input} value={currency} onChangeText={setCurrency} placeholder={t('admin.add.currencyPlaceholder')} autoCapitalize="characters" maxLength={3} />
                    </View>
                </View>

                <Text style={s.label}>{t('admin.add.garmentType')}</Text>
                <View style={s.chipRow}>
                    {GARMENT_TYPES.map((gt) => (
                        <TouchableOpacity key={gt.value} style={[s.chip, garmentType === gt.value && s.chipActive]} onPress={() => setGarmentType(gt.value)}>
                            <Text style={[s.chipText, garmentType === gt.value && s.chipTextActive]}>{gt.label}</Text>
                        </TouchableOpacity>
                    ))}
                </View>

                <Text style={s.label}>{t('admin.add.category')}</Text>
                <View style={s.chipRow}>
                    {CATEGORIES.map((c) => (
                        <TouchableOpacity key={c.value} style={[s.chip, category === c.value && s.chipActive]} onPress={() => setCategory(c.value)}>
                            <Text style={[s.chipText, category === c.value && s.chipTextActive]}>{c.label}</Text>
                        </TouchableOpacity>
                    ))}
                </View>

                <Text style={s.label}>{t('admin.add.description')}</Text>
                <TextInput style={[s.input, s.textArea]} value={description} onChangeText={setDescription} placeholder={t('admin.add.descriptionPlaceholder')} multiline numberOfLines={3} />

                <Text style={s.label}>{t('admin.add.sortOrder')}</Text>
                <TextInput style={s.input} value={sortOrder} onChangeText={setSortOrder} placeholder={t('admin.add.sortOrderPlaceholder')} keyboardType="number-pad" />

                <View style={s.switchRow}>
                    <Text style={s.label}>{t('admin.add.isActive')}</Text>
                    <Switch value={isActive} onValueChange={setIsActive} />
                </View>

                <TouchableOpacity style={[s.submitBtn, submitting && s.submitBtnDisabled]} onPress={handleSubmit} disabled={submitting}>
                    {submitting ? <ActivityIndicator color="#FFF" /> : <Text style={s.submitBtnText}>{t('admin.add.addItem')}</Text>}
                </TouchableOpacity>
                <View style={{ height: 40 }} />
            </ScrollView>
        </KeyboardAvoidingView>
    );
};

const s = StyleSheet.create({
    scroll: { flex: 1, paddingHorizontal: 20 },
    imagePicker: { height: 160, borderRadius: 16, backgroundColor: '#E5E5EA', marginBottom: 16, overflow: 'hidden' },
    imagePreview: { width: '100%', height: '100%', resizeMode: 'cover' },
    imagePlaceholder: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    imagePlaceholderText: { fontSize: 14, color: '#8E8E93', marginTop: 6 },
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
    submitBtn: { backgroundColor: '#007AFF', borderRadius: 14, paddingVertical: 16, alignItems: 'center', marginTop: 24 },
    submitBtnDisabled: { opacity: 0.6 },
    submitBtnText: { fontSize: 17, fontWeight: '600', color: '#FFF' },
});
