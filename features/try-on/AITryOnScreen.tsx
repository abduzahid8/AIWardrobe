/**
 * AITryOnScreen — Slim composition layer
 *
 * Business logic lives in:
 *   - hooks/usePhotoPicker.ts
 *   - hooks/useTryOnWizard.ts
 *   - hooks/useTryOnAPI.ts
 *
 * Styles live in: ./styles.ts
 * Types live in:  ./types.ts
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Image, ActivityIndicator, Alert, SafeAreaView, Dimensions, StyleSheet, TextInput, KeyboardAvoidingView, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect, useRoute, RouteProp } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { supabase } from '../../lib/supabase';
import useAuthStore from '../../store/auth';
import useSubscriptionStore from '../../store/subscriptionStore';
import AppColors from '../../constants/AppColors';
import LiquidGlass2026Theme from '../../constants/LiquidGlass2026Theme';
import { WebView } from 'react-native-webview';

import { usePhotoPicker } from './hooks/usePhotoPicker';
import { useTryOnWizard } from './hooks/useTryOnWizard';
import { useTryOnAPI } from './hooks/useTryOnAPI';
import type { WardrobeItem, ShopCatalogItem } from './types';
import { SHOP_CATALOG_ITEMS, SHOP_CATEGORIES } from '../../data/shopCatalogItems';
import styles from './styles';
import { generate3Dhtml, BODY_TYPES, BodyTypeId } from './utils/mannequin3D';
import { MANNEQUIN_MODEL_URL, MANNEQUIN_USE_PROCEDURAL_FALLBACK } from './utils/mannequinConfig';
import { getMannequinBase64 } from './hooks/useMannequin';

type AITryOnRouteParams = { asTab?: boolean };

const AITryOnScreen = () => {
    const navigation = useNavigation();
    const route = useRoute<RouteProp<{ params: AITryOnRouteParams }, 'params'>>();
    const asTab = (route.params as AITryOnRouteParams)?.asTab === true;
    const { user } = useAuthStore();
    const { isPremium } = useSubscriptionStore();
    const { t } = useTranslation();

    // Hooks
    const {
        humanImage, clothImage, setClothImage,
        pickFullLengthPhoto, pickGarmentPhoto,
        showFullLengthPhotoOptions, showGarmentPhotoOptions,
    } = usePhotoPicker();

    const {
        tryOnMode, setTryOnMode,
        tryOnStep, goToStep,
        activeTab, setActiveTab,
    } = useTryOnWizard();

    const { loading, saving, resultImage, isMock, handleTryOn, handleSaveToWardrobe } = useTryOnAPI();

    // ── Mannequin Try-On State ───────────────────────────────────────────
    // Uses fashn.ai tryon-v1.6 API:
    //   model_image  = mannequin_front.png (the grey mannequin asset)
    //   garment_image = selected shop item
    // Result: AI renders the mannequin actually wearing the garment.
    const [mannequinShopItem, setMannequinShopItem] = useState<ShopCatalogItem | null>(null);
    const [mannequinShopFilter, setMannequinShopFilter] = useState<string>('upper_body');
    const [mannequinLoading, setMannequinLoading] = useState(false);
    const [mannequinResult, setMannequinResult] = useState<string | null>(null);
    const [mannequinSaving, setMannequinSaving] = useState(false);


    const handleMannequinTryOn = async () => {
        if (!mannequinShopItem) return;
        if (!user) {
            Alert.alert('Sign in required', 'Please sign in to use the virtual try-on.');
            return;
        }

        setMannequinLoading(true);
        setMannequinResult(null);

        try {
            // Convert local mannequin asset to base64 — works offline, no Supabase URL needed.
            // Both fashn.ai and Replicate accept data URIs: data:image/png;base64,...
            const mannequinBase64 = await getMannequinBase64(
                require('../../assets/images/mannequin_front.png')
            );

            const fashnKey = process.env.EXPO_PUBLIC_FASHN_API_KEY;
            const replicateToken = process.env.EXPO_PUBLIC_REPLICATE_TOKEN;

            // ── Path A: fashn.ai (purpose-built for mannequin try-on) ──────
            if (fashnKey) {
                const startRes = await fetch('https://api.fashn.ai/v1/run', {
                    method: 'POST',
                    headers: {
                        Authorization: `Bearer ${fashnKey}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model_name: 'tryon-v1.6',
                        inputs: {
                            model_image: mannequinBase64,
                            garment_image: mannequinShopItem.imageUrl,
                            garment_photo_type: 'flat_lay',
                        },
                    }),
                });
                if (!startRes.ok) throw new Error(await startRes.text());
                const { id } = await startRes.json();

                // Poll status
                let output: string | null = null;
                for (let i = 0; i < 40; i++) {
                    await new Promise(r => setTimeout(r, 2000));
                    const poll = await fetch(`https://api.fashn.ai/v1/status/${id}`, {
                        headers: { Authorization: `Bearer ${fashnKey}` },
                    });
                    const status = await poll.json();
                    if (status.status === 'completed' && status.output?.image_url) {
                        output = status.output.image_url;
                        break;
                    }
                    if (status.status === 'failed') throw new Error(status.error || 'fashn.ai failed');
                }
                if (!output) throw new Error('Timed out — please try again.');
                setMannequinResult(output);

            // ── Path B: Replicate IDM-VTON with mannequin reference ────────
            } else if (replicateToken) {
                const startRes = await fetch('https://api.replicate.com/v1/predictions', {
                    method: 'POST',
                    headers: {
                        Authorization: `Token ${replicateToken}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        version: '0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985',
                        input: {
                            human_img: mannequinBase64,
                            garm_img: mannequinShopItem.imageUrl,
                            garment_des: mannequinShopItem.name || 'clothing',
                            category: mannequinShopItem.garmentType === 'lower_body' ? 'lower_body' : 'upper_body',
                            n_samples: 1,
                            seed: 42,
                        },
                    }),
                });
                if (!startRes.ok) throw new Error(await startRes.text());
                let result = await startRes.json();

                while (result.status !== 'succeeded' && result.status !== 'failed') {
                    await new Promise(r => setTimeout(r, 2000));
                    const poll = await fetch(result.urls.get, {
                        headers: { Authorization: `Token ${replicateToken}` },
                    });
                    result = await poll.json();
                }
                if (result.status === 'failed') throw new Error(result.error || 'Replicate failed');
                const output = Array.isArray(result.output) ? result.output[0] : result.output;
                setMannequinResult(output);

            } else {
                throw new Error('No API key configured. Add EXPO_PUBLIC_FASHN_API_KEY or EXPO_PUBLIC_REPLICATE_TOKEN to .env');
            }
        } catch (err: any) {
            Alert.alert('Try-On Failed', err?.message || 'Please try again.');
        } finally {
            setMannequinLoading(false);
        }
    };

    const handleMannequinSave = async () => {
        if (!user || !mannequinResult) return;
        setMannequinSaving(true);
        try {
            const { error } = await supabase.from('clothing_items').insert({
                user_id: user.id,
                type: 'AI Try-On Result',
                color: 'Mixed',
                style: 'Casual',
                description: `Mannequin try-on — ${mannequinShopItem?.name || 'outfit'}`,
                season: 'All Seasons',
                image_url: mannequinResult,
                category: 'outfit',
            });
            if (error) throw error;
            Alert.alert(t('aiTryOn.savedTitle'), t('aiTryOn.savedMessage'));
        } catch {
            Alert.alert(t('aiTryOn.errorTitle'), t('aiTryOn.saveFailed'));
        } finally {
            setMannequinSaving(false);
        }
    };


    // ── Try Yourself State ──────────────────────────────────────────────
    const [selectedShopItemForSelf, setSelectedShopItemForSelf] = useState<ShopCatalogItem | null>(null);
    const [selfShopFilter, setSelfShopFilter] = useState<string>('all');

    // Wardrobe items (local state — lightweight)
    const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
    const [loadingWardrobe, setLoadingWardrobe] = useState(false);
    const [selectedWardrobeItem, setSelectedWardrobeItem] = useState<WardrobeItem | null>(null);

    const getGarmentType = (): string => {
        if (selectedShopItemForSelf) return selectedShopItemForSelf.garmentType;
        const cat = (selectedWardrobeItem?.category || selectedWardrobeItem?.type || '').toLowerCase();
        if (['pants', 'jeans', 'shorts', 'skirt', 'lower'].some((k) => cat.includes(k))) return 'lower_body';
        if (['dress', 'full'].some((k) => cat.includes(k))) return 'dresses';
        return 'upper_body';
    };

    const getFilteredShopItems = (filter: string) =>
        filter === 'all'
            ? SHOP_CATALOG_ITEMS
            : SHOP_CATALOG_ITEMS.filter((i) => i.garmentType === filter);

    const loadWardrobeItems = useCallback(async () => {
        try {
            if (!user) return;
            setLoadingWardrobe(true);
            const { data, error } = await supabase
                .from('clothing_items')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });
            if (error) throw error;
            if (data) {
                const items: WardrobeItem[] = data.map((item: any) => ({
                    id: item.id,
                    imageUrl: item.image_url,
                    type: item.type, category: item.category, color: item.color,
                }));
                const tryable = items.filter((i) => {
                    const cat = (i.category || i.type || '').toLowerCase();
                    return ['shirt', 'top', 'dress', 'jacket', 'blouse', 'sweater', 'upper'].some((k) => cat.includes(k));
                });
                setWardrobeItems(tryable.length > 0 ? tryable : items.slice(0, 20));
            }
        } catch (err) {
        } finally {
            setLoadingWardrobe(false);
        }
    }, [user]);

    useFocusEffect(
        useCallback(() => {
            if (activeTab === 'wardrobe') loadWardrobeItems();
        }, [activeTab, loadWardrobeItems])
    );

    const handleSelectWardrobeItem = (item: WardrobeItem) => {
        setSelectedWardrobeItem(item);
        if (item.imageUrl) setClothImage(item.imageUrl);
    };

    // ─── RENDER ──────────────────────────────────────
    return (
        <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === "ios" ? "padding" : "height"}>
            <SafeAreaView style={styles.container}>
                {/* Header */}
                <View style={[styles.header, { justifyContent: 'center' }]}>
                    {asTab ? (
                        <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                            <Text style={styles.headerTitle}>{t('aiTryOn.title')}</Text>
                        </View>
                    ) : (
                        <>
                            <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }} style={{ position: 'absolute', left: 20, zIndex: 10 }}>
                                <Ionicons name="chevron-back" size={28} color="#0A1931" />
                            </TouchableOpacity>
                            <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                                <Text style={styles.headerTitle}>{t('aiTryOn.title')}</Text>
                            </View>
                        </>
                    )}
                </View>

                {/* Segmented Control */}
                <View style={styles.segmentContainer}>
                    <View style={styles.modeToggleWrap}>
                        {(['try your self', 'model'] as const).map((mode) => (
                            <TouchableOpacity
                                key={mode}
                                style={[styles.modeToggleOption, tryOnMode === mode && styles.modeToggleOptionActive]}
                                onPress={() => setTryOnMode(mode)}
                                activeOpacity={0.8}
                            >
                                <Text style={[styles.modeToggleText, tryOnMode === mode && styles.modeToggleTextActive]}>
                                    {mode === 'try your self' ? 'Try your self' : '3D Body Model'}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                <ScrollView contentContainerStyle={[styles.scrollContent, { paddingBottom: 60 }]} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
                    {tryOnMode === 'model' ? (
                        /* ── Model Mode — AI Mannequin Try-On ── */
                        <>
                            {/* Mannequin preview card */}
                            <View style={[styles.mannequinCard, { minHeight: 380, overflow: 'hidden', justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8F9FB' }]}>

                                {mannequinResult ? (
                                    /* ── AI Result ── */
                                    <>
                                        <Image
                                            source={{ uri: mannequinResult }}
                                            style={{ width: '100%', height: '100%', borderRadius: 20 }}
                                            resizeMode="contain"
                                        />
                                        {/* Close result */}
                                        <TouchableOpacity
                                            onPress={() => setMannequinResult(null)}
                                            style={{ position: 'absolute', top: 12, right: 12, backgroundColor: 'rgba(0,0,0,0.45)', borderRadius: 20, padding: 6 }}
                                        >
                                            <Ionicons name="close" size={18} color="#fff" />
                                        </TouchableOpacity>
                                        <View style={{ position: 'absolute', top: 12, left: 12, backgroundColor: 'rgba(0,85,255,0.9)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 10 }}>
                                            <Text style={{ color: '#fff', fontSize: 11, fontWeight: '700' }}>✨ AI Mannequin Try-On</Text>
                                        </View>
                                    </>
                                ) : mannequinLoading ? (
                                    /* ── Generating ── */
                                    <View style={{ alignItems: 'center', gap: 16 }}>
                                        <ActivityIndicator size="large" color={AppColors.primary} />
                                        <Text style={{ fontSize: 14, fontWeight: '700', color: AppColors.primary }}>Generating try-on…</Text>
                                        <Text style={{ fontSize: 12, color: AppColors.textMuted, textAlign: 'center', paddingHorizontal: 24 }}>
                                            The AI is placing the garment on the mannequin. This takes ~20–40 seconds.
                                        </Text>
                                    </View>
                                ) : (
                                    /* ── Default: mannequin preview + optional selected garment thumbnail ── */
                                    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 20 }}>
                                        <View style={{ position: 'relative' }}>
                                            <Image
                                                source={require('../../assets/images/mannequin_front.png')}
                                                style={{ width: 180, height: 300, borderRadius: 12 }}
                                                resizeMode="contain"
                                            />
                                            {/* Garment badge in corner when selected */}
                                            {mannequinShopItem && (
                                                <View style={{ position: 'absolute', top: -8, right: -8, borderRadius: 12, overflow: 'hidden', borderWidth: 2, borderColor: AppColors.primary }}>
                                                    <Image
                                                        source={{ uri: mannequinShopItem.imageUrl }}
                                                        style={{ width: 56, height: 56 }}
                                                        resizeMode="cover"
                                                    />
                                                </View>
                                            )}
                                        </View>
                                        {mannequinShopItem ? (
                                            <Text style={{ marginTop: 12, fontSize: 13, fontWeight: '700', color: AppColors.primary }}>
                                                {mannequinShopItem.brand} — {mannequinShopItem.name}
                                            </Text>
                                        ) : (
                                            <Text style={{ marginTop: 12, fontSize: 13, color: AppColors.textMuted }}>
                                                Select a garment below, then tap Try On
                                            </Text>
                                        )}
                                    </View>
                                )}
                            </View>

                            {/* Try On button */}
                            <View style={{ paddingHorizontal: 20, marginTop: 16 }}>
                                <TouchableOpacity
                                    style={[styles.mannequinGenerateButton, (!mannequinShopItem || mannequinLoading) && styles.mannequinGenerateButtonDisabled]}
                                    onPress={handleMannequinTryOn}
                                    disabled={!mannequinShopItem || mannequinLoading}
                                    activeOpacity={0.85}
                                >
                                    <Ionicons name="sparkles" size={20} color="#fff" />
                                    <Text style={styles.mannequinGenerateButtonText}>
                                        {mannequinLoading
                                            ? 'Generating…'
                                            : mannequinShopItem
                                                ? `✨ Try On — ${mannequinShopItem.brand}`
                                                : 'Select a garment first'}
                                    </Text>
                                </TouchableOpacity>

                                {/* Save result */}
                                {mannequinResult && !mannequinLoading && (
                                    <TouchableOpacity
                                        style={[styles.mannequinSaveButton, { marginTop: 12 }]}
                                        onPress={handleMannequinSave}
                                        disabled={mannequinSaving}
                                    >
                                        <Ionicons name="heart" size={18} color="#fff" />
                                        <Text style={styles.mannequinSaveButtonText}>
                                            {mannequinSaving ? t('aiTryOn.saving') : t('aiTryOn.saveToWardrobe')}
                                        </Text>
                                    </TouchableOpacity>
                                )}
                            </View>

                            {/* Shop Catalog */}
                            <Text style={[styles.shopSectionLabel, { marginTop: 24 }]}>Choose Garment</Text>

                            {/* Category filter */}
                            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.shopFilterRow}>
                                {SHOP_CATEGORIES.filter((cat) => cat.key !== 'all').map((cat) => (
                                    <TouchableOpacity
                                        key={cat.key}
                                        style={[styles.shopFilterChip, mannequinShopFilter === cat.key && styles.shopFilterChipActive]}
                                        onPress={() => { setMannequinShopFilter(cat.key); setMannequinShopItem(null); setMannequinResult(null); }}
                                        activeOpacity={0.8}
                                    >
                                        <Text style={[styles.shopFilterChipText, mannequinShopFilter === cat.key && styles.shopFilterChipTextActive]}>
                                            {cat.label}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>

                            {/* Items grid */}
                            <View style={styles.shopCatalogGrid}>
                                {getFilteredShopItems(mannequinShopFilter).map((item) => {
                                    const isSelected = mannequinShopItem?.id === item.id;
                                    return (
                                        <TouchableOpacity
                                            key={item.id}
                                            style={[styles.shopItemCard, isSelected && styles.shopItemCardSelected]}
                                            onPress={() => { setMannequinShopItem(isSelected ? null : item); setMannequinResult(null); }}
                                            activeOpacity={0.85}
                                        >
                                            <Image source={{ uri: item.imageUrl }} style={styles.shopItemImage} />
                                            <View style={styles.shopItemInfo}>
                                                <Text style={styles.shopItemBrand}>{item.brand}</Text>
                                                <Text style={styles.shopItemName} numberOfLines={1}>{item.name}</Text>
                                                <Text style={styles.shopItemPrice}>${item.price.toFixed(2)}</Text>
                                            </View>
                                            {isSelected && (
                                                <View style={styles.shopItemSelectedBadge}>
                                                    <Ionicons name="checkmark-circle" size={22} color="#0055FF" />
                                                </View>
                                            )}
                                        </TouchableOpacity>
                                    );
                                })}
                            </View>
                        </>
                    ) : (
                        /* ── Try Your Self Mode (Wizard) ── */
                        <>
                            {/* Step 1 — Full-length photo */}
                            {tryOnStep === 1 && (
                                <>
                                    <Text style={styles.stepLabel}>1. Your full-length photo</Text>
                                    <Text style={styles.stepHint}>Stand clearly with your full body in frame for best results.</Text>
                                    <View style={styles.fullLengthCard}>
                                        {humanImage ? (
                                            <TouchableOpacity onPress={() => showFullLengthPhotoOptions(() => goToStep(2))} style={{ flex: 1 }} activeOpacity={0.85}>
                                                <Image source={{ uri: humanImage }} style={styles.fullLengthImage} />
                                            </TouchableOpacity>
                                        ) : (
                                            <View style={styles.fullLengthPlaceholder}>
                                                <View style={styles.placeholderIconWrap}>
                                                    <View style={styles.placeholderIconCircle}>
                                                        <Ionicons name="person" size={56} color="#0055FF" />
                                                    </View>
                                                </View>
                                                <Text style={styles.placeholderTitle}>Add full-length photo</Text>
                                                <Text style={styles.placeholderSub}>Camera or gallery — full body works best</Text>
                                                <View style={styles.photoOptionsRow}>
                                                    <TouchableOpacity style={styles.photoOption} onPress={() => pickFullLengthPhoto('camera', () => goToStep(2))} activeOpacity={0.8}>
                                                        <View style={styles.photoOptionIconWrap}>
                                                            <Ionicons name="camera" size={22} color="#0055FF" />
                                                        </View>
                                                        <Text style={styles.photoOptionText}>Camera</Text>
                                                    </TouchableOpacity>
                                                    <TouchableOpacity style={styles.photoOption} onPress={() => pickFullLengthPhoto('library', () => goToStep(2))} activeOpacity={0.8}>
                                                        <View style={styles.photoOptionIconWrap}>
                                                            <Ionicons name="image" size={22} color="#0055FF" />
                                                        </View>
                                                        <Text style={styles.photoOptionText}>Gallery</Text>
                                                    </TouchableOpacity>
                                                </View>
                                            </View>
                                        )}
                                    </View>
                                </>
                            )}

                            {/* Step 2 — Garment selection */}
                            {tryOnStep === 2 && (
                                <>
                                    <Text style={styles.stepLabel}>2. Outfit to try on</Text>
                                    <View style={styles.tabContainer}>
                                        {(['upload', 'wardrobe', 'shop'] as const).map((tab) => (
                                            <TouchableOpacity
                                                key={tab}
                                                style={[styles.tab, activeTab === tab && styles.tabActive]}
                                                onPress={() => {
                                                    setActiveTab(tab);
                                                    if (tab !== 'wardrobe') setSelectedWardrobeItem(null);
                                                    if (tab !== 'shop') setSelectedShopItemForSelf(null);
                                                }}
                                            >
                                                <Ionicons
                                                    name={tab === 'upload' ? 'cloud-upload-outline' : tab === 'wardrobe' ? 'shirt-outline' : 'bag-handle-outline'}
                                                    size={15}
                                                    color={activeTab === tab ? '#fff' : AppColors.textMuted}
                                                />
                                                <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
                                                    {tab === 'upload' ? t('aiTryOn.upload') : tab === 'wardrobe' ? t('aiTryOn.myWardrobe') : 'Shop'}
                                                </Text>
                                            </TouchableOpacity>
                                        ))}
                                    </View>

                                    {activeTab === 'upload' && (
                                        <View style={styles.garmentCard}>
                                            {clothImage && !selectedWardrobeItem && !selectedShopItemForSelf ? (
                                                <TouchableOpacity onPress={showGarmentPhotoOptions} style={{ flex: 1 }} activeOpacity={0.85}>
                                                    <Image source={{ uri: clothImage }} style={styles.garmentImage} />
                                                </TouchableOpacity>
                                            ) : (
                                                <View style={styles.fullLengthPlaceholder}>
                                                    <View style={styles.placeholderIconWrap}>
                                                        <View style={styles.placeholderIconCircle}>
                                                            <Ionicons name="shirt" size={52} color="#0055FF" />
                                                        </View>
                                                    </View>
                                                    <Text style={styles.placeholderTitle}>Add clothing photo</Text>
                                                    <Text style={styles.placeholderSub}>Camera or gallery — item on flat surface</Text>
                                                    <View style={styles.photoOptionsRow}>
                                                        <TouchableOpacity style={styles.photoOption} onPress={() => pickGarmentPhoto('camera')} activeOpacity={0.8}>
                                                            <View style={styles.photoOptionIconWrap}>
                                                                <Ionicons name="camera" size={22} color="#0055FF" />
                                                            </View>
                                                            <Text style={styles.photoOptionText}>Camera</Text>
                                                        </TouchableOpacity>
                                                        <TouchableOpacity style={styles.photoOption} onPress={() => pickGarmentPhoto('library')} activeOpacity={0.8}>
                                                            <View style={styles.photoOptionIconWrap}>
                                                                <Ionicons name="image" size={22} color="#0055FF" />
                                                            </View>
                                                            <Text style={styles.photoOptionText}>Gallery</Text>
                                                        </TouchableOpacity>
                                                    </View>
                                                </View>
                                            )}
                                        </View>
                                    )}

                                    {activeTab === 'wardrobe' && (
                                        <View style={styles.wardrobeSection}>
                                            {loadingWardrobe ? (
                                                <View style={styles.wardrobeLoading}>
                                                    <ActivityIndicator size="small" color={AppColors.primary} />
                                                    <Text style={styles.wardrobeLoadingText}>{t('aiTryOn.loadingWardrobe')}</Text>
                                                </View>
                                            ) : wardrobeItems.length === 0 ? (
                                                <View style={styles.wardrobeEmpty}>
                                                    <Ionicons name="shirt-outline" size={36} color={AppColors.textLight} />
                                                    <Text style={styles.wardrobeEmptyText}>{t('aiTryOn.noWardrobeItems')}</Text>
                                                    <TouchableOpacity style={styles.scanButton} onPress={() => (navigation as any).navigate('WardrobeVideo')}>
                                                        <Text style={styles.scanButtonText}>{t('aiTryOn.scanWardrobe')}</Text>
                                                    </TouchableOpacity>
                                                </View>
                                            ) : (
                                                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.wardrobeScroll}>
                                                    {wardrobeItems.map((item) => {
                                                        const isSelected = selectedWardrobeItem?.id === item.id;
                                                        return (
                                                            <TouchableOpacity key={item.id} style={[styles.wardrobeItemCard, isSelected && styles.wardrobeItemCardSelected]} onPress={() => handleSelectWardrobeItem(item)}>
                                                                {item.imageUrl ? (
                                                                    <Image source={{ uri: item.imageUrl }} style={styles.wardrobeItemImage} />
                                                                ) : (
                                                                    <View style={styles.wardrobeItemPlaceholder}>
                                                                        <Ionicons name="shirt-outline" size={24} color={AppColors.textLight} />
                                                                    </View>
                                                                )}
                                                                {isSelected && (
                                                                    <View style={styles.selectedBadge}>
                                                                        <Ionicons name="checkmark-circle" size={20} color="#34C759" />
                                                                    </View>
                                                                )}
                                                            </TouchableOpacity>
                                                        );
                                                    })}
                                                </ScrollView>
                                            )}
                                            {selectedWardrobeItem && (
                                                <View style={styles.selectedInfo}>
                                                    <Ionicons name="checkmark-circle" size={16} color="#34C759" />
                                                    <Text style={styles.selectedInfoText}>
                                                        {selectedWardrobeItem.type || selectedWardrobeItem.category || 'Item'} {t('aiTryOn.selected')}
                                                    </Text>
                                                </View>
                                            )}
                                        </View>
                                    )}

                                    {activeTab === 'shop' && (
                                        <View>
                                            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={[styles.shopFilterRow, { marginBottom: 14 }]}>
                                                {SHOP_CATEGORIES.map((cat) => (
                                                    <TouchableOpacity
                                                        key={cat.key}
                                                        style={[styles.shopFilterChip, selfShopFilter === cat.key && styles.shopFilterChipActive]}
                                                        onPress={() => setSelfShopFilter(cat.key)}
                                                        activeOpacity={0.8}
                                                    >
                                                        <Text style={[styles.shopFilterChipText, selfShopFilter === cat.key && styles.shopFilterChipTextActive]}>
                                                            {cat.label}
                                                        </Text>
                                                    </TouchableOpacity>
                                                ))}
                                            </ScrollView>
                                            <View style={styles.shopCatalogGrid}>
                                                {getFilteredShopItems(selfShopFilter).map((item) => {
                                                    const isSelected = selectedShopItemForSelf?.id === item.id;
                                                    return (
                                                        <TouchableOpacity
                                                            key={item.id}
                                                            style={[styles.shopItemCard, isSelected && styles.shopItemCardSelected]}
                                                            onPress={() => {
                                                                const next = isSelected ? null : item;
                                                                setSelectedShopItemForSelf(next);
                                                                setClothImage(next ? next.imageUrl : null);
                                                            }}
                                                            activeOpacity={0.85}
                                                        >
                                                            <Image source={{ uri: item.imageUrl }} style={styles.shopItemImage} />
                                                            <View style={styles.shopItemInfo}>
                                                                <Text style={styles.shopItemBrand}>{item.brand}</Text>
                                                                <Text style={styles.shopItemName} numberOfLines={1}>{item.name}</Text>
                                                                <Text style={styles.shopItemPrice}>${item.price.toFixed(2)}</Text>
                                                            </View>
                                                            {isSelected && (
                                                                <View style={styles.shopItemSelectedBadge}>
                                                                    <Ionicons name="checkmark-circle" size={22} color="#0055FF" />
                                                                </View>
                                                            )}
                                                        </TouchableOpacity>
                                                    );
                                                })}
                                            </View>
                                            {selectedShopItemForSelf && (
                                                <View style={styles.selectedInfo}>
                                                    <Ionicons name="checkmark-circle" size={16} color="#0055FF" />
                                                    <Text style={[styles.selectedInfoText, { color: '#0055FF' }]}>
                                                        {selectedShopItemForSelf.name} — {selectedShopItemForSelf.brand}
                                                    </Text>
                                                </View>
                                            )}
                                        </View>
                                    )}

                                    <View style={styles.wizardNavigation}>
                                        <TouchableOpacity style={styles.secondaryButton} onPress={() => goToStep(1)}>
                                            <Text style={styles.secondaryButtonText}>Back</Text>
                                        </TouchableOpacity>
                                        <TouchableOpacity
                                            style={[styles.primaryButtonFlex, (!clothImage && !selectedWardrobeItem && !selectedShopItemForSelf) && styles.primaryButtonDisabled]}
                                            onPress={() => goToStep(3)}
                                            disabled={!clothImage && !selectedWardrobeItem && !selectedShopItemForSelf}
                                        >
                                            <Text style={styles.primaryButtonText}>Continue</Text>
                                        </TouchableOpacity>
                                    </View>
                                </>
                            )}

                            {/* Step 3 — Preview & Generate */}
                            {tryOnStep === 3 && (
                                <>
                                    <Text style={styles.stepLabel}>3. Preview</Text>
                                    <View style={styles.resultContainer}>
                                        {loading ? (
                                            <View style={styles.loadingBox}>
                                                <ActivityIndicator size="large" color={AppColors.primary} />
                                                <Text style={styles.loadingText}>{t('aiTryOn.generating')}</Text>
                                                <Text style={styles.loadingSub}>{t('aiTryOn.takesTime')}</Text>
                                            </View>
                                        ) : resultImage ? (
                                            <View style={{ flex: 1, backgroundColor: isMock ? '#E2E8F0' : 'transparent', justifyContent: 'center' }}>
                                                <Image source={{ uri: resultImage }} style={isMock ? [styles.resultImage, { opacity: 0.9, resizeMode: 'contain' }] : styles.resultImage} />
                                                {isMock && (
                                                    <View style={{ position: 'absolute', top: 10, left: 0, right: 0, alignItems: 'center' }}>
                                                        <View style={{ backgroundColor: 'rgba(0,0,0,0.6)', padding: 6, borderRadius: 8 }}>
                                                            <Text style={{ color: '#fff', fontSize: 12, fontWeight: '600' }}>Demo Output</Text>
                                                        </View>
                                                    </View>
                                                )}
                                            </View>
                                        ) : (
                                            <View style={styles.resultPlaceholder}>
                                                <Ionicons name="sparkles-outline" size={44} color={AppColors.textLight} />
                                                <Text style={styles.resultPlaceholderText}>Result will appear here</Text>
                                            </View>
                                        )}
                                    </View>
                                    <View style={styles.wizardNavigation}>
                                        <TouchableOpacity style={styles.secondaryButton} onPress={() => goToStep(2)}>
                                            <Text style={styles.secondaryButtonText}>Back</Text>
                                        </TouchableOpacity>
                                        <TouchableOpacity
                                            style={[styles.primaryButtonFlex, loading && styles.primaryButtonDisabled]}
                                            onPress={() => handleTryOn(humanImage, clothImage, getGarmentType())}
                                            disabled={loading}
                                        >
                                            <Text style={styles.primaryButtonText}>
                                                {loading ? t('aiTryOn.processing') : t('aiTryOn.generate')}
                                            </Text>
                                        </TouchableOpacity>
                                    </View>
                                </>
                            )}

                            {/* Save button — shown only on step 3 */}
                            {tryOnStep === 3 && resultImage && (
                                <TouchableOpacity style={styles.saveButton} onPress={handleSaveToWardrobe} disabled={saving}>
                                    <Ionicons name="heart" size={20} color="#fff" style={{ marginRight: 8 }} />
                                    <Text style={styles.saveButtonText}>{saving ? t('aiTryOn.saving') : t('aiTryOn.saveToWardrobe')}</Text>
                                </TouchableOpacity>
                            )}
                        </>
                    )}
                </ScrollView>
            </SafeAreaView>
        </KeyboardAvoidingView>
    );
};

export default AITryOnScreen;
