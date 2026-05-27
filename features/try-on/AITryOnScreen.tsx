import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Image, ActivityIndicator, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { LinearGradient } from 'expo-linear-gradient';
import { supabase } from '../../lib/supabase';
import { useSubscriptionGate } from '../../src/hooks/useSubscriptionGate';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import apiClient from '../../src/services/apiClient';

import type { ShopCatalogItem } from './types';
import { INSPO_MENS_SHOP_ITEMS } from '../../data/inspoMensShopItems';
import { useShopCatalog } from '../../hooks/useShopCatalog';
import styles from './styles';
import useTryOnLooksStore from '../../store/tryOnLooksStore';

const MANNEQUIN_IMAGE = require('../../assets/images/mannequin_front.png');
const TRY_ON_CATEGORY_KEYS = new Set(['upper_body', 'lower_body', 'shoes']);
const GARMENT_LABELS: Record<ShopCatalogItem['garmentType'], string> = {
    upper_body: 'Top',
    lower_body: 'Bottom',
    dresses: 'Dress',
    shoes: 'Shoes',
    outfit: 'Outfit',
    accessory: 'Accessory',
};

type SlotKey = 'layer' | 'top' | 'pants' | 'shoes';

interface SlotDef {
    key: SlotKey;
    label: string;
    category: 'upper_body' | 'lower_body' | 'shoes';
    icon: 'layers-outline' | 'shirt-outline' | 'bag-handle-outline' | 'footsteps-outline';
}

const SLOTS: SlotDef[] = [
    { key: 'layer', label: 'Layer', category: 'upper_body', icon: 'layers-outline' },
    { key: 'top', label: 'Top', category: 'upper_body', icon: 'shirt-outline' },
    { key: 'pants', label: 'Pants', category: 'lower_body', icon: 'bag-handle-outline' },
    { key: 'shoes', label: 'Shoes', category: 'shoes', icon: 'footsteps-outline' },
];

// Correct fashion dressing order for sequential FLUX.1-Kontext calls:
// 1. top    — shirt / t-shirt applied to torso
// 2. layer  — outer jacket / coat applied over the top
// 3. pants  — trousers applied to lower body
// 4. shoes  — footwear placed at the very bottom
const APPLY_ORDER: SlotKey[] = ['top', 'layer', 'pants', 'shoes'];

type Slots = Record<SlotKey, ShopCatalogItem | null>;
const EMPTY_SLOTS: Slots = { layer: null, top: null, pants: null, shoes: null };

type AITryOnRouteParams = { asTab?: boolean; initialGarmentUri?: string; initialGarmentType?: string };

const formatPrice = (item: ShopCatalogItem | null) => {
    if (!item) return '--';
    return `$${item.price % 1 === 0 ? item.price.toFixed(0) : item.price.toFixed(2)}`;
};

const AITryOnScreen = () => {
    const navigation = useNavigation();
    const route = useRoute<RouteProp<{ params: AITryOnRouteParams }, 'params'>>();
    const asTab = (route.params as AITryOnRouteParams)?.asTab === true;
    const { t } = useTranslation();

    const [slots, setSlots] = useState<Slots>(EMPTY_SLOTS);
    const [activeSlot, setActiveSlot] = useState<SlotKey>('top');
    const [aiResultImage, setAiResultImage] = useState<string | null>(null);
    const [aiLoading, setAiLoading] = useState(false);
    const [aiProgress, setAiProgress] = useState<string | null>(null);
    const [aiError, setAiError] = useState<string | null>(null);
    const [lookSaved, setLookSaved] = useState(false);
    const [isModelReady, setIsModelReady] = useState(false);
    const [diagnostics, setDiagnostics] = useState<any>(null);
    const [pipelineVersion, setPipelineVersion] = useState<'sequential_v1' | 'fused_v2' | 'fused_v3'>('fused_v3');

    const activeSlotDef = useMemo(() => SLOTS.find((s) => s.key === activeSlot)!, [activeSlot]);
    const mannequinShopFilter = activeSlotDef.category;
    const {
        items: syncedShopItems,
        loading: shopCatalogLoading,
        loadingMore: shopCatalogLoadingMore,
        error: shopCatalogError,
        hasMore: shopCatalogHasMore,
        loadMore: loadMoreShopCatalog,
        refresh: refreshShopCatalog,
    } = useShopCatalog({ category: mannequinShopFilter });

    const { requireFeature, getRemaining, hasActiveSubscription, consume } = useSubscriptionGate();
    const tryOnsRemaining = getRemaining('tryOns');
    const saveLook = useTryOnLooksStore((s) => s.saveLook);
    const mannequinB64Ref = useRef<string | null>(null);

    useEffect(() => {
        (async () => {
            try {
                const asset = Asset.fromModule(MANNEQUIN_IMAGE);
                await asset.downloadAsync();

                if (!asset.localUri) {
                    throw new Error('Asset localUri is null after download');
                }

                const b64 = await FileSystem.readAsStringAsync(asset.localUri, { encoding: 'base64' as any });
                if (!b64 || b64.length < 100) {
                    throw new Error('Read base64 is empty or too short');
                }

                mannequinB64Ref.current = `data:image/png;base64,${b64}`;
                setIsModelReady(true);
                console.log('[AITryOn] Mannequin preloaded successfully, length:', b64.length);
            } catch (error) {
                console.warn('[AITryOn] Failed to preload mannequin image:', error);
                setAiError('Model preview is still loading. Please wait a moment and try again.');
            }
        })();
    }, []);

    const fallbackShopItems = useMemo(() => {
        if (!TRY_ON_CATEGORY_KEYS.has(mannequinShopFilter)) {
            return INSPO_MENS_SHOP_ITEMS;
        }

        return INSPO_MENS_SHOP_ITEMS.filter((item) => item.garmentType === mannequinShopFilter);
    }, [mannequinShopFilter]);

    const filledSlots = useMemo(
        () => APPLY_ORDER.filter((k) => slots[k] !== null) as SlotKey[],
        [slots]
    );
    const filledCount = filledSlots.length;
    const hasAnySelection = filledCount > 0;
    const showingFallbackCatalog = syncedShopItems.length === 0 && !shopCatalogLoading;
    // Show a loading spinner only on the very first page load (no items yet and spinner is active)
    const isInitialCatalogLoad = shopCatalogLoading && syncedShopItems.length === 0;
    const shopItems = showingFallbackCatalog ? fallbackShopItems : syncedShopItems;

    const selectedCategoryLabel = `${filledCount}/4`;
    const remainingCountLabel = tryOnsRemaining === -1 ? '∞' : String(tryOnsRemaining);
    const statusLabel = aiLoading ? t('tryOn.statusLive') : aiResultImage ? t('tryOn.statusDone') : hasAnySelection ? t('tryOn.statusReady') : isModelReady ? t('tryOn.statusIdle') : t('tryOn.statusPrep');

    const summaryStats = useMemo(
        () => [
            { label: t('tryOn.tryOns'), value: remainingCountLabel, warning: tryOnsRemaining !== -1 && tryOnsRemaining <= 1 },
            { label: t('tryOn.pieces'), value: selectedCategoryLabel },
            { label: t('tryOn.status'), value: statusLabel },
            { label: t('tryOn.plan'), value: hasActiveSubscription ? t('tryOn.pro') : t('tryOn.free') },
            { label: 'AI', value: 'Mobile-VTON 🎯', accent: true },
        ],
        [hasActiveSubscription, remainingCountLabel, selectedCategoryLabel, statusLabel, tryOnsRemaining]
    );

    const statusCard = useMemo(() => {
        if (aiLoading) {
            return {
                icon: 'sparkles' as const,
                title: t('tryOn.generatingPreview'),
                body: aiProgress
                    ? aiProgress
                    : `Dressing mannequin step by step (${filledCount} piece${filledCount === 1 ? '' : 's'}). Est. ${25 * Math.max(filledCount, 1)}–${50 * Math.max(filledCount, 1)}s.`,
                tone: 'accent' as const,
            };
        }

        if (aiError) {
            return {
                icon: 'alert-circle' as const,
                title: t('tryOn.tryOnNeedsAnotherAttempt'),
                body: aiError,
                tone: 'error' as const,
            };
        }

        if (aiResultImage) {
            return {
                icon: 'checkmark-circle' as const,
                title: t('tryOn.previewIsReady'),
                body: t('tryOn.previewReadyBody', { filledCount }),
                tone: 'success' as const,
            };
        }

        if (!isModelReady) {
            return {
                icon: 'time-outline' as const,
                title: t('tryOn.preparingModel'),
                body: t('tryOn.preparingModelBody'),
                tone: 'neutral' as const,
            };
        }

        if (hasAnySelection) {
            const names = filledSlots.map((k) => slots[k]!.name).join(' + ');
            return {
                icon: 'shirt-outline' as const,
                title: t('tryOn.readyToGenerate'),
                body: t('tryOn.piecesSelected', { count: filledCount, names }),
                tone: 'accent' as const,
            };
        }

        return {
            icon: 'hand-left-outline' as const,
            title: t('tryOn.pickYourPieces'),
            body: t('tryOn.chooseUpToFourPieces'),
            tone: 'neutral' as const,
        };
    }, [aiError, aiLoading, aiProgress, aiResultImage, filledCount, filledSlots, hasAnySelection, isModelReady, slots]);

    const getGarmentImageUrl = useCallback(async (item: ShopCatalogItem): Promise<string> => {
        // Local bundled asset (require) → must be converted to base64 to be readable by the edge function.
        if (typeof item.imageUrl === 'number') {
            const asset = Asset.fromModule(item.imageUrl);
            await asset.downloadAsync();
            const localUri = asset.localUri!;
            const b64 = await FileSystem.readAsStringAsync(localUri, { encoding: 'base64' as any });
            const ext = localUri.split('.').pop()?.toLowerCase() ?? 'png';
            const mime = ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/png';
            return `data:${mime};base64,${b64}`;
        }

        // Remote URL (Zara catalog etc.) → pass the URL directly.
        // The edge function fetches it server-side which avoids a large base64 body.
        const url = item.imageUrl as string;
        if (url.startsWith('http://') || url.startsWith('https://')) {
            return url;
        }

        // Already a data-URI or raw base64 → pass through.
        return url;
    }, []);

    const handleClear = useCallback(() => {
        setSlots(EMPTY_SLOTS);
        setAiResultImage(null);
        setAiError(null);
        setLookSaved(false);
        setAiProgress(null);
        setDiagnostics(null);
    }, []);

    const handleClearSlot = useCallback((key: SlotKey) => {
        setSlots((prev) => ({ ...prev, [key]: null }));
        setAiResultImage(null);
        setAiError(null);
        setLookSaved(false);
    }, []);

    const handleSelectItem = useCallback((item: ShopCatalogItem) => {
        setAiError(null);
        setLookSaved(false);
        setAiResultImage(null);
        setSlots((prev) => {
            // Toggle off if same item is in the active slot.
            if (prev[activeSlot]?.id === item.id) {
                return { ...prev, [activeSlot]: null };
            }
            return { ...prev, [activeSlot]: item };
        });
    }, [activeSlot]);

    const buildWearDescription = useCallback((slotKey: SlotKey, item: ShopCatalogItem) => {
        const parts = [item.name?.trim(), item.description?.trim()].filter(Boolean);
        const summary = parts.join(' — ');
        return `${slotKey}: ${summary || item.brand || 'selected garment'}`;
    }, []);

    // Pipeline is synchronous via NVIDIA NIM — no polling required.
    // (Helper kept here as a no-op for any legacy call sites.)
    const pollTask = async (_taskId: string, _slotLabel: string, _step: number, _total: number, _wasComposite: boolean) => {
        throw new Error('Sync pipeline: results are returned in the submit response, no polling required.');
    };

    const handleAITryOn = useCallback(async () => {
        if (filledCount === 0) {
            setAiError('Pick at least one piece (top, layer, pants, or shoes) to generate a try-on.');
            return;
        }

        if (!requireFeature('tryOns')) return;
        if (tryOnsRemaining === 0) {
            setAiError("You've used all your free try-ons. Upgrade for more!");
            return;
        }

        setAiLoading(true);
        setAiResultImage(null);
        setAiError(null);
        setAiProgress(null);

        try {
            const mannequinImage = mannequinB64Ref.current;
            if (!mannequinImage) {
                setAiError('Model preview is still loading. Please wait a moment and try again.');
                return;
            }

            const orderedSlots = APPLY_ORDER.filter((k) => slots[k] !== null) as SlotKey[];
            const visibleTotal = orderedSlots.length;
            setAiProgress(`Dressing mannequin (${visibleTotal} piece${visibleTotal === 1 ? '' : 's'})…`);

            const garments = await Promise.all(
                orderedSlots.map(async (slotKey) => {
                    const item = slots[slotKey]!;
                    const slotDef = SLOTS.find((s) => s.key === slotKey)!;
                    return {
                        label: slotKey,
                        type: slotDef.category,
                        garment_image: await getGarmentImageUrl(item),
                        name: item.name,
                        description: item.description ?? '',
                        wearDescription: buildWearDescription(slotKey, item),
                    };
                }),
            );

            let data: any;
            const runTryOnRequest = async () => {
                const endpoint = '/api/tryon/mobile-vton';
                console.log(`[AITryOn] Using Mobile-VTON endpoint: ${endpoint}`);
                const response = await apiClient.post(
                    endpoint,
                    {
                        mannequin_image: mannequinImage,
                        garments,
                        total: visibleTotal,
                        pipeline_version: pipelineVersion,
                    },
                    { timeout: 180_000 },
                );
                return response.data;
            };

            try {
                data = await runTryOnRequest();
            } catch (err: any) {
                const apiError = err?.response?.data?.error || err?.message;
                throw new Error(apiError || 'Outfit render failed.');
            }

            if (!data?.success) {
                throw new Error(data?.error || 'Outfit render failed.');
            }
            if (!data.resultUrl) throw new Error('No result image returned from renderer.');

            console.log(`[AITryOn] outfit render OK in ${data.elapsedMs}ms (${data.methodUsed || 'gemini-flash'})`);
            if (data.diagnostics) {
                console.log('[AITryOn] diagnostics:', JSON.stringify(data.diagnostics, null, 2));
                setDiagnostics(data.diagnostics);
            }
            setAiProgress(`Preview ready ✓  (${visibleTotal}/${visibleTotal})`);
            setAiResultImage(data.resultUrl as string);
            setLookSaved(false);
            const usage = await consume('tryOns');
            if (!usage.allowed) console.warn('[TryOn] quota consume denied after success');
        } catch (error: any) {
            const message = error?.message || 'Unexpected error during try-on.';
            console.warn('[TryOn] error:', message);
            setAiError(message);
        } finally {
            setAiLoading(false);
            setAiProgress(null);
        }
    }, [buildWearDescription, consume, filledCount, getGarmentImageUrl, requireFeature, slots, tryOnsRemaining]);

    const handleSaveLook = useCallback(() => {
        if (!aiResultImage || !hasAnySelection || lookSaved) {
            return;
        }

        const primary = (slots.top ?? slots.layer ?? slots.pants ?? slots.shoes)!;
        const name = filledSlots.map((k) => slots[k]!.name).join(' + ');

        saveLook({
            resultUrl: aiResultImage,
            garmentName: name || primary.name || 'Look',
            garmentBrand: primary.brand,
            garmentType: primary.garmentType ?? 'upper_body',
            garmentImageUrl: typeof primary.imageUrl === 'string' ? primary.imageUrl : undefined,
        });

        setLookSaved(true);
        Alert.alert(t('tryOn.saved'), t('tryOn.savedMessage'));
    }, [aiResultImage, filledSlots, hasAnySelection, lookSaved, saveLook, slots]);

    const activeSlotItem = slots[activeSlot];
    const canReset = Boolean(hasAnySelection || aiResultImage || aiError);

    return (
        <LinearGradient colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']} style={styles.backgroundGradient}>
            <View style={styles.backgroundOrbTop} />
            <View style={styles.backgroundOrbBottom} />

            <SafeAreaView style={styles.safeArea}>
                <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                    <View style={styles.headerRow}>
                        {asTab ? (
                            <View style={styles.headerPill}>
                                <Ionicons name="sparkles" size={16} color="#183A67" />
                                <Text style={styles.headerPillText}>AI Studio</Text>
                            </View>
                        ) : (
                            <TouchableOpacity
                                onPress={() => navigation.goBack()}
                                hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                                style={styles.headerPill}
                                activeOpacity={0.85}
                            >
                                <Ionicons name="chevron-back" size={18} color="#183A67" />
                                <Text style={styles.headerPillText}>Back</Text>
                            </TouchableOpacity>
                        )}

                        <View style={styles.headerRightActions}>
                            <TouchableOpacity
                                style={styles.headerPill}
                                onPress={() => (navigation as any).navigate('OutfitInspo')}
                                activeOpacity={0.85}
                            >
                                <Ionicons name="eye-outline" size={16} color="#183A67" />
                                <Text style={styles.headerPillText}>Inspo</Text>
                            </TouchableOpacity>

                            {/* Mobile-VTON — CVPR 2026 on-device try-on */}
                            <View style={[styles.headerPill, { backgroundColor: '#EDE7F6' }]}>
                                <Ionicons name="sparkles" size={16} color="#4527A0" />
                                <Text style={[styles.headerPillText, { color: '#4527A0' }]}>Mobile-VTON</Text>
                            </View>

                            <TouchableOpacity
                                style={[styles.headerPill, !canReset && styles.headerPillDisabled]}
                                onPress={handleClear}
                                disabled={!canReset}
                                activeOpacity={0.85}
                            >
                                <Ionicons name="refresh-outline" size={16} color="#183A67" />
                                <Text style={styles.headerPillText}>Reset</Text>
                            </TouchableOpacity>
                        </View>
                    </View>

                    <Text style={styles.pageTitle}>{t('aiTryOn.title')}</Text>

                    <View style={styles.heroCard}>
                        <View style={styles.heroHeaderRow}>
                            <View style={styles.heroCopy}>
                                <Text style={styles.heroEyebrow}>{t('aiTryOn.virtualFitStudio')}</Text>
                                <Text style={styles.heroTitle}>
                                    {aiResultImage ? t('aiTryOn.previewReady') : t('aiTryOn.previewOnMannequin')}
                                </Text>
                            </View>

                            <View style={styles.planPill}>
                                <Ionicons name="sparkles" size={14} color="#183A67" />
                                <Text style={styles.planPillText}>
                                    {hasActiveSubscription ? t('aiTryOn.proPlan') : t('aiTryOn.freePlan')}
                                </Text>
                            </View>
                        </View>

                        <View style={styles.previewShell}>
                            <Image
                                source={aiResultImage ? { uri: aiResultImage } : MANNEQUIN_IMAGE}
                                style={styles.previewImage}
                                resizeMode={aiResultImage ? 'cover' : 'contain'}
                            />

                            {lookSaved && aiResultImage && !aiLoading && (
                                <View style={styles.savedBadge}>
                                    <Ionicons name="checkmark-circle" size={16} color="#FFFFFF" />
                                    <Text style={styles.savedBadgeText}>{t('aiTryOn.savedToLooks')}</Text>
                                </View>
                            )}

                            {aiLoading && (
                                <View style={styles.previewLoadingOverlay}>
                                    <View style={styles.loadingCard}>
                                        <ActivityIndicator size="large" color="#183A67" />
                                        <Text style={styles.loadingTitle}>{t('aiTryOn.generatingPreview')}</Text>
                                        <Text style={styles.loadingText}>
                                            {t('aiTryOn.generatingPreviewBody')}
                                        </Text>
                                    </View>
                                </View>
                            )}

                            {!aiLoading && (
                                <View style={styles.previewCaptionWrap}>
                                    <View style={[styles.previewCaptionPill, aiResultImage && styles.previewCaptionPillDark]}>
                                        <Text style={[styles.previewCaptionText, aiResultImage && styles.previewCaptionTextDark]}>
                                            {hasAnySelection
                                                ? t('aiTryOn.piecesSelected', { count: filledCount })
                                                : t('aiTryOn.pickPiecesBelow')}
                                        </Text>
                                    </View>
                                </View>
                            )}
                        </View>

                        <View style={styles.heroActionsRow}>
                            {aiResultImage ? (
                                <>
                                    <TouchableOpacity
                                        style={[styles.primaryActionButton, lookSaved && styles.buttonDisabled]}
                                        onPress={handleSaveLook}
                                        disabled={lookSaved}
                                        activeOpacity={0.88}
                                    >
                                        <LinearGradient
                                            colors={['#244F85', '#112A4A']}
                                            start={{ x: 0, y: 0.5 }}
                                            end={{ x: 1, y: 0.5 }}
                                            style={styles.primaryActionGradient}
                                        >
                                            <Ionicons
                                                name={lookSaved ? 'checkmark-circle' : 'bookmark-outline'}
                                                size={18}
                                                color="#FFFFFF"
                                            />
                                            <Text style={styles.primaryActionText}>{lookSaved ? t('aiTryOn.saved') : t('aiTryOn.saveLook')}</Text>
                                        </LinearGradient>
                                    </TouchableOpacity>

                                    <TouchableOpacity style={styles.secondaryActionButton} onPress={handleClear} activeOpacity={0.88}>
                                        <Ionicons name="refresh-outline" size={18} color="#183A67" />
                                        <Text style={styles.secondaryActionText}>{t('aiTryOn.startOver')}</Text>
                                    </TouchableOpacity>
                                </>
                            ) : (
                                <>
                                    <TouchableOpacity
                                        style={[
                                            styles.primaryActionButton,
                                            (!hasAnySelection || aiLoading || !isModelReady) && styles.buttonDisabled,
                                        ]}
                                        onPress={handleAITryOn}
                                        disabled={!hasAnySelection || aiLoading || !isModelReady}
                                        activeOpacity={0.88}
                                    >
                                        <LinearGradient
                                            colors={['#244F85', '#112A4A']}
                                            start={{ x: 0, y: 0.5 }}
                                            end={{ x: 1, y: 0.5 }}
                                            style={styles.primaryActionGradient}
                                        >
                                            <Ionicons name="sparkles" size={18} color="#FFFFFF" />
                                            <Text style={styles.primaryActionText}>
                                                {!isModelReady
                                                    ? t('aiTryOn.preparingModel')
                                                    : hasAnySelection
                                                        ? `${t('aiTryOn.tryOnWithAI')} (${filledCount})`
                                                        : t('aiTryOn.chooseGarments')}
                                            </Text>
                                        </LinearGradient>
                                    </TouchableOpacity>

                                    <TouchableOpacity
                                        style={[styles.secondaryActionButton, !hasAnySelection && styles.secondaryActionDisabled]}
                                        onPress={handleClear}
                                        disabled={!hasAnySelection}
                                        activeOpacity={0.88}
                                    >
                                        <Ionicons name="close-outline" size={18} color="#183A67" />
                                        <Text style={styles.secondaryActionText}>{t('aiTryOn.clear')}</Text>
                                    </TouchableOpacity>
                                </>
                            )}
                        </View>
                    </View>

                    <View style={styles.statsCard}>
                        {summaryStats.map((stat, index) => (
                            <React.Fragment key={stat.label}>
                                <View style={styles.statItem}>
                                    <Text style={[styles.statValue, stat.warning && styles.statValueWarning, stat.accent && { color: '#2E7D32' }]}>{stat.value}</Text>
                                    <Text style={styles.statLabel}>{stat.label}</Text>
                                </View>
                                {index < summaryStats.length - 1 && <View style={styles.statDivider} />}
                            </React.Fragment>
                        ))}
                    </View>

                    <View
                        style={[
                            styles.statusCard,
                            statusCard.tone === 'accent' && styles.statusCardAccent,
                            statusCard.tone === 'success' && styles.statusCardSuccess,
                            statusCard.tone === 'error' && styles.statusCardError,
                        ]}
                    >
                        <View
                            style={[
                                styles.statusIconWrap,
                                statusCard.tone === 'accent' && styles.statusIconWrapAccent,
                                statusCard.tone === 'success' && styles.statusIconWrapSuccess,
                                statusCard.tone === 'error' && styles.statusIconWrapError,
                            ]}
                        >
                            <Ionicons
                                name={statusCard.icon}
                                size={18}
                                color={
                                    statusCard.tone === 'success'
                                        ? '#157347'
                                        : statusCard.tone === 'error'
                                            ? '#C14444'
                                            : '#183A67'
                                }
                            />
                        </View>

                        <View style={styles.statusCopy}>
                            <Text style={styles.statusTitle}>{statusCard.title}</Text>
                            <Text style={styles.statusText}>{statusCard.body}</Text>
                        </View>
                    </View>

                    {__DEV__ && (
                        <View style={{
                            flexDirection: 'row',
                            alignItems: 'center',
                            gap: 8,
                            marginBottom: 8,
                        }}>
                            <TouchableOpacity
                                onPress={() => setPipelineVersion(v => {
                                    if (v === 'fused_v3') return 'fused_v2';
                                    if (v === 'fused_v2') return 'sequential_v1';
                                    return 'fused_v3';
                                })}
                                style={{
                                    backgroundColor: pipelineVersion === 'fused_v3' ? '#E8F5E9' : (pipelineVersion === 'fused_v2' ? '#E3F2FD' : '#FFF3E0'),
                                    borderRadius: 10,
                                    paddingHorizontal: 12,
                                    paddingVertical: 6,
                                    borderWidth: 1,
                                    borderColor: pipelineVersion === 'fused_v3' ? '#4CAF50' : (pipelineVersion === 'fused_v2' ? '#2196F3' : '#FF9800'),
                                }}
                            >
                                <Text style={{ fontSize: 11, fontWeight: '700', color: pipelineVersion === 'fused_v3' ? '#2E7D32' : (pipelineVersion === 'fused_v2' ? '#1565C0' : '#E65100') }}>
                                    {pipelineVersion === 'fused_v3' ? 'Fused v3 (Single Pass)' : (pipelineVersion === 'fused_v2' ? 'Fused v2' : 'Sequential v1')}
                                </Text>
                            </TouchableOpacity>
                            <Text style={{ fontSize: 10, color: '#7D889A' }}>
                                Tap to toggle pipeline
                            </Text>
                        </View>
                    )}

                    {__DEV__ && diagnostics && (
                        <View style={{
                            backgroundColor: '#F8FAFF',
                            borderRadius: 14,
                            padding: 12,
                            marginTop: 4,
                            borderWidth: 1,
                            borderColor: 'rgba(17,46,82,0.08)',
                        }}>
                            <Text style={{ fontSize: 11, fontWeight: '700', color: '#5F6D84', marginBottom: 6 }}>
                                Pipeline Diagnostics
                            </Text>
                            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}>
                                <View style={{ backgroundColor: '#FFFFFF', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }}>
                                    <Text style={{ fontSize: 10, color: '#7D889A' }}>Version</Text>
                                    <Text style={{ fontSize: 12, fontWeight: '700', color: '#183A67' }}>
                                        {diagnostics.pipelineVersion || 'sequential_v1'}
                                    </Text>
                                </View>
                                <View style={{ backgroundColor: '#FFFFFF', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }}>
                                    <Text style={{ fontSize: 10, color: '#7D889A' }}>Total</Text>
                                    <Text style={{ fontSize: 12, fontWeight: '700', color: '#183A67' }}>
                                        {(diagnostics.totalElapsedMs || 0).toFixed(0)}ms
                                    </Text>
                                </View>
                                <View style={{ backgroundColor: '#FFFFFF', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }}>
                                    <Text style={{ fontSize: 10, color: '#7D889A' }}>VRAM</Text>
                                    <Text style={{ fontSize: 12, fontWeight: '700', color: '#183A67' }}>
                                        {(diagnostics.peakVramMb || 0).toFixed(0)}MB
                                    </Text>
                                </View>
                                {Object.entries(diagnostics.cacheHits || {}).length > 0 && (
                                    <View style={{ backgroundColor: '#E8F5E9', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }}>
                                        <Text style={{ fontSize: 10, color: '#2E7D32' }}>Cache Hits</Text>
                                        <Text style={{ fontSize: 12, fontWeight: '700', color: '#1B5E20' }}>
                                            {(Object.values(diagnostics.cacheHits || {}) as number[]).reduce((a, b) => a + b, 0)}
                                        </Text>
                                    </View>
                                )}
                                {diagnostics.degraded && (
                                    <View style={{ backgroundColor: '#FFEBEE', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }}>
                                        <Text style={{ fontSize: 10, color: '#C14444' }}>Degraded</Text>
                                        <Text style={{ fontSize: 12, fontWeight: '700', color: '#B71C1C' }}>
                                            {diagnostics.degradedReason || 'Yes'}
                                        </Text>
                                    </View>
                                )}
                            </View>
                            {diagnostics.renderedGarments && (
                                <Text style={{ fontSize: 10, color: '#7D889A', marginTop: 6 }}>
                                    Order: {(diagnostics.renderedGarments || []).join(' → ')}
                                </Text>
                            )}
                        </View>
                    )}

                    <View style={styles.catalogCard}>
                        <View style={styles.catalogHeaderRow}>
                            <View>
                                <Text style={styles.catalogEyebrow}>{t('aiTryOn.style')}</Text>
                                <Text style={styles.catalogTitle}>{t('aiTryOn.chooseLabel', { label: activeSlotDef.label })}</Text>
                            </View>

                            <View style={styles.catalogCountPill}>
                                <Text style={styles.catalogCountText}>{t('aiTryOn.itemsCount', { count: shopItems.length })}</Text>
                            </View>
                        </View>

                        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterRow}>
                            {SLOTS.map((slot) => {
                                const isActive = activeSlot === slot.key;
                                const filled = slots[slot.key];
                                return (
                                    <TouchableOpacity
                                        key={slot.key}
                                        style={[styles.filterChip, isActive && styles.filterChipActive]}
                                        onPress={() => setActiveSlot(slot.key)}
                                        activeOpacity={0.85}
                                    >
                                        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                                            <Ionicons
                                                name={slot.icon}
                                                size={14}
                                                color={isActive ? '#FFFFFF' : '#5F6D84'}
                                            />
                                            <Text
                                                style={[
                                                    styles.filterChipText,
                                                    isActive && styles.filterChipTextActive,
                                                ]}
                                            >
                                                {slot.label}
                                            </Text>
                                            {filled && (
                                                <View
                                                    style={{
                                                        width: 8,
                                                        height: 8,
                                                        borderRadius: 4,
                                                        backgroundColor: isActive ? '#FFFFFF' : '#157347',
                                                    }}
                                                />
                                            )}
                                        </View>
                                    </TouchableOpacity>
                                );
                            })}
                        </ScrollView>

                        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
                            {SLOTS.map((slot) => {
                                const item = slots[slot.key];
                                const isActive = activeSlot === slot.key;
                                const imgSrc = item
                                    ? typeof item.imageUrl === 'string'
                                        ? { uri: item.imageUrl }
                                        : item.imageUrl
                                    : null;
                                return (
                                    <TouchableOpacity
                                        key={slot.key}
                                        onPress={() => setActiveSlot(slot.key)}
                                        activeOpacity={0.85}
                                        style={{
                                            flexBasis: '48%',
                                            flexGrow: 1,
                                            flexDirection: 'row',
                                            alignItems: 'center',
                                            gap: 10,
                                            padding: 10,
                                            borderRadius: 14,
                                            backgroundColor: '#FFFFFF',
                                            borderWidth: 1.5,
                                            borderColor: isActive ? '#173A65' : 'rgba(17,46,82,0.08)',
                                        }}
                                    >
                                        <View
                                            style={{
                                                width: 44,
                                                height: 44,
                                                borderRadius: 10,
                                                backgroundColor: '#F2F5FB',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                overflow: 'hidden',
                                            }}
                                        >
                                            {imgSrc ? (
                                                <Image source={imgSrc} style={{ width: '100%', height: '100%' }} resizeMode="contain" />
                                            ) : (
                                                <Ionicons name={slot.icon} size={20} color="#7D889A" />
                                            )}
                                        </View>
                                        <View style={{ flex: 1 }}>
                                            <Text style={{ fontSize: 11, fontWeight: '700', color: '#7D889A', letterSpacing: 0.4 }}>
                                                {slot.label.toUpperCase()}
                                            </Text>
                                            <Text
                                                style={{ fontSize: 13, fontWeight: '600', color: '#183A67' }}
                                                numberOfLines={1}
                                            >
                                                {item ? item.name : 'Tap to choose'}
                                            </Text>
                                        </View>
                                        {item && (
                                            <TouchableOpacity
                                                onPress={() => handleClearSlot(slot.key)}
                                                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                                            >
                                                <Ionicons name="close-circle" size={18} color="#C14444" />
                                            </TouchableOpacity>
                                        )}
                                    </TouchableOpacity>
                                );
                            })}
                        </View>

                        {(shopCatalogError || showingFallbackCatalog) && (
                            <View style={styles.catalogStatusBanner}>
                                <Text style={styles.catalogStatusText}>
                                    {showingFallbackCatalog
                                        ? t('aiTryOn.liveCatalogEmpty')
                                        : t('aiTryOn.liveCatalogFailed')}
                                </Text>
                                <TouchableOpacity onPress={refreshShopCatalog} activeOpacity={0.85}>
                                    <Text style={styles.catalogStatusAction}>{t('aiTryOn.retry')}</Text>
                                </TouchableOpacity>
                            </View>
                        )}

                        <View style={styles.catalogGrid}>
                            {isInitialCatalogLoad ? (
                                <View style={styles.loadingCatalogCard}>
                                    <ActivityIndicator size="large" color="#183A67" />
                                    <Text style={styles.loadingCatalogTitle}>{t('aiTryOn.loadingCatalog')}</Text>
                                    <Text style={styles.loadingCatalogText}>
                                        {t('aiTryOn.loadingCatalogBody')}
                                    </Text>
                                </View>
                            ) : shopItems.length === 0 ? (
                                <View style={styles.emptyStateCard}>
                                    <Ionicons name="shirt-outline" size={28} color="#7D889A" />
                                    <Text style={styles.emptyStateTitle}>{t('aiTryOn.noItemsCategory')}</Text>
                                    <Text style={styles.emptyStateText}>
                                        {t('aiTryOn.noItemsCategoryBody')}
                                    </Text>
                                </View>
                            ) : (
                                shopItems.map((item) => {
                                    const isSelected = activeSlotItem?.id === item.id;

                                    return (
                                        <TouchableOpacity
                                            key={item.id}
                                            style={[styles.itemCard, isSelected && styles.itemCardSelected]}
                                            onPress={() => handleSelectItem(item)}
                                            activeOpacity={0.9}
                                            disabled={aiLoading}
                                        >
                                            <View style={styles.itemImageWrap}>
                                                <Image
                                                    source={typeof item.imageUrl === 'string' ? { uri: item.imageUrl } : item.imageUrl}
                                                    style={styles.itemImage}
                                                    resizeMode="contain"
                                                />

                                                {isSelected && (
                                                    <View style={styles.itemSelectedBadge}>
                                                        <Ionicons name="checkmark-circle" size={22} color="#183A67" />
                                                    </View>
                                                )}
                                            </View>

                                            <View style={styles.itemInfo}>
                                                <Text style={styles.itemBrand}>{item.brand}</Text>
                                                <Text style={styles.itemName} numberOfLines={2}>
                                                    {item.name}
                                                </Text>

                                                <View style={styles.itemFooter}>
                                                    <Text style={styles.itemPrice}>{formatPrice(item)}</Text>
                                                    <View style={styles.itemTypePill}>
                                                        <Text style={styles.itemTypeText}>
                                                            {GARMENT_LABELS[item.garmentType] ?? t('aiTryOn.item')}
                                                        </Text>
                                                    </View>
                                                </View>
                                            </View>
                                        </TouchableOpacity>
                                    );
                                })
                            )}
                        </View>

                        {!showingFallbackCatalog && shopCatalogHasMore && !isInitialCatalogLoad && (
                            <TouchableOpacity
                                style={styles.loadMoreButton}
                                onPress={loadMoreShopCatalog}
                                disabled={shopCatalogLoadingMore}
                                activeOpacity={0.88}
                            >
                                {shopCatalogLoadingMore ? (
                                    <ActivityIndicator size="small" color="#183A67" />
                                ) : (
                                    <Text style={styles.loadMoreButtonText}>{t('aiTryOn.loadMoreMenswear')}</Text>
                                )}
                            </TouchableOpacity>
                        )}
                    </View>
                </ScrollView>
            </SafeAreaView>
        </LinearGradient>
    );
};

export default AITryOnScreen;
