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

import type { ShopCatalogItem } from './types';
import { SHOP_CATEGORIES } from '../../data/shopCatalogItems';
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
};

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

    const [selectedItem, setSelectedItem] = useState<ShopCatalogItem | null>(null);
    const [mannequinShopFilter, setMannequinShopFilter] = useState<string>('upper_body');
    const [aiResultImage, setAiResultImage] = useState<string | null>(null);
    const [aiLoading, setAiLoading] = useState(false);
    const [aiError, setAiError] = useState<string | null>(null);
    const [tryOnCount, setTryOnCount] = useState(0);
    const [lookSaved, setLookSaved] = useState(false);
    const [isModelReady, setIsModelReady] = useState(false);
    const {
        items: syncedShopItems,
        loading: shopCatalogLoading,
        loadingMore: shopCatalogLoadingMore,
        error: shopCatalogError,
        hasMore: shopCatalogHasMore,
        loadMore: loadMoreShopCatalog,
        refresh: refreshShopCatalog,
    } = useShopCatalog({ category: mannequinShopFilter });

    const { requireFeature, getRemaining, hasActiveSubscription } = useSubscriptionGate();
    const tryOnsRemaining = getRemaining('tryOns');
    const saveLook = useTryOnLooksStore((s) => s.saveLook);
    const mannequinB64Ref = useRef<string | null>(null);

    useEffect(() => {
        (async () => {
            try {
                const asset = Asset.fromModule(MANNEQUIN_IMAGE);
                await asset.downloadAsync();
                const b64 = await FileSystem.readAsStringAsync(asset.localUri!, { encoding: 'base64' as any });
                mannequinB64Ref.current = `data:image/png;base64,${b64}`;
                setIsModelReady(true);
            } catch (error) {
                console.warn('Failed to preload mannequin image:', error);
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
    const showingFallbackCatalog = syncedShopItems.length === 0;
    const isInitialCatalogLoad = !showingFallbackCatalog && shopCatalogLoading && syncedShopItems.length === 0;
    const shopItems = showingFallbackCatalog ? fallbackShopItems : syncedShopItems;

    const selectedCategoryLabel = selectedItem ? GARMENT_LABELS[selectedItem.garmentType] ?? 'Item' : 'None';
    const remainingCountLabel = tryOnsRemaining === -1 ? '∞' : String(tryOnsRemaining);
    const statusLabel = aiLoading ? 'Live' : aiResultImage ? 'Done' : selectedItem ? 'Ready' : isModelReady ? 'Idle' : 'Prep';

    const summaryStats = useMemo(
        () => [
            { label: 'TRY-ONS', value: remainingCountLabel, warning: tryOnsRemaining !== -1 && tryOnsRemaining <= 1 },
            { label: 'TYPE', value: selectedCategoryLabel },
            { label: 'STATUS', value: statusLabel },
            { label: 'PLAN', value: hasActiveSubscription ? 'Pro' : 'Free' },
        ],
        [hasActiveSubscription, remainingCountLabel, selectedCategoryLabel, statusLabel, tryOnsRemaining]
    );

    const statusCard = useMemo(() => {
        if (aiLoading) {
            return {
                icon: 'sparkles' as const,
                title: 'Generating your preview',
                body: 'We are fitting the selected garment onto the mannequin now. This usually takes 15-30 seconds.',
                tone: 'accent' as const,
            };
        }

        if (aiError) {
            return {
                icon: 'alert-circle' as const,
                title: 'Try-on needs another attempt',
                body: aiError,
                tone: 'error' as const,
            };
        }

        if (aiResultImage) {
            return {
                icon: 'checkmark-circle' as const,
                title: 'Preview is ready',
                body: 'Save this look or swap garments to create another AI try-on.',
                tone: 'success' as const,
            };
        }

        if (!isModelReady) {
            return {
                icon: 'time-outline' as const,
                title: 'Preparing the model',
                body: 'Your mannequin assets are loading so your first preview feels instant.',
                tone: 'neutral' as const,
            };
        }

        if (selectedItem) {
            return {
                icon: 'shirt-outline' as const,
                title: 'Ready to generate',
                body: `${selectedItem.name} is selected. Tap "Try On With AI" to create the preview.`,
                tone: 'accent' as const,
            };
        }

        return {
            icon: 'hand-left-outline' as const,
            title: 'Pick one garment',
            body: 'Choose a top, bottom, or pair of shoes below to preview it on the mannequin.',
            tone: 'neutral' as const,
        };
    }, [aiError, aiLoading, aiResultImage, isModelReady, selectedItem]);

    const getGarmentImageUrl = useCallback(async (item: ShopCatalogItem): Promise<string> => {
        if (typeof item.imageUrl === 'number') {
            const asset = Asset.fromModule(item.imageUrl);
            await asset.downloadAsync();
            const localUri = asset.localUri!;
            const b64 = await FileSystem.readAsStringAsync(localUri, { encoding: 'base64' as any });
            const ext = localUri.split('.').pop()?.toLowerCase() ?? 'png';
            const mime = ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/png';
            return `data:${mime};base64,${b64}`;
        }

        return item.imageUrl as string;
    }, []);

    const handleClear = useCallback(() => {
        setSelectedItem(null);
        setAiResultImage(null);
        setAiError(null);
        setLookSaved(false);
    }, []);

    const handleSelectItem = useCallback((item: ShopCatalogItem) => {
        setAiError(null);
        setLookSaved(false);
        setAiResultImage(null);
        setSelectedItem((prev) => (prev?.id === item.id ? null : item));
    }, []);

    const handleAITryOn = useCallback(async () => {
        if (!selectedItem) {
            setAiError('Select a garment first to generate a try-on.');
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

        try {
            const garmentUrl = await getGarmentImageUrl(selectedItem);
            const mannequinImage = mannequinB64Ref.current;

            if (!mannequinImage) {
                setAiError('Model preview is still loading. Please wait a moment and try again.');
                return;
            }

            const { data, error } = await supabase.functions.invoke('mannequin-tryon', {
                body: {
                    mannequin_image: mannequinImage,
                    garment_image: garmentUrl,
                    garment_type: selectedItem.garmentType || 'upper_body',
                },
            });

            if (!error && data?.success && data?.resultUrl) {
                setAiResultImage(data.resultUrl);
                setLookSaved(false);
                setTryOnCount((prev) => prev + 1);
            } else {
                const message = error?.message || data?.error || 'Try-on failed. Please try again.';
                console.warn('AI try-on failed:', message);
                setAiError(message);
            }
        } catch (error: any) {
            const message = error?.message || 'Unexpected error during try-on.';
            console.warn('AI try-on error:', message);
            setAiError(message);
        } finally {
            setAiLoading(false);
        }
    }, [getGarmentImageUrl, requireFeature, selectedItem, tryOnsRemaining]);

    const handleSaveLook = useCallback(() => {
        if (!aiResultImage || !selectedItem || lookSaved) {
            return;
        }

        saveLook({
            resultUrl: aiResultImage,
            garmentName: selectedItem.name ?? 'Look',
            garmentBrand: selectedItem.brand,
            garmentType: selectedItem.garmentType ?? 'upper_body',
            garmentImageUrl: typeof selectedItem.imageUrl === 'string' ? selectedItem.imageUrl : undefined,
        });

        setLookSaved(true);
        Alert.alert('Saved!', 'Your look has been saved to your Profile.');
    }, [aiResultImage, lookSaved, saveLook, selectedItem]);

    const selectedImageSource = selectedItem
        ? typeof selectedItem.imageUrl === 'string'
            ? { uri: selectedItem.imageUrl }
            : selectedItem.imageUrl
        : null;
    const canReset = Boolean(selectedItem || aiResultImage || aiError);

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

                    <Text style={styles.pageTitle}>{t('aiTryOn.title')}</Text>

                    <View style={styles.heroCard}>
                        <View style={styles.heroHeaderRow}>
                            <View style={styles.heroCopy}>
                                <Text style={styles.heroEyebrow}>VIRTUAL FIT STUDIO</Text>
                                <Text style={styles.heroTitle}>
                                    {aiResultImage ? 'Your preview is ready' : 'Preview on mannequin'}
                                </Text>
                            </View>

                            <View style={styles.planPill}>
                                <Ionicons name="sparkles" size={14} color="#183A67" />
                                <Text style={styles.planPillText}>
                                    {hasActiveSubscription ? 'Pro Plan' : 'Free Plan'}
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
                                    <Text style={styles.savedBadgeText}>Saved to looks</Text>
                                </View>
                            )}

                            {aiLoading && (
                                <View style={styles.previewLoadingOverlay}>
                                    <View style={styles.loadingCard}>
                                        <ActivityIndicator size="large" color="#183A67" />
                                        <Text style={styles.loadingTitle}>Generating preview</Text>
                                        <Text style={styles.loadingText}>
                                            Building your AI try-on with the selected garment. This usually takes 15-30 seconds.
                                        </Text>
                                    </View>
                                </View>
                            )}

                            {!aiLoading && (
                                <View style={styles.previewCaptionWrap}>
                                    <View style={[styles.previewCaptionPill, aiResultImage && styles.previewCaptionPillDark]}>
                                        <Text style={[styles.previewCaptionText, aiResultImage && styles.previewCaptionTextDark]}>
                                            {selectedItem ? selectedItem.name : 'Select a garment below to get started'}
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
                                            <Text style={styles.primaryActionText}>{lookSaved ? 'Saved' : 'Save Look'}</Text>
                                        </LinearGradient>
                                    </TouchableOpacity>

                                    <TouchableOpacity style={styles.secondaryActionButton} onPress={handleClear} activeOpacity={0.88}>
                                        <Ionicons name="refresh-outline" size={18} color="#183A67" />
                                        <Text style={styles.secondaryActionText}>Start Over</Text>
                                    </TouchableOpacity>
                                </>
                            ) : (
                                <>
                                    <TouchableOpacity
                                        style={[
                                            styles.primaryActionButton,
                                            (!selectedItem || aiLoading || !isModelReady) && styles.buttonDisabled,
                                        ]}
                                        onPress={handleAITryOn}
                                        disabled={!selectedItem || aiLoading || !isModelReady}
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
                                                {!isModelReady ? 'Preparing Model' : selectedItem ? 'Try On With AI' : 'Choose a Garment'}
                                            </Text>
                                        </LinearGradient>
                                    </TouchableOpacity>

                                    <TouchableOpacity
                                        style={[styles.secondaryActionButton, !selectedItem && styles.secondaryActionDisabled]}
                                        onPress={handleClear}
                                        disabled={!selectedItem}
                                        activeOpacity={0.88}
                                    >
                                        <Ionicons name="close-outline" size={18} color="#183A67" />
                                        <Text style={styles.secondaryActionText}>Clear</Text>
                                    </TouchableOpacity>
                                </>
                            )}
                        </View>
                    </View>

                    <View style={styles.statsCard}>
                        {summaryStats.map((stat, index) => (
                            <React.Fragment key={stat.label}>
                                <View style={styles.statItem}>
                                    <Text style={[styles.statValue, stat.warning && styles.statValueWarning]}>{stat.value}</Text>
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

                    <View style={styles.catalogCard}>
                        <View style={styles.catalogHeaderRow}>
                            <View>
                                <Text style={styles.catalogEyebrow}>STYLE</Text>
                                <Text style={styles.catalogTitle}>Choose Garment</Text>
                            </View>

                            <View style={styles.catalogCountPill}>
                                <Text style={styles.catalogCountText}>{shopItems.length} items</Text>
                            </View>
                        </View>

                        {selectedItem && selectedImageSource && (
                            <View style={styles.selectedGarmentCard}>
                                <View style={styles.selectedGarmentImageWrap}>
                                    <Image source={selectedImageSource} style={styles.selectedGarmentImage} resizeMode="contain" />
                                </View>

                                <View style={styles.selectedGarmentCopy}>
                                    <Text style={styles.selectedGarmentBrand}>{selectedItem.brand}</Text>
                                    <Text style={styles.selectedGarmentName} numberOfLines={1}>
                                        {selectedItem.name}
                                    </Text>
                                    <Text style={styles.selectedGarmentMeta}>
                                        {selectedCategoryLabel} • {formatPrice(selectedItem)}
                                    </Text>
                                </View>

                                <View style={styles.selectedBadge}>
                                    <Text style={styles.selectedBadgeText}>Selected</Text>
                                </View>
                            </View>
                        )}

                        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterRow}>
                            {SHOP_CATEGORIES.filter((cat) => TRY_ON_CATEGORY_KEYS.has(cat.key)).map((cat) => (
                                <TouchableOpacity
                                    key={cat.key}
                                    style={[styles.filterChip, mannequinShopFilter === cat.key && styles.filterChipActive]}
                                    onPress={() => setMannequinShopFilter(cat.key)}
                                    activeOpacity={0.85}
                                >
                                    <Text
                                        style={[
                                            styles.filterChipText,
                                            mannequinShopFilter === cat.key && styles.filterChipTextActive,
                                        ]}
                                    >
                                        {cat.label}
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </ScrollView>

                        {(shopCatalogError || showingFallbackCatalog) && (
                            <View style={styles.catalogStatusBanner}>
                                <Text style={styles.catalogStatusText}>
                                    {showingFallbackCatalog
                                        ? 'Live Zara menswear is empty right now. Showing backup menswear.'
                                        : 'Live catalog refresh failed. Showing the latest synced menswear.'}
                                </Text>
                                <TouchableOpacity onPress={refreshShopCatalog} activeOpacity={0.85}>
                                    <Text style={styles.catalogStatusAction}>Retry</Text>
                                </TouchableOpacity>
                            </View>
                        )}

                        <View style={styles.catalogGrid}>
                            {isInitialCatalogLoad ? (
                                <View style={styles.loadingCatalogCard}>
                                    <ActivityIndicator size="large" color="#183A67" />
                                    <Text style={styles.loadingCatalogTitle}>Loading menswear catalog</Text>
                                    <Text style={styles.loadingCatalogText}>
                                        Pulling a larger set of live men&apos;s items for try-on previews.
                                    </Text>
                                </View>
                            ) : shopItems.length === 0 ? (
                                <View style={styles.emptyStateCard}>
                                    <Ionicons name="shirt-outline" size={28} color="#7D889A" />
                                    <Text style={styles.emptyStateTitle}>No items in this category yet</Text>
                                    <Text style={styles.emptyStateText}>
                                        Switch filters or add more catalog pieces for try-on previews.
                                    </Text>
                                </View>
                            ) : (
                                shopItems.map((item) => {
                                    const isSelected = selectedItem?.id === item.id;

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
                                                            {GARMENT_LABELS[item.garmentType] ?? 'Item'}
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
                                    <Text style={styles.loadMoreButtonText}>Load more menswear</Text>
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
