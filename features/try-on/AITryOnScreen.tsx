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

import React, { useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Image, ActivityIndicator, Alert, SafeAreaView, Dimensions, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect, useRoute, RouteProp } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { supabase } from '../../lib/supabase';
import useAuthStore from '../../store/auth';
import useSubscriptionStore from '../../store/subscriptionStore';
import AppColors from '../../constants/AppColors';

import { usePhotoPicker } from './hooks/usePhotoPicker';
import { useTryOnWizard } from './hooks/useTryOnWizard';
import { useTryOnAPI } from './hooks/useTryOnAPI';
import type { WardrobeItem } from './types';
import styles from './styles';

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

    const { loading, saving, resultImage, handleTryOn, handleSaveToWardrobe } = useTryOnAPI();

    // Wardrobe items (local state — lightweight)
    const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
    const [loadingWardrobe, setLoadingWardrobe] = useState(false);
    const [selectedWardrobeItem, setSelectedWardrobeItem] = useState<WardrobeItem | null>(null);

    /** Derive garment_type from selected item's category */
    const getGarmentType = (): string => {
        const cat = (selectedWardrobeItem?.category || selectedWardrobeItem?.type || '').toLowerCase();
        if (['pants', 'jeans', 'shorts', 'skirt', 'lower'].some((k) => cat.includes(k))) return 'lower_body';
        if (['dress', 'full'].some((k) => cat.includes(k))) return 'dresses';
        return 'upper_body';
    };

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
                                {mode === 'try your self' ? 'Try your self' : 'Model'}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                {tryOnMode === 'model' ? (
                    /* ── Model Mode ── */
                    <>
                        <View style={styles.stackedCardsWrap}>
                            {['Left', 'Center', 'Right'].map((pos) => (
                                <View key={pos} style={[styles.stackedCard, (styles as any)[`stackedCard${pos}`]]}>
                                    <View style={styles.stackedCardPlaceholder}>
                                        <Ionicons name="person" size={48} color={AppColors.textLight} />
                                    </View>
                                </View>
                            ))}
                        </View>
                        <View style={styles.digitalModelSection}>
                            <View style={styles.digitalModelTitleRow}>
                                <Text style={styles.digitalModelTitle}>{t('aiTryOn.digitalModelTitle')}</Text>
                                {!isPremium && (
                                    <View style={styles.proBadge}>
                                        <Text style={styles.proBadgeText}>{t('aiTryOn.digitalModelPro')}</Text>
                                    </View>
                                )}
                            </View>
                            <Text style={styles.digitalModelDescription}>{t('aiTryOn.digitalModelDescription')}</Text>
                        </View>
                        <TouchableOpacity
                            style={styles.upgradeButton}
                            onPress={() => (navigation as any).navigate(isPremium ? 'CreateAvatar' : 'Paywall')}
                            activeOpacity={0.85}
                        >
                            <Text style={styles.upgradeButtonText}>
                                {isPremium ? 'Create Your Digital Avatar' : t('aiTryOn.upgradeToPro')}
                            </Text>
                        </TouchableOpacity>
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
                                    {(['upload', 'wardrobe'] as const).map((tab) => (
                                        <TouchableOpacity
                                            key={tab}
                                            style={[styles.tab, activeTab === tab && styles.tabActive]}
                                            onPress={() => { setActiveTab(tab); if (tab === 'upload') setSelectedWardrobeItem(null); }}
                                        >
                                            <Ionicons name={tab === 'upload' ? 'cloud-upload-outline' : 'shirt-outline'} size={16} color={activeTab === tab ? '#fff' : AppColors.textMuted} />
                                            <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>{t(tab === 'upload' ? 'aiTryOn.upload' : 'aiTryOn.myWardrobe')}</Text>
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                {activeTab === 'upload' ? (
                                    <View style={styles.garmentCard}>
                                        {clothImage && !selectedWardrobeItem ? (
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
                                ) : (
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

                                <View style={styles.wizardNavigation}>
                                    <TouchableOpacity style={styles.secondaryButton} onPress={() => goToStep(1)}>
                                        <Text style={styles.secondaryButtonText}>Back</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity
                                        style={[styles.primaryButtonFlex, (!clothImage && !selectedWardrobeItem) && styles.primaryButtonDisabled]}
                                        onPress={() => goToStep(3)}
                                        disabled={!clothImage && !selectedWardrobeItem}
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
                                        <Image source={{ uri: resultImage }} style={styles.resultImage} />
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
    );
};

export default AITryOnScreen;
