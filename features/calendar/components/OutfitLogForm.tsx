/**
 * OutfitLogForm — Slot-based Outfit Logger
 * Clear flow: tap a slot (Top/Bottoms/Shoes) → pick item from grid below
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    Modal,
    TouchableOpacity,
    Image,
    ScrollView,
    StyleSheet,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';
import { type WardrobeItem, type OccasionId, type ClothingCategory, matchesCategory } from '../types';
import type { Product } from '../../../src/services/shoppingService';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SLOT_WIDTH = (SCREEN_WIDTH - 48 - 24) / 3;
const GRID_ITEM = (SCREEN_WIDTH - 48 - 16) / 3;

interface Slot {
    readonly id: ClothingCategory;
    readonly label: string;
    readonly icon: string;
    readonly hint: string;
}

interface OutfitLogFormProps {
    visible: boolean;
    wardrobeItems: WardrobeItem[];
    shopItems?: Product[];
    selectedItems: WardrobeItem[];
    selectedOccasion: OccasionId;
    onClose: () => void;
    onToggleItem: (item: WardrobeItem) => void;
    onSelectOccasion: (id: OccasionId) => void;
    onSave: () => void;
}

// Combined item type for display
interface DisplayItem extends WardrobeItem {
    isShopItem?: boolean;
    price?: number;
    brand?: string;
    localImage?: string;
}

export const OutfitLogForm: React.FC<OutfitLogFormProps> = ({
    visible,
    wardrobeItems,
    shopItems = [],
    selectedItems,
    selectedOccasion: _selectedOccasion,
    onClose,
    onToggleItem,
    onSelectOccasion: _onSelectOccasion,
    onSave,
}) => {
    const { t } = useTranslation();
    
    const SLOTS: readonly Slot[] = [
        { id: 'top',   label: t('outfitLogForm.top'),     icon: '👕', hint: t('outfitLogForm.topHint') },
        { id: 'pants', label: t('outfitLogForm.bottoms'), icon: '👖', hint: t('outfitLogForm.bottomsHint') },
        { id: 'shoes', label: t('outfitLogForm.shoes'),   icon: '👟', hint: t('outfitLogForm.shoesHint') },
    ] as const;
    // Helper: Convert Product to WardrobeItem format
    const productToWardrobeItem = useCallback((product: Product): DisplayItem => ({
        id: `shop-${product.id}`,
        type: product.category,
        image: product.imageUrl,
        imageUrl: product.imageUrl,
        color: product.color,
        name: product.name,
        category: product.category,
        isShopItem: true,
        price: product.price,
        brand: product.brand,
    }), []);

    // Convert shop items to wardrobe format
    const displayShopItems = shopItems.map(productToWardrobeItem);
    const [activeSlot, setActiveSlot] = useState<ClothingCategory>('top');

    const currentSlot = SLOTS.find(s => s.id === activeSlot)!;

    // Items visible in the grid — filtered by active slot category (wardrobe + shop)
    const wardrobeGridItems: DisplayItem[] = wardrobeItems.filter(item => {
        const typeStr = (item.type || item.category || '').toLowerCase();
        return matchesCategory(typeStr, currentSlot.id);
    });

    const shopGridItems: DisplayItem[] = displayShopItems.filter(item =>
        matchesCategory(item.type || item.category || '', currentSlot.id)
    );

    // Fallback: if no items match the slot, show all wardrobe items so the user always sees something
    const wardrobeDisplay: DisplayItem[] = wardrobeGridItems.length > 0
        ? wardrobeGridItems
        : wardrobeItems.map(item => ({ ...item, isShopItem: false }));

    const shopDisplay: DisplayItem[] = shopGridItems;

    // Which item is currently assigned to each slot
    const slotItem = useCallback((slotId: ClothingCategory): WardrobeItem | null => {
        return selectedItems.find(item => matchesCategory(item.type, slotId)) ?? null;
    }, [selectedItems]);

    const handleSlotTap = (slotId: ClothingCategory) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setActiveSlot(slotId);
    };

    const handleItemTap = (item: WardrobeItem) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        const existing = slotItem(activeSlot);
        if (existing && existing.id !== item.id) {
            onToggleItem(existing);
        }
        onToggleItem(item);
        // Auto-advance to next unfilled slot
        const slots = SLOTS;
        const currentIdx = slots.findIndex(s => s.id === activeSlot);
        const nextSlot = slots.slice(currentIdx + 1).find(s => !slotItem(s.id));
        if (nextSlot) {
            setTimeout(() => setActiveSlot(nextSlot.id), 160);
        }
    };

    const filledCount = SLOTS.filter(s => slotItem(s.id)).length;

    // Resolve remote or local asset images for slot previews.
    const getImageSource = (item: WardrobeItem) => {
        const localImage = (item as DisplayItem).localImage;
        if (typeof localImage === 'string' && localImage.length > 0) {
            return { uri: localImage };
        }

        const imageUri = item.image || item.imageUrl || '';
        return imageUri ? { uri: imageUri } : null;
    };

    return (
        <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
            <View style={styles.overlay}>
                <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={onClose} />

                <View style={styles.sheet}>
                    {/* Handle */}
                    <View style={styles.handle} />

                    {/* Header */}
                    <View style={styles.header}>
                        <View style={{ flex: 1 }}>
                            <Text style={styles.title}>What did you wear?</Text>
                            <Text style={styles.subtitle}>
                                Tap a slot below, then pick from your wardrobe
                            </Text>
                        </View>
                        <TouchableOpacity style={styles.closeBtn} onPress={onClose}>
                            <Ionicons name="close" size={18} color="#475569" />
                        </TouchableOpacity>
                    </View>

                    {/* ── OUTFIT SLOTS ────────────────────────── */}
                    <View style={styles.slotsRow}>
                        {SLOTS.map(slot => {
                            const picked = slotItem(slot.id);
                            const isActive = activeSlot === slot.id;
                            return (
                                <TouchableOpacity
                                    key={slot.id}
                                    style={[
                                        styles.slotCard,
                                        isActive && styles.slotCardActive,
                                        picked && !isActive && styles.slotCardFilled,
                                    ]}
                                    onPress={() => handleSlotTap(slot.id)}
                                    activeOpacity={0.8}
                                >
                                    {picked ? (
                                        getImageSource(picked) ? (
                                            <Image
                                                source={getImageSource(picked)!}
                                                style={styles.slotImg}
                                                resizeMode="cover"
                                            />
                                        ) : (
                                            <View style={styles.slotEmpty}>
                                                <Text style={styles.slotEmoji}>{slot.icon}</Text>
                                            </View>
                                        )
                                    ) : (
                                        <View style={styles.slotEmpty}>
                                            <Text style={styles.slotEmoji}>{slot.icon}</Text>
                                        </View>
                                    )}
                                    <Text style={[styles.slotLabel, isActive && styles.slotLabelActive]}>
                                        {slot.label}
                                    </Text>
                                    {/* Active indicator dot */}
                                    {isActive && <View style={styles.activeDot} />}
                                    {/* Filled checkmark */}
                                    {picked && (
                                        <View style={styles.filledCheck}>
                                            <Ionicons name="checkmark" size={10} color="#FFF" />
                                        </View>
                                    )}
                                </TouchableOpacity>
                            );
                        })}
                    </View>

                    {/* Progress indicator */}
                    <View style={styles.progressRow}>
                        <Text style={styles.progressText}>
                            {filledCount === 0
                                ? `Tap a slot to start — now picking ${currentSlot.label}`
                                : filledCount === 3
                                ? '✓ All slots filled — ready to save!'
                                : `${filledCount}/3 filled — now picking ${currentSlot.label}`}
                        </Text>
                    </View>

                    {/* ── WARDROBE GRID ───────────────────────── */}
                    <View style={styles.gridHeader}>
                        <Text style={styles.gridTitle}>
                            {currentSlot.icon}  Choose {currentSlot.label}
                        </Text>
                        <Text style={styles.gridHint}>{currentSlot.hint}</Text>
                    </View>

                    <ScrollView
                        style={styles.gridScroll}
                        showsVerticalScrollIndicator={false}
                        contentContainerStyle={styles.gridContent}
                    >
                        {wardrobeDisplay.length === 0 && shopDisplay.length === 0 ? (
                            <View style={styles.emptyGrid}>
                                <Text style={styles.emptyEmoji}>{currentSlot.icon}</Text>
                                <Text style={styles.emptyTitle}>No items yet</Text>
                                <Text style={styles.emptyHint}>Add clothes to your wardrobe to get started</Text>
                            </View>
                        ) : (
                            <>
                                {/* ── MY WARDROBE SECTION ── */}
                                {wardrobeDisplay.length > 0 && (
                                    <>
                                        <View style={styles.sectionHeader}>
                                            <Ionicons name="shirt-outline" size={14} color="#6366F1" />
                                            <Text style={styles.sectionTitle}>My Wardrobe</Text>
                                            <Text style={styles.sectionCount}>{wardrobeDisplay.length}</Text>
                                        </View>
                                        <View style={styles.grid}>
                                            {wardrobeDisplay.map((item, idx) => {
                                                const isPicked = slotItem(activeSlot)?.id === item.id;
                                                const imageSource = getImageSource(item);
                                                return (
                                                    <TouchableOpacity
                                                        key={item.id || idx}
                                                        style={[styles.gridItem, isPicked && styles.gridItemPicked]}
                                                        onPress={() => handleItemTap(item)}
                                                        activeOpacity={0.82}
                                                    >
                                                        {imageSource ? (
                                                            <Image source={imageSource} style={styles.gridImg} resizeMode="cover" />
                                                        ) : (
                                                            <View style={[styles.gridImg, styles.gridImgPlaceholder]}>
                                                                <Ionicons name="shirt-outline" size={32} color="#94A3B8" />
                                                            </View>
                                                        )}
                                                        {isPicked && <View style={styles.pickedOverlay} />}
                                                        {isPicked && (
                                                            <View style={styles.pickedCheck}>
                                                                <Ionicons name="checkmark" size={16} color="#FFF" />
                                                            </View>
                                                        )}
                                                    </TouchableOpacity>
                                                );
                                            })}
                                        </View>
                                    </>
                                )}

                                {/* ── FROM SHOP SECTION ── */}
                                {shopGridItems.length > 0 && (
                                    <>
                                        <View style={[styles.sectionHeader, styles.sectionHeaderShop]}>
                                            <Ionicons name="bag-outline" size={14} color="#F59E0B" />
                                            <Text style={[styles.sectionTitle, styles.sectionTitleShop]}>From Shop</Text>
                                            <Text style={[styles.sectionCount, styles.sectionCountShop]}>{shopGridItems.length}</Text>
                                        </View>
                                        <View style={styles.grid}>
                                            {shopGridItems.map((item, idx) => {
                                                const isPicked = slotItem(activeSlot)?.id === item.id;
                                                const imageSource = getImageSource(item);
                                                return (
                                                    <TouchableOpacity
                                                        key={item.id || idx}
                                                        style={[
                                                            styles.gridItem,
                                                            styles.gridItemShop,
                                                            isPicked && styles.gridItemPicked,
                                                        ]}
                                                        onPress={() => handleItemTap(item)}
                                                        activeOpacity={0.82}
                                                    >
                                                        {imageSource ? (
                                                            <Image source={imageSource} style={styles.gridImg} resizeMode="cover" />
                                                        ) : (
                                                            <View style={[styles.gridImg, styles.gridImgPlaceholder]}>
                                                                <Ionicons name="bag-outline" size={32} color="#F59E0B" />
                                                            </View>
                                                        )}
                                                        <View style={styles.shopBadge}>
                                                            <Ionicons name="bag" size={10} color="#FFF" />
                                                        </View>
                                                        {item.price != null && (
                                                            <View style={styles.priceBadge}>
                                                                <Text style={styles.priceText}>${item.price}</Text>
                                                            </View>
                                                        )}
                                                        {isPicked && <View style={styles.pickedOverlay} />}
                                                        {isPicked && (
                                                            <View style={styles.pickedCheck}>
                                                                <Ionicons name="checkmark" size={16} color="#FFF" />
                                                            </View>
                                                        )}
                                                    </TouchableOpacity>
                                                );
                                            })}
                                        </View>
                                    </>
                                )}
                            </>
                        )}
                    </ScrollView>

                    {/* ── OCCASION + SAVE ─────────────────────── */}
                    <View style={styles.bottomSection}>
                        {/* Save */}
                        <TouchableOpacity
                            style={[styles.saveBtn, filledCount === 0 && styles.saveBtnDisabled]}
                            onPress={() => {
                                if (filledCount > 0) {
                                    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                                    onSave();
                                }
                            }}
                            disabled={filledCount === 0}
                            activeOpacity={0.85}
                        >
                            <Ionicons
                                name={filledCount > 0 ? 'checkmark-circle' : 'shirt-outline'}
                                size={22}
                                color={filledCount > 0 ? '#FFF' : '#94A3B8'}
                            />
                            <Text style={[styles.saveBtnText, filledCount === 0 && styles.saveBtnTextOff]}>
                                {filledCount === 0
                                    ? 'Fill at least one slot'
                                    : `Save Outfit  ·  ${filledCount} item${filledCount > 1 ? 's' : ''}`}
                            </Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: { flex: 1, justifyContent: 'flex-end' },
    backdrop: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(15,23,42,0.6)',
    },
    sheet: {
        backgroundColor: '#F2F4FA',
        borderTopLeftRadius: 38,
        borderTopRightRadius: 38,
        paddingTop: 14,
        paddingBottom: 32,
        maxHeight: '94%',
        shadowColor: '#0F172A',
        shadowOffset: { width: 0, height: -10 },
        shadowOpacity: 0.2,
        shadowRadius: 36,
        elevation: 24,
    },

    // Handle
    handle: {
        width: 44, height: 5,
        borderRadius: 3,
        backgroundColor: '#CBD5E1',
        alignSelf: 'center',
        marginBottom: 20,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        paddingHorizontal: 24,
        marginBottom: 22,
        gap: 12,
    },
    title: {
        fontSize: 24,
        fontWeight: '800',
        color: '#0F172A',
        letterSpacing: -0.4,
    },
    subtitle: {
        fontSize: 13,
        color: '#94A3B8',
        fontWeight: '500',
        marginTop: 3,
        lineHeight: 18,
    },
    closeBtn: {
        width: 36, height: 36, borderRadius: 18,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderWidth: 1.5, borderColor: '#E2E8F0',
        justifyContent: 'center', alignItems: 'center',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1, shadowRadius: 6,
        elevation: 2,
    },

    // Outfit slots
    slotsRow: {
        flexDirection: 'row',
        paddingHorizontal: 24,
        gap: 8,
        marginBottom: 12,
    },
    slotCard: {
        width: SLOT_WIDTH,
        backgroundColor: 'rgba(255,255,255,0.8)',
        borderRadius: 22,
        alignItems: 'center',
        paddingVertical: 12,
        paddingHorizontal: 6,
        borderWidth: 2,
        borderColor: 'rgba(255,255,255,0.9)',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.1, shadowRadius: 10,
        elevation: 2,
        position: 'relative',
    },
    slotCardActive: {
        borderColor: '#6366F1',
        backgroundColor: 'rgba(99,102,241,0.07)',
        shadowColor: '#6366F1',
        shadowOpacity: 0.18,
    },
    slotCardFilled: {
        borderColor: '#10B981',
        backgroundColor: 'rgba(16,185,129,0.05)',
    },
    slotEmpty: {
        width: SLOT_WIDTH - 20,
        height: SLOT_WIDTH - 20,
        borderRadius: 16,
        backgroundColor: '#EEF0F8',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 8,
    },
    slotEmoji: { fontSize: 28 },
    slotImg: {
        width: SLOT_WIDTH - 20,
        height: SLOT_WIDTH - 20,
        borderRadius: 16,
        backgroundColor: '#EEF0F8',
        marginBottom: 8,
    },
    slotLabel: {
        fontSize: 11,
        fontWeight: '700',
        color: '#64748B',
        letterSpacing: 0.3,
    },
    slotLabelActive: { color: '#6366F1' },
    activeDot: {
        position: 'absolute',
        bottom: 6,
        width: 6, height: 6,
        borderRadius: 3,
        backgroundColor: '#6366F1',
    },
    filledCheck: {
        position: 'absolute',
        top: 6, right: 6,
        width: 20, height: 20,
        borderRadius: 10,
        backgroundColor: '#10B981',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#F2F4FA',
    },

    // Progress
    progressRow: {
        paddingHorizontal: 24,
        marginBottom: 16,
    },
    progressText: {
        fontSize: 12,
        color: '#6366F1',
        fontWeight: '600',
    },

    // Grid header
    gridHeader: {
        paddingHorizontal: 24,
        marginBottom: 12,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    gridTitle: {
        fontSize: 15,
        fontWeight: '800',
        color: '#0F172A',
    },
    gridHint: {
        fontSize: 12,
        color: '#94A3B8',
        fontWeight: '500',
    },

    // Section headers
    sectionHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginBottom: 10,
        marginTop: 4,
    },
    sectionHeaderShop: {
        marginTop: 18,
    },
    sectionTitle: {
        fontSize: 13,
        fontWeight: '700',
        color: '#6366F1',
        flex: 1,
    },
    sectionTitleShop: {
        color: '#F59E0B',
    },
    sectionCount: {
        fontSize: 12,
        fontWeight: '600',
        color: '#6366F1',
        backgroundColor: 'rgba(99,102,241,0.1)',
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 10,
    },
    sectionCountShop: {
        color: '#F59E0B',
        backgroundColor: 'rgba(245,158,11,0.1)',
    },

    // Items grid
    gridScroll: { maxHeight: 300 },
    gridContent: { paddingHorizontal: 24, paddingVertical: 8 },
    grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
    gridItem: {
        width: GRID_ITEM,
        height: GRID_ITEM,
        borderRadius: 18,
        overflow: 'hidden',
        backgroundColor: '#E4E8F2',
        borderWidth: 2.5,
        borderColor: 'transparent',
    },
    gridItemPicked: {
        borderColor: '#6366F1',
    },
    gridItemShop: {
        borderColor: '#F59E0B',
        borderWidth: 2,
    },
    shopBadge: {
        position: 'absolute',
        top: 6,
        left: 6,
        width: 20,
        height: 20,
        borderRadius: 10,
        backgroundColor: '#F59E0B',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#FFF',
    },
    priceBadge: {
        position: 'absolute',
        bottom: 6,
        left: 4,
        right: 4,
        backgroundColor: 'rgba(0,0,0,0.55)',
        borderRadius: 8,
        paddingVertical: 2,
        alignItems: 'center',
    },
    priceText: {
        fontSize: 11,
        fontWeight: '700',
        color: '#FFF',
    },
    gridImg: { width: '100%', height: '100%' },
    gridImgPlaceholder: {
        backgroundColor: '#E2E8F0',
        justifyContent: 'center',
        alignItems: 'center',
    },
    pickedOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(99,102,241,0.15)',
    },
    pickedCheck: {
        position: 'absolute',
        top: 6, right: 6,
        width: 28, height: 28,
        borderRadius: 14,
        backgroundColor: '#6366F1',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2.5,
        borderColor: '#FFF',
    },
    emptyGrid: {
        alignItems: 'center',
        paddingVertical: 32,
    },
    emptyEmoji: { fontSize: 40, marginBottom: 10 },
    emptyTitle: { fontSize: 15, fontWeight: '700', color: '#475569', marginBottom: 4 },
    emptyHint: { fontSize: 12, color: '#94A3B8' },

    // Bottom section
    bottomSection: {
        paddingTop: 14,
        borderTopWidth: 1,
        borderTopColor: 'rgba(226,232,240,0.8)',
        marginTop: 6,
    },

    // Save button
    saveBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
        marginHorizontal: 24,
        backgroundColor: '#0F172A',
        paddingVertical: 18,
        borderRadius: 30,
        shadowColor: '#0F172A',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 18,
        elevation: 8,
    },
    saveBtnDisabled: {
        backgroundColor: 'rgba(255,255,255,0.75)',
        borderWidth: 1.5,
        borderColor: '#E2E8F0',
        shadowOpacity: 0,
        elevation: 0,
    },
    saveBtnText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: '800',
        letterSpacing: 0.1,
    },
    saveBtnTextOff: { color: '#94A3B8' },
});

export default OutfitLogForm;
