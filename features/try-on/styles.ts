/**
 * Try-On feature styles — Liquid Glass theme
 */

import { StyleSheet } from 'react-native';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: 12,
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        fontWeight: '700',
        color: colors.text.primary,
        letterSpacing: 0.3,
    },
    headerSpacer: { width: 28 },
    scrollContent: {
        padding: spacing.screenPadding,
        paddingBottom: 40,
    },

    // Segmented Control (consistent with all tabs)
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 15,
        paddingBottom: 20,
    },
    modeToggleWrap: {
        flexDirection: 'row',
        backgroundColor: colors.background.tertiary,
        borderRadius: 24,
        padding: 4,
        width: 300,
    },
    modeToggleOption: {
        flex: 1,
        paddingVertical: 10,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 20,
    },
    modeToggleOptionActive: {
        backgroundColor: colors.background.primary,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 4,
        elevation: 2,
    },
    modeToggleText: {
        fontSize: 15,
        fontWeight: '500',
        color: colors.text.tertiary,
    },
    modeToggleTextActive: {
        color: colors.text.primary,
        fontWeight: '600',
    },

    // Stacked Cards (Model Mode)
    stackedCardsWrap: {
        height: 260,
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
    },
    stackedCard: {
        position: 'absolute',
        width: 110,
        height: 220,
        borderRadius: radius.xl,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        borderWidth: 1,
        borderColor: colors.border.subtle,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
        elevation: 6,
    },
    stackedCardLeft: { transform: [{ translateX: -100 }, { scale: 0.88 }], zIndex: 1 },
    stackedCardCenter: { transform: [{ translateX: 0 }, { scale: 1 }], zIndex: 3, width: 130, height: 240 },
    stackedCardRight: { transform: [{ translateX: 100 }, { scale: 0.88 }], zIndex: 1 },
    stackedCardPlaceholder: {
        flex: 1,
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Digital Model Section
    digitalModelSection: {
        marginBottom: 24,
        paddingHorizontal: 4,
        alignItems: 'center',
    },
    digitalModelTitleRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
        flexWrap: 'wrap',
    },
    digitalModelTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
        marginRight: 8,
        textAlign: 'center',
    },
    proBadge: {
        backgroundColor: '#A855F7',
        paddingHorizontal: 8,
        paddingVertical: 3,
        borderRadius: 6,
    },
    proBadgeText: {
        fontSize: 11,
        fontWeight: '800',
        color: '#fff',
        letterSpacing: 0.5,
    },
    digitalModelDescription: {
        fontSize: 15,
        color: colors.text.secondary,
        lineHeight: 22,
        textAlign: 'center',
    },
    upgradeButton: {
        backgroundColor: colors.text.primary,
        paddingVertical: 14,
        borderRadius: radius.button,
        width: '100%',
        alignItems: 'center',
        marginTop: 35,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 4,
    },
    upgradeButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },

    // Steps
    stepLabel: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '700',
        marginBottom: 4,
        marginLeft: 2,
    },
    stepHint: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        marginBottom: 16,
        marginLeft: 2,
    },

    // Photo Cards
    fullLengthCard: {
        width: '100%',
        height: 380,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.xl,
        overflow: 'hidden',
        marginBottom: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 20,
        elevation: 8,
    },
    fullLengthImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
        borderRadius: radius.xl,
    },
    fullLengthPlaceholder: {
        flex: 1,
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 16,
        paddingTop: 12,
    },

    placeholderIconWrap: {
        marginBottom: 20,
        alignItems: 'center',
        justifyContent: 'center',
    },
    placeholderIconCircle: {
        width: 104,
        height: 104,
        borderRadius: 52,
        backgroundColor: colors.background.tertiary,
        alignItems: 'center',
        justifyContent: 'center',
    },

    placeholderTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: 8,
        letterSpacing: -0.3,
    },
    placeholderSub: {
        fontSize: 15,
        color: colors.text.tertiary,
        textAlign: 'center',
        marginBottom: 36,
        paddingHorizontal: 16,
    },

    photoOptionsRow: {
        flexDirection: 'row',
        gap: 12,
        justifyContent: 'center',
        width: '100%',
    },
    photoOption: {
        flex: 1,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        paddingVertical: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    photoOptionIconWrap: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: colors.background.primary,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.04,
        shadowRadius: 6,
    },
    photoOptionText: {
        fontSize: 15,
        fontWeight: '600',
        color: colors.text.primary,
    },

    // Tabs (Upload / Wardrobe)
    tabContainer: {
        flexDirection: 'row',
        marginBottom: 16,
        borderRadius: radius.lg,
        backgroundColor: colors.background.tertiary,
        padding: 4,
    },
    tab: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        borderRadius: radius.md,
        gap: 8,
    },
    tabActive: {
        backgroundColor: colors.text.primary,
    },
    tabText: {
        fontSize: 15,
        fontWeight: '600',
        color: colors.text.tertiary,
    },
    tabTextActive: {
        color: '#FFFFFF',
    },

    // Garment Card
    garmentCard: {
        width: '100%',
        height: 380,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.xl,
        overflow: 'hidden',
        marginBottom: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 20,
        elevation: 8,
    },
    garmentImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
        borderRadius: radius.xl,
    },

    // Wardrobe
    wardrobeSection: {
        minHeight: 180,
        marginBottom: 24,
    },
    wardrobeLoading: {
        paddingVertical: 40,
        alignItems: 'center',
    },
    wardrobeLoadingText: {
        marginTop: 10,
        fontSize: 14,
        color: colors.text.tertiary,
    },
    wardrobeEmpty: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 32,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.xl,
        borderWidth: 1,
        borderColor: colors.border.subtle,
        borderStyle: 'dashed',
    },
    wardrobeEmptyText: {
        marginTop: 10,
        fontSize: 14,
        color: colors.text.tertiary,
    },
    scanButton: {
        marginTop: 16,
        backgroundColor: colors.text.primary,
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: radius.button,
    },
    scanButtonText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
    wardrobeScroll: {
        paddingVertical: 8,
    },
    wardrobeItemCard: {
        width: 84,
        height: 116,
        borderRadius: radius.md,
        backgroundColor: colors.background.secondary,
        marginRight: 12,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: colors.border.subtle,
    },
    wardrobeItemCardSelected: {
        borderColor: '#34C759',
        borderWidth: 2,
    },
    wardrobeItemImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
    },
    wardrobeItemPlaceholder: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.background.secondary,
    },
    selectedBadge: {
        position: 'absolute',
        top: 6,
        right: 6,
        backgroundColor: '#fff',
        borderRadius: 12,
    },
    selectedInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 12,
        paddingHorizontal: 14,
        paddingVertical: 10,
        backgroundColor: '#E8F5E9',
        borderRadius: radius.md,
    },
    selectedInfoText: {
        marginLeft: 8,
        fontSize: 13,
        color: '#2E7D32',
        fontWeight: '600',
    },

    // Result
    resultContainer: {
        width: '100%',
        height: 400,
        borderRadius: radius.xl,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 5,
    },
    resultImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
    },
    loadingBox: {
        alignItems: 'center',
        padding: 24,
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
        fontWeight: '600',
        color: colors.text.primary,
    },
    loadingSub: {
        marginTop: 6,
        fontSize: 13,
        color: colors.text.tertiary,
    },
    resultPlaceholder: {
        alignItems: 'center',
    },
    resultPlaceholderText: {
        marginTop: 12,
        fontSize: 14,
        color: colors.text.tertiary,
    },

    // Buttons
    wizardNavigation: {
        flexDirection: 'row',
        gap: 12,
        marginTop: 12,
        width: '100%',
    },
    secondaryButton: {
        backgroundColor: colors.background.secondary,
        paddingVertical: 14,
        paddingHorizontal: 28,
        borderRadius: radius.button,
        alignItems: 'center',
        justifyContent: 'center',
    },
    secondaryButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    primaryButtonFlex: {
        flex: 1,
        backgroundColor: colors.text.primary,
        paddingVertical: 14,
        borderRadius: radius.button,
        alignItems: 'center',
        justifyContent: 'center',
    },
    primaryButton: {
        backgroundColor: colors.text.primary,
        paddingVertical: 14,
        borderRadius: radius.button,
        width: '100%',
        alignItems: 'center',
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 4,
    },
    primaryButtonDisabled: {
        opacity: 0.5,
    },
    primaryButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    saveButton: {
        backgroundColor: colors.text.primary,
        paddingVertical: 14,
        borderRadius: radius.button,
        width: '100%',
        alignItems: 'center',
        flexDirection: 'row',
        justifyContent: 'center',
        marginBottom: 24,
    },
    saveButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },

    // ── Mannequin Card ──────────────────────────────────────────────────────
    mannequinCard: {
        width: '100%',
        height: 420,
        borderRadius: radius.xl,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        marginBottom: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.08,
        shadowRadius: 20,
        elevation: 8,
    },
    mannequinImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
    },
    mannequinLoadingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(255,255,255,0.82)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    mannequinResultOverlay: {
        ...StyleSheet.absoluteFillObject,
    },
    mannequinResultImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
    },
    mannequinPlaceholderOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(255,255,255,0.70)',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 24,
    },
    mannequinPlaceholderText: {
        fontSize: 15,
        color: colors.text.secondary,
        textAlign: 'center',
        fontWeight: '500',
        lineHeight: 22,
    },

    // ── Gender Toggle ───────────────────────────────────────────────────────
    genderToggleRow: {
        flexDirection: 'row',
        backgroundColor: colors.background.tertiary,
        borderRadius: radius.chip,
        padding: 4,
        marginBottom: 14,
        alignSelf: 'center',
        width: 220,
    },
    genderToggleOption: {
        flex: 1,
        paddingVertical: 9,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 20,
    },
    genderToggleActive: {
        backgroundColor: colors.background.primary,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.07,
        shadowRadius: 6,
        elevation: 2,
    },
    genderToggleText: {
        fontSize: 14,
        fontWeight: '500',
        color: colors.text.tertiary,
    },
    genderToggleTextActive: {
        color: colors.text.primary,
        fontWeight: '700',
    },

    // ── Size Selector ───────────────────────────────────────────────────────
    sizeRow: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 16,
        flexWrap: 'wrap',
    },
    sizeButton: {
        paddingHorizontal: 18,
        paddingVertical: 8,
        borderRadius: radius.chip,
        backgroundColor: colors.background.secondary,
        borderWidth: 1.5,
        borderColor: colors.border.subtle,
    },
    sizeButtonActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    sizeButtonText: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    sizeButtonTextActive: {
        color: '#FFFFFF',
    },

    // ── Measurements Row ────────────────────────────────────────────────────
    measurementsRow: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 24,
        flexWrap: 'wrap',
    },
    measurementChip: {
        flex: 1,
        minWidth: 68,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.md,
        paddingVertical: 10,
        paddingHorizontal: 10,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: colors.border.subtle,
    },
    measurementChipLabel: {
        fontSize: 10,
        fontWeight: '600',
        color: colors.text.tertiary,
        textTransform: 'uppercase',
        letterSpacing: 0.6,
        marginBottom: 3,
    },
    measurementChipText: {
        fontSize: 13,
        fontWeight: '700',
        color: colors.text.primary,
    },

    // ── Shop Section Header ─────────────────────────────────────────────────
    shopSectionLabel: {
        fontSize: 17,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: 12,
        letterSpacing: -0.2,
    },

    // ── Shop Filter Chips ───────────────────────────────────────────────────
    shopFilterRow: {
        paddingBottom: 4,
        gap: 8,
        flexDirection: 'row',
    },
    shopFilterChip: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: radius.chip,
        backgroundColor: colors.background.secondary,
        borderWidth: 1.5,
        borderColor: colors.border.subtle,
    },
    shopFilterChipActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    shopFilterChipText: {
        fontSize: 13,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    shopFilterChipTextActive: {
        color: '#FFFFFF',
    },

    // ── Shop Catalog Grid ───────────────────────────────────────────────────
    shopCatalogGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
        marginTop: 12,
        marginBottom: 20,
    },
    shopItemCard: {
        width: '47%',
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.05,
        shadowRadius: 8,
        elevation: 3,
    },
    shopItemCardSelected: {
        borderColor: '#0055FF',
    },
    shopItemImage: {
        width: '100%',
        height: 160,
        resizeMode: 'cover',
    },
    shopItemInfo: {
        padding: 10,
    },
    shopItemBrand: {
        fontSize: 10,
        fontWeight: '700',
        color: colors.text.tertiary,
        textTransform: 'uppercase',
        letterSpacing: 0.6,
        marginBottom: 2,
    },
    shopItemName: {
        fontSize: 13,
        fontWeight: '600',
        color: colors.text.primary,
        marginBottom: 4,
    },
    shopItemPrice: {
        fontSize: 13,
        fontWeight: '700',
        color: colors.text.primary,
    },
    shopItemSelectedBadge: {
        position: 'absolute',
        top: 8,
        right: 8,
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 2,
    },

    // ── Mannequin View Toggle (Front / Side) ───────────────────────────────
    mannequinViewToggle: {
        position: 'absolute',
        bottom: 12,
        left: 0,
        right: 0,
        flexDirection: 'row',
        justifyContent: 'center',
    },
    mannequinViewButton: {
        paddingHorizontal: 20,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.80)',
        marginHorizontal: 3,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.06)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.08,
        shadowRadius: 4,
        elevation: 2,
    },
    mannequinViewButtonActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    mannequinViewButtonText: {
        fontSize: 13,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    mannequinViewButtonTextActive: {
        color: '#FFFFFF',
    },

    // ── Mannequin Generate Button ───────────────────────────────────────────
    mannequinGenerateButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.text.primary,
        paddingVertical: 16,
        borderRadius: radius.button,
        width: '100%',
        marginBottom: 12,
        gap: 8,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.12,
        shadowRadius: 12,
        elevation: 5,
    },
    mannequinGenerateButtonDisabled: {
        opacity: 0.45,
    },
    mannequinGenerateButtonText: {
        color: '#FFFFFF',
        fontSize: 16,
        fontWeight: '700',
        letterSpacing: 0.2,
    },

    // ── Mannequin Save Button ───────────────────────────────────────────────
    mannequinSaveButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#0055FF',
        paddingVertical: 14,
        borderRadius: radius.button,
        width: '100%',
        marginBottom: 24,
        gap: 8,
    },
    mannequinSaveButtonText: {
        color: '#FFFFFF',
        fontSize: 15,
        fontWeight: '600',
    },
});

export default styles;
