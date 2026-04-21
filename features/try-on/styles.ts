/**
 * Try-On feature styles
 */

import { StyleSheet } from 'react-native';
import LiquidGlass2026Theme from '../../constants/LiquidGlass2026Theme';
import AppColors from '../../constants/AppColors';

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingTop: 12, // Moved Try On text lower
        paddingBottom: 14,
        backgroundColor: '#FFFFFF',
    },
    headerTitle: {
        ...LiquidGlass2026Theme.typography.scale.titleLarge,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: 0.3,
    },
    headerSpacer: { width: 28 },
    scrollContent: { padding: 20, paddingBottom: 40 },

    segmentContainer: {
        alignItems: 'center',
        paddingTop: 15,
        paddingBottom: 20,
        backgroundColor: '#FFFFFF',
    },
    modeToggleWrap: {
        flexDirection: 'row',
        backgroundColor: '#E5E5EA', // iOS System Gray 5
        borderRadius: 24, // Rounded
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
        backgroundColor: '#FFFFFF',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    modeToggleText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#8E8E93',
    },
    modeToggleTextActive: {
        color: '#0A1931',
        fontWeight: '600',
    },

    // Stacked Cards (Model Mode)
    stackedCardsWrap: { height: 260, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 24 },
    stackedCard: {
        position: 'absolute', width: 110, height: 220, borderRadius: 20, overflow: 'hidden',
        backgroundColor: AppColors.surface, borderWidth: 1, borderColor: AppColors.border,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.1, shadowRadius: 12, elevation: 6,
    },
    stackedCardLeft: { transform: [{ translateX: -100 }, { scale: 0.88 }], zIndex: 1 },
    stackedCardCenter: { transform: [{ translateX: 0 }, { scale: 1 }], zIndex: 3, width: 130, height: 240 },
    stackedCardRight: { transform: [{ translateX: 100 }, { scale: 0.88 }], zIndex: 1 },
    stackedCardPlaceholder: { flex: 1, backgroundColor: AppColors.surfaceSecondary, alignItems: 'center', justifyContent: 'center' },

    // Digital Model Section
    digitalModelSection: { marginBottom: 24, paddingHorizontal: 4, alignItems: 'center' },
    digitalModelTitleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: 8, flexWrap: 'wrap' },
    digitalModelTitle: { fontSize: 20, fontWeight: '700', color: AppColors.text, marginRight: 8, textAlign: 'center' },
    proBadge: { backgroundColor: '#A855F7', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
    proBadgeText: { fontSize: 11, fontWeight: '800', color: '#fff', letterSpacing: 0.5 },
    digitalModelDescription: { fontSize: 15, color: AppColors.textSecondary, lineHeight: 22, textAlign: 'center' },
    upgradeButton: {
        backgroundColor: '#0A1931', paddingVertical: 14, borderRadius: 30, width: '100%', alignItems: 'center', marginTop: 35,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 10, elevation: 4,
    },
    upgradeButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },

    // Hero
    hero: { alignItems: 'center', marginBottom: 28 },
    heroIconWrap: { width: 64, height: 64, borderRadius: 32, backgroundColor: '#0A1931', alignItems: 'center', justifyContent: 'center', marginBottom: 16 },
    heroTitle: { ...LiquidGlass2026Theme.typography.scale.headlineMedium, fontWeight: '800', color: '#0A1931', textAlign: 'center', marginBottom: 10, letterSpacing: -0.5 },
    heroSubtitle: { ...LiquidGlass2026Theme.typography.scale.bodyLarge, color: '#4B5563', textAlign: 'center', lineHeight: 22, maxWidth: 320 },

    // Steps
    stepLabel: { ...LiquidGlass2026Theme.typography.scale.titleMedium, color: '#0A1931', fontWeight: '700', marginBottom: 4, marginLeft: 2 },
    stepHint: { ...LiquidGlass2026Theme.typography.scale.bodyMedium, color: '#6B7280', marginBottom: 16, marginLeft: 2 },

    // Photo Cards
    fullLengthCard: {
        width: '100%', height: 380, backgroundColor: '#FFFFFF', borderRadius: 44, borderWidth: 1, borderColor: 'rgba(0,0,0,0.03)',
        overflow: 'visible', marginBottom: 24,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 24 }, shadowOpacity: 0.08, shadowRadius: 36, elevation: 12
    },
    fullLengthImage: { width: '100%', height: '100%', resizeMode: 'cover', borderRadius: 44 },
    fullLengthPlaceholder: { flex: 1, height: '100%', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16, paddingTop: 12 },

    placeholderIconWrap: { marginBottom: 20, alignItems: 'center', justifyContent: 'center' },
    placeholderIconCircle: {
        width: 104, height: 104, borderRadius: 52, backgroundColor: '#F0F4FF', alignItems: 'center', justifyContent: 'center',
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.12, shadowRadius: 24
    },

    placeholderTitle: { fontSize: 24, fontWeight: '800', color: '#0A1931', marginBottom: 8, letterSpacing: -0.6 },
    placeholderSub: { fontSize: 15, color: '#8A94A6', textAlign: 'center', marginBottom: 36, letterSpacing: 0, paddingHorizontal: 16 },

    photoOptionsRow: { flexDirection: 'row', gap: 12, justifyContent: 'center', width: '100%' },
    photoOption: {
        flex: 1, backgroundColor: '#F7F9FC', borderRadius: 28, paddingVertical: 18, alignItems: 'center', justifyContent: 'center',
        borderWidth: 1, borderColor: 'rgba(0,0,0,0.02)'
    },
    photoOptionIconWrap: {
        width: 44, height: 44, borderRadius: 22, backgroundColor: '#FFFFFF', alignItems: 'center', justifyContent: 'center', marginBottom: 12,
        shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.06, shadowRadius: 10
    },
    photoOptionText: { fontSize: 15, fontWeight: '700', color: '#0A1931' },

    // Tabs
    tabContainer: { flexDirection: 'row', marginBottom: 16, borderRadius: 16, backgroundColor: '#F3F4F6', padding: 6 },
    tab: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 12, borderRadius: 12, gap: 8 },
    tabActive: { backgroundColor: '#0A1931' },
    tabText: { fontSize: 15, fontWeight: '600', color: '#6B7280' },
    tabTextActive: { color: '#FFFFFF' },

    // Garment Card
    garmentCard: {
        width: '100%', height: 380, backgroundColor: '#FFFFFF', borderRadius: 44, borderWidth: 1, borderColor: 'rgba(0,0,0,0.03)',
        overflow: 'visible', marginBottom: 24,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 24 }, shadowOpacity: 0.08, shadowRadius: 36, elevation: 12
    },
    garmentImage: { width: '100%', height: '100%', resizeMode: 'cover', borderRadius: 44 },

    // Wardrobe
    wardrobeSection: { minHeight: 180, marginBottom: 24 },
    wardrobeLoading: { paddingVertical: 40, alignItems: 'center' },
    wardrobeLoadingText: { marginTop: 10, fontSize: 14, color: AppColors.textMuted },
    wardrobeEmpty: { alignItems: 'center', justifyContent: 'center', paddingVertical: 32, backgroundColor: AppColors.surface, borderRadius: 20, borderWidth: 1, borderColor: AppColors.border, borderStyle: 'dashed' },
    wardrobeEmptyText: { marginTop: 10, fontSize: 14, color: AppColors.textMuted },
    scanButton: { marginTop: 16, backgroundColor: AppColors.primary, paddingHorizontal: 20, paddingVertical: 12, borderRadius: 24 },
    scanButtonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
    wardrobeScroll: { paddingVertical: 8 },
    wardrobeItemCard: { width: 84, height: 116, borderRadius: 14, backgroundColor: AppColors.surface, marginRight: 12, overflow: 'hidden', borderWidth: 2, borderColor: AppColors.border },
    wardrobeItemCardSelected: { borderColor: '#34C759', borderWidth: 2 },
    wardrobeItemImage: { width: '100%', height: '100%', resizeMode: 'cover' },
    wardrobeItemPlaceholder: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: AppColors.surfaceSecondary },
    selectedBadge: { position: 'absolute', top: 6, right: 6, backgroundColor: '#fff', borderRadius: 12 },
    selectedInfo: { flexDirection: 'row', alignItems: 'center', marginTop: 12, paddingHorizontal: 14, paddingVertical: 10, backgroundColor: '#E8F5E9', borderRadius: 10 },
    selectedInfoText: { marginLeft: 8, fontSize: 13, color: '#2E7D32', fontWeight: '600' },

    // Result
    resultContainer: {
        width: '100%', height: 400, borderRadius: 24, overflow: 'hidden', backgroundColor: AppColors.surface,
        alignItems: 'center', justifyContent: 'center', marginBottom: 20, borderWidth: 1, borderColor: AppColors.border,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.08, shadowRadius: 16, elevation: 5,
    },
    resultImage: { width: '100%', height: '100%', resizeMode: 'cover' },
    loadingBox: { alignItems: 'center', padding: 24 },
    loadingText: { marginTop: 16, fontSize: 16, fontWeight: '600', color: AppColors.text },
    loadingSub: { marginTop: 6, fontSize: 13, color: AppColors.textMuted },
    resultPlaceholder: { alignItems: 'center' },
    resultPlaceholderText: { marginTop: 12, fontSize: 14, color: AppColors.textMuted },

    // Buttons
    wizardNavigation: { flexDirection: 'row', gap: 12, marginTop: 12, width: '100%' },
    secondaryButton: { backgroundColor: '#F8FAFC', paddingVertical: 14, paddingHorizontal: 28, borderRadius: 30, alignItems: 'center', justifyContent: 'center' },
    secondaryButtonText: { fontSize: 16, fontWeight: '600', color: '#475569' },
    primaryButtonFlex: { flex: 1, backgroundColor: '#0A1931', paddingVertical: 14, borderRadius: 30, alignItems: 'center', justifyContent: 'center' },
    primaryButton: {
        backgroundColor: '#0A1931', paddingVertical: 14, borderRadius: 30, width: '100%', alignItems: 'center', marginBottom: 12,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 10, elevation: 4,
    },
    primaryButtonDisabled: { opacity: 0.7 },
    primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
    saveButton: { backgroundColor: '#0A1931', paddingVertical: 14, borderRadius: 30, width: '100%', alignItems: 'center', flexDirection: 'row', justifyContent: 'center', marginBottom: 24 },
    saveButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});

export default styles;
