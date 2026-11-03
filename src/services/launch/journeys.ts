/**
 * Phase 7 (Sec. 2), 7A — Critical journey inventory + developer-readable
 * path map. Launch acceptance criteria: every journey below must have a
 * known entry, fallback, flag, entitlement, and error state.
 */

export type JourneyArea =
    | 'account_startup'
    | 'wardrobe'
    | 'outfit'
    | 'shopping'
    | 'fit'
    | 'visual'
    | 'subscription'
    | 'data';

export interface CriticalJourney {
    id: string;
    area: JourneyArea;
    name: string;
    /** Screens/entry points. */
    entries: string[];
    /** Services/stores the path depends on. */
    dependsOn: string[];
    /** Safe fallback when the remote/provider leg fails. */
    fallback: string;
    /** Feature flag guarding the journey (Phase 5 system). */
    flag: string | null;
    /** Entitlement required (Phase 6 matrix featureId or 'none'). */
    entitlement: string;
    analyticsEvent: string;
}

export const CRITICAL_JOURNEYS: CriticalJourney[] = [
    { id: 'first_launch', area: 'account_startup', name: 'First launch → onboarding', entries: ['Onboarding'], dependsOn: ['activation', 'subscriptionStore'], fallback: 'offline-capable onboarding with skip paths', flag: null, entitlement: 'none', analyticsEvent: 'onboarding_started' },
    { id: 'returning_launch', area: 'account_startup', name: 'Returning launch → home', entries: ['Home'], dependsOn: ['dailyUsageStore', 'entitlements'], fallback: 'cached entitlement (unknown → free)', flag: null, entitlement: 'none', analyticsEvent: 'app_opened' },
    { id: 'offline_launch', area: 'account_startup', name: 'Offline launch', entries: ['Home', 'Wardrobe'], dependsOn: ['local persistence'], fallback: 'cached wardrobe + saved outfits, unavailable states elsewhere', flag: null, entitlement: 'none', analyticsEvent: 'offline_launch' },
    { id: 'expired_auth', area: 'account_startup', name: 'Expired auth session', entries: ['Auth gate'], dependsOn: ['authEvents', 'supabase auth'], fallback: 'signed-out state, local wardrobe visible', flag: null, entitlement: 'none', analyticsEvent: 'auth_expired' },
    { id: 'account_deletion', area: 'account_startup', name: 'Account deletion path', entries: ['Settings'], dependsOn: ['privacy.deletionMap'], fallback: 'support contact when store constraints require it', flag: null, entitlement: 'none', analyticsEvent: 'account_deletion_requested' },
    { id: 'add_item_manual', area: 'wardrobe', name: 'Add item manually', entries: ['Wardrobe'], dependsOn: ['wardrobeTools'], fallback: 'local draft, sync later', flag: null, entitlement: 'wardrobe_management', analyticsEvent: 'wardrobe_item_added' },
    { id: 'add_item_photo', area: 'wardrobe', name: 'Add item from photo', entries: ['Wardrobe', 'ClosetScan'], dependsOn: ['vision inferenceCache', 'image upload safety'], fallback: 'manual add with saved photo', flag: 'visual_parser', entitlement: 'basic_outfit_generation', analyticsEvent: 'photo_imported' },
    { id: 'edit_item', area: 'wardrobe', name: 'Edit wardrobe item', entries: ['Wardrobe detail'], dependsOn: ['wardrobeTools'], fallback: 'local edit, sync later', flag: null, entitlement: 'wardrobe_management', analyticsEvent: 'wardrobe_item_edited' },
    { id: 'delete_archive_item', area: 'wardrobe', name: 'Delete/archive item', entries: ['Wardrobe detail'], dependsOn: ['wardrobeTools'], fallback: 'archive instead of delete when sync fails', flag: null, entitlement: 'wardrobe_management', analyticsEvent: 'wardrobe_item_removed' },
    { id: 'duplicate_review', area: 'wardrobe', name: 'Duplicate risk review', entries: ['BuyCheck', 'Wardrobe'], dependsOn: ['duplicate risk (Phase 2)'], fallback: 'heuristic warning', flag: null, entitlement: 'basic_buycheck', analyticsEvent: 'duplicate_reviewed' },
    { id: 'wardrobe_offline', area: 'wardrobe', name: 'View wardrobe offline', entries: ['Wardrobe'], dependsOn: ['local persistence'], fallback: 'cached items, no refresh', flag: null, entitlement: 'none', analyticsEvent: 'wardrobe_viewed_offline' },
    { id: 'generate_outfit', area: 'outfit', name: 'Generate outfit (all modes)', entries: ['Stylist', 'Home'], dependsOn: ['outfit ranking (Phase 3/4)', 'entitlements'], fallback: 'manual planning + wardrobe-only', flag: 'new_ranking', entitlement: 'basic_outfit_generation', analyticsEvent: 'outfit_generated' },
    { id: 'anchor_item', area: 'outfit', name: 'Anchor-item outfit', entries: ['Stylist'], dependsOn: ['outfit ranking'], fallback: 'outfits without anchor lock', flag: null, entitlement: 'basic_outfit_generation', analyticsEvent: 'anchor_outfit_generated' },
    { id: 'save_wear_feedback', area: 'outfit', name: 'Save / wear-log / feedback', entries: ['Outfit detail', 'Plans'], dependsOn: ['OutfitMemory', 'growth feedback'], fallback: 'local log, sync later', flag: null, entitlement: 'none', analyticsEvent: 'outfit_worn' },
    { id: 'weekly_plan', area: 'outfit', name: 'Weekly plan', entries: ['Plans'], dependsOn: ['weekly_autonomous_planner'], fallback: 'manual planning', flag: 'daily_stylist', entitlement: 'weekly_autonomous_planner', analyticsEvent: 'weekly_plan_ready' },
    { id: 'browse_search', area: 'shopping', name: 'Browse / search catalog', entries: ['Shop'], dependsOn: ['merchantProducts', 'catalog discovery'], fallback: 'cached catalog, no refresh', flag: 'new_retrieval', entitlement: 'none', analyticsEvent: 'catalog_searched' },
    { id: 'family_variant', area: 'shopping', name: 'Family/variant selection', entries: ['Product detail'], dependsOn: ['family-aware UI (Phase 1)'], fallback: 'selected variant preserved, no silent swap', flag: null, entitlement: 'none', analyticsEvent: 'variant_selected' },
    { id: 'buycheck', area: 'shopping', name: 'BuyCheck verdict', entries: ['Product detail', 'Shop'], dependsOn: ['BuyCheck (Phase 2)', 'price intelligence'], fallback: 'basic verdict without price history', flag: 'buycheck_v2', entitlement: 'basic_buycheck', analyticsEvent: 'buycheck_completed' },
    { id: 'find_better', area: 'shopping', name: 'Find Better / Same Look Less / Max Gain', entries: ['Product detail'], dependsOn: ['catalog discovery', 'price comparison'], fallback: 'unavailable-with-reason, no fake alternatives', flag: null, entitlement: 'deep_buycheck', analyticsEvent: 'find_better_viewed' },
    { id: 'unsupported_merchant', area: 'shopping', name: 'Unsupported merchant URL', entries: ['Share ingest'], dependsOn: ['shareReceiving', 'URL safety'], fallback: 'manual add with clear limitation', flag: null, entitlement: 'none', analyticsEvent: 'unsupported_merchant' },
    { id: 'fit_feedback', area: 'fit', name: 'Fit feedback + reference garment', entries: ['Fit profile'], dependsOn: ['Fit DNA (Phase 2)'], fallback: 'stored feedback, recommendation deferred', flag: 'fit_dna_v2', entitlement: 'none', analyticsEvent: 'fit_feedback_given' },
    { id: 'size_recommendation', area: 'fit', name: 'Size recommendation', entries: ['Product detail'], dependsOn: ['Fit DNA', 'evidence aggregation'], fallback: 'insufficient-evidence state, never a guarantee', flag: null, entitlement: 'none', analyticsEvent: 'size_recommended' },
    { id: 'analyze_garment', area: 'visual', name: 'Analyze garment photo', entries: ['ClosetScan', 'Wardrobe'], dependsOn: ['visual parsing (Phase 4)', 'upload safety'], fallback: 'manual add', flag: 'visual_parser', entitlement: 'advanced_visual_analysis', analyticsEvent: 'garment_analyzed' },
    { id: 'recreate_look', area: 'visual', name: 'Recreate look', entries: ['Stylist'], dependsOn: ['recreate-this-look', 'visual compatibility'], fallback: 'partial match with missing-piece list', flag: 'recreate_look', entitlement: 'recreate_look', analyticsEvent: 'look_recreated' },
    { id: 'visual_fallback', area: 'visual', name: 'Visual provider failure', entries: ['Stylist', 'ClosetScan'], dependsOn: ['circuit breaker (Phase 5)'], fallback: 'saved metadata only, clearly labeled', flag: 'remote_vision_provider', entitlement: 'none', analyticsEvent: 'visual_fallback' },
    { id: 'subscription_states', area: 'subscription', name: 'Free/premium/restore/expired/billing/offline', entries: ['Paywall', 'Settings'], dependsOn: ['iapService', 'subscriptionStore', 'growth lifecycle'], fallback: 'unknown → free features, wardrobe intact', flag: null, entitlement: 'none', analyticsEvent: 'subscription_state_changed' },
    { id: 'privacy_controls', area: 'data', name: 'Export / delete / privacy controls', entries: ['Settings'], dependsOn: ['privacy export + deletion map'], fallback: 'support contact for store-constrained steps', flag: null, entitlement: 'none', analyticsEvent: 'privacy_action' },
    { id: 'notification_prefs', area: 'data', name: 'Notification preferences', entries: ['Settings'], dependsOn: ['growth notification prefs'], fallback: 'all optional, default important-only for promos', flag: null, entitlement: 'none', analyticsEvent: 'notification_pref_changed' },
];

export function journeysForArea(area: 'account_startup' | 'wardrobe' | 'outfit' | 'shopping' | 'fit' | 'visual' | 'subscription' | 'data'): CriticalJourney[] {
    return CRITICAL_JOURNEYS.filter((j) => j.area === area);
}
