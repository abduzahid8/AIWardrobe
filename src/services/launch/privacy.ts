/**
 * Phase 7Q–7T — Deletion map, export architecture, privacy controls,
 * data classification. No remote execution; interfaces + local models.
 */

import type { DataClass, DiagnosticPackage } from '../../types/launch';

export const DELETION_SCOPES = [
    'wardrobe_items',
    'images',
    'outfits',
    'fit_evidence',
    'style_evidence',
    'purchase_history',
    'price_watches',
    'recommendation_feedback',
    'analytics_identifiers',
    'subscription_references',
] as const;

export type DeletionScope = (typeof DELETION_SCOPES)[number];

export interface DeletionPlan {
    scope: DeletionScope;
    action: 'delete_rows' | 'delete_storage_objects' | 'anonymize' | 'support_flow';
    note: string;
}

export function deletionPlan(): DeletionPlan[] {
    return [
        { scope: 'wardrobe_items', action: 'delete_rows', note: 'Owner-scoped rows.' },
        { scope: 'images', action: 'delete_storage_objects', note: 'Private buckets + orphan cleanup.' },
        { scope: 'outfits', action: 'delete_rows', note: 'Includes history and plans.' },
        { scope: 'fit_evidence', action: 'delete_rows', note: 'Fit DNA inputs.' },
        { scope: 'style_evidence', action: 'delete_rows', note: 'Style DNA inputs.' },
        { scope: 'purchase_history', action: 'delete_rows', note: 'BuyCheck outcomes.' },
        { scope: 'price_watches', action: 'delete_rows', note: 'Watch rows + jobs.' },
        { scope: 'recommendation_feedback', action: 'delete_rows', note: 'Helpful/not-helpful.' },
        { scope: 'analytics_identifiers', action: 'anonymize', note: 'Break link to identity; aggregates remain.' },
        { scope: 'subscription_references', action: 'support_flow', note: 'Store billing records require the store subscription-cancel flow first.' },
    ];
}

// ── Export (7R) ──────────────────────────────────────────────────

export const EXPORT_SECTIONS = [
    'wardrobe_metadata',
    'outfit_history',
    'fit_style_preferences',
    'goals',
    'purchase_decisions',
    'user_notes',
    'saved_reference_metadata',
] as const;

export const EXPORT_EXCLUDED = ['secrets', 'internal_analytics', 'raw_private_images', 'provider_payloads'] as const;

export function buildExportManifest(sections: string[]): { included: string[]; excluded: string[] } {
    const allowed = new Set<string>(EXPORT_SECTIONS as readonly string[]);
    return { included: sections.filter((s) => allowed.has(s)), excluded: [...EXPORT_EXCLUDED] };
}

// ── Classification (7T) ─────────────────────────────────────────

export const DATA_CLASSIFICATION: Record<string, DataClass> = {
    display_name: 'account',
    wardrobe_item: 'wardrobe_private',
    outfit_history: 'wardrobe_private',
    fit_measurement: 'body_fit_private',
    fit_feedback: 'body_fit_private',
    purchase_record: 'purchase_private',
    price_watch: 'purchase_private',
    wardrobe_photo: 'image_private',
    reference_photo: 'image_private',
    error_report: 'operational',
    analytics_event: 'analytics',
    catalog_listing: 'public',
};

const EXTERNAL_AI_FORBIDDEN: DataClass[] = ['body_fit_private', 'image_private', 'purchase_private', 'wardrobe_private'];

/** Raw private classes must never go to external AI providers unprocessed. */
export function maySendToExternalAi(dataClass: DataClass): boolean {
    return !EXTERNAL_AI_FORBIDDEN.includes(dataClass);
}

export function classifyField(field: string): DataClass {
    return DATA_CLASSIFICATION[field] ?? 'operational';
}

// ── Privacy controls (7S) ────────────────────────────────────────

export const PRIVACY_CONTROLS = [
    'personalization',
    'fit_feedback_use',
    'image_analysis',
    'notifications',
    'analytics_diagnostics',
    'reference_image_retention',
    'external_ai_processing',
] as const;

export function defaultPrivacyControls(): Record<string, boolean> {
    return {
        personalization: true,
        fit_feedback_use: true,
        image_analysis: true,
        notifications: false,
        analytics_diagnostics: true,
        reference_image_retention: true,
        external_ai_processing: false,
    };
}

// ── Diagnostics (7BD) ───────────────────────────────────────────

export function buildDiagnosticPackage(input: {
    appVersion: string;
    environment: DiagnosticPackage['environment'];
    requestIds: string[];
    featureFlags: Record<string, string>;
    errors: string[];
    subscriptionCategory: DiagnosticPackage['subscriptionCategory'];
    syncStatus: DiagnosticPackage['syncStatus'];
}): DiagnosticPackage {
    return {
        appVersion: input.appVersion,
        environment: input.environment,
        requestIds: input.requestIds.slice(0, 20),
        featureFlags: input.featureFlags,
        sanitizedErrors: input.errors.map((e) => e.slice(0, 300)),
        subscriptionCategory: input.subscriptionCategory,
        syncStatus: input.syncStatus,
    };
}
