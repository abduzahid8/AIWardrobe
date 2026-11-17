/**
 * Phase 7H — Versioned local persisted state.
 * Every persisted slice carries schemaVersion; unknown/corrupt state
 * recovers to safe defaults without silently dropping wardrobe data.
 */

export const LAUNCH_STORAGE_VERSIONS = {
    subscription: 3,
    dailyUsage: 2,
    wardrobe: 2,
    growth: 1,
} as const;

export type StorageSlice = keyof typeof LAUNCH_STORAGE_VERSIONS;

export interface VersionedState<T> {
    schemaVersion: number;
    data: T;
}

export function wrapWithVersion<T>(slice: StorageSlice, data: T): VersionedState<T> {
    return { schemaVersion: LAUNCH_STORAGE_VERSIONS[slice], data };
}

export type MigrationOutcome<T> =
    | { status: 'current'; state: VersionedState<T> }
    | { status: 'migrated'; state: VersionedState<T> }
    | { status: 'reset_to_defaults'; state: VersionedState<T>; reason: string };

/**
 * Migrates a persisted slice. `migrate` transforms old data forward;
 * on unknown versions or migration throws, returns safe defaults —
 * wardrobe slices must pass a merge that preserves items.
 */
export function migrateStoredState<T>(input: {
    slice: StorageSlice;
    stored: unknown;
    defaults: T;
    migrate?: (oldVersion: number, data: unknown) => T;
}): MigrationOutcome<T> {
    const current = LAUNCH_STORAGE_VERSIONS[input.slice];
    if (typeof input.stored !== 'object' || input.stored === null) {
        return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'not_an_object' };
    }
    const rec = input.stored as Record<string, unknown>;
    if (typeof rec.schemaVersion !== 'number') {
        // Unversioned legacy state: adopt data when it looks intact, else defaults.
        if (rec.data !== undefined && input.migrate) {
            try {
                return { status: 'migrated', state: wrapWithVersion(input.slice, input.migrate(0, rec.data)) };
            } catch {
                return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'legacy_migration_failed' };
            }
        }
        return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'missing_version' };
    }
    if (rec.schemaVersion === current) return { status: 'current', state: input.stored as VersionedState<T> };
    if (rec.schemaVersion > current) {
        return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'newer_than_app' };
    }
    if (!input.migrate) {
        return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'no_migration_path' };
    }
    try {
        return { status: 'migrated', state: wrapWithVersion(input.slice, input.migrate(rec.schemaVersion, rec.data)) };
    } catch {
        return { status: 'reset_to_defaults', state: wrapWithVersion(input.slice, input.defaults), reason: 'migration_threw' };
    }
}
