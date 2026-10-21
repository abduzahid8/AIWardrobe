import { supabase } from '../../lib/supabase';

export type AccountNameErrorCode = 'invalid_handle' | 'username_exists' | 'invalid_display_name' | 'availability_unavailable';
export class AccountNameError extends Error {
    constructor(public readonly code: AccountNameErrorCode) { super(code); this.name = 'AccountNameError'; }
}
export const normalizeHandle = (handle: string): string => handle.trim().toLowerCase();
export const isValidHandle = (handle: string): boolean => /^[a-z0-9_]{3,30}$/.test(normalizeHandle(handle));
export function displayNameOf(user?: { display_name?: string | null; username?: string | null; handle?: string | null } | null): string {
    return user?.display_name?.trim() || user?.username?.trim() || user?.handle || '';
}
export async function assertHandleAvailable(handle: string): Promise<void> {
    if (!isValidHandle(handle)) throw new AccountNameError('invalid_handle');
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let response;
    try {
        const query = supabase.rpc('is_username_available', { check_username: normalizeHandle(handle) });
        const request = typeof query.abortSignal === 'function' ? query.abortSignal(controller.signal) : query;
        response = await Promise.race([
            request,
            new Promise<never>((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new AccountNameError('availability_unavailable')); }, 15000); }),
        ]);
    } finally { if (timer) clearTimeout(timer); }
    const { data, error } = response;
    if (error || typeof data !== 'boolean') throw new AccountNameError('availability_unavailable');
    if (!data) throw new AccountNameError('username_exists');
}
export function profileNameUpdates(user: { handle?: string; display_name?: string; username?: string }, displayName: string, handle: string): Record<string, string> {
    const name = displayName.trim(), normalized = normalizeHandle(handle);
    if (!name || name.length > 200) throw new AccountNameError('invalid_display_name');
    if (!isValidHandle(normalized)) throw new AccountNameError('invalid_handle');
    const changes: Record<string, string> = {};
    if (name !== (user.display_name ?? user.username ?? '')) changes.display_name = name;
    if (normalized !== user.handle) changes.handle = normalized;
    return changes;
}
