import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

/**
 * body-profiles — Supabase-native replacement for the Express bodyProfiles
 * router (backend/api/routes/bodyProfiles.js).
 *
 * Single-endpoint dispatch (Edge Functions have one handler):
 *   { action: 'me' }                          → active (else most recent) profile
 *   { action: 'list' }                         → all profiles, newest first
 *   { action: 'create', profile: {...} }       → insert (deactivates others)
 *   { action: 'update', id, patch: {...} }     → partial update, version bump
 *   { action: 'activate', id }                 → mark active (exclusive)
 *   { action: 'delete', id }                   → delete
 *
 * Auth: Supabase JWT required; all rows are scoped to auth user id
 * server-side via the service-role client (mirrors authenticateToken +
 * per-route user_id scoping). Validation mirrors validators.js bounds.
 */

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const BODY_TYPES = ['ectomorph', 'average', 'mesomorph', 'endomorph', 'hourglass', 'pear'];
const GENDERS = ['male', 'female', 'other', 'prefer_not_to_say'];
const SOURCES = ['manual', 'apple_measure', 'photo_sam_3d_body', 'arkit_height', 'hybrid'];
const CONFIDENCES = ['low', 'medium', 'high'];
const STATUSES = ['draft', 'analyzing', 'ready', 'failed'];

function rowToProfile(row: any): any {
    if (!row) return null;
    return {
        id: row.id,
        userId: row.user_id,
        name: row.name || undefined,
        status: row.status,
        isActive: row.is_active,
        gender: row.gender || undefined,
        height: row.height_value_cm != null ? {
            valueCm: row.height_value_cm,
            confidence: row.height_confidence,
            source: row.height_source,
            updatedAt: row.height_updated_at,
        } : null,
        weightKg: row.weight_kg,
        bodyType: row.body_type,
        measurements: row.measurements || {},
        mesh: row.mesh || undefined,
        privacy: row.privacy || { retainSourcePhoto: false, retainMesh: true },
        version: row.version,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
    };
}

function validProfile(p: any): string | null {
    if (!p || typeof p !== 'object') return 'profile object is required';
    const h = p.height?.valueCm;
    if (h !== undefined && h !== null && (typeof h !== 'number' || h < 80 || h > 250)) {
        return 'height.valueCm must be 80-250 cm';
    }
    if (p.height?.source && !SOURCES.includes(p.height.source)) return 'invalid height.source';
    if (p.weightKg !== undefined && p.weightKg !== null && (typeof p.weightKg !== 'number' || p.weightKg < 20 || p.weightKg > 300)) {
        return 'weightKg must be 20-300 kg';
    }
    if (p.bodyType && !BODY_TYPES.includes(p.bodyType)) return 'invalid bodyType';
    if (p.gender && !GENDERS.includes(p.gender)) return 'invalid gender';
    return null;
}

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }
    const json = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })

    const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
    const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

    const authHeader = req.headers.get('Authorization') || ''
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : ''
    if (!token) return json({ error: 'Authentication required', code: 'NO_TOKEN' }, 401)
    const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        global: { headers: { Authorization: authHeader } },
    })
    const { data: { user }, error: authErr } = await userClient.auth.getUser()
    if (authErr || !user) return json({ error: 'Invalid or expired token', code: 'INVALID_TOKEN' }, 401)

    let body: any = {}
    try {
        body = await req.json()
    } catch {
        return json({ error: 'Invalid JSON body' }, 400)
    }
    const action = String(body.action || '')
    const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    const tbl = () => svc.from('body_profiles')

    try {
        if (action === 'me') {
            const { data: active, error } = await tbl().select('*')
                .eq('user_id', user.id).eq('is_active', true).maybeSingle()
            if (error) throw error
            if (active) return json({ profile: rowToProfile(active) })
            const { data: recent, error: recentErr } = await tbl().select('*')
                .eq('user_id', user.id).order('updated_at', { ascending: false }).limit(1).maybeSingle()
            if (recentErr) throw recentErr
            return json({ profile: recent ? rowToProfile(recent) : null })
        }

        if (action === 'list') {
            const { data, error } = await tbl().select('*')
                .eq('user_id', user.id).order('updated_at', { ascending: false })
            if (error) throw error
            return json({ profiles: (data || []).map(rowToProfile) })
        }

        if (action === 'create') {
            const p = body.profile || {}
            const invalid = validProfile(p)
            if (invalid) return json({ error: invalid }, 400)
            const wantsActive = p.isActive !== false
            if (wantsActive) {
                await tbl().update({ is_active: false }).eq('user_id', user.id).eq('is_active', true)
            }
            const { data, error } = await tbl().insert([{
                user_id: user.id,
                name: p.name || null,
                status: 'draft',
                is_active: wantsActive,
                gender: p.gender || null,
                height_value_cm: p.height?.valueCm ?? null,
                height_confidence: p.height?.confidence || 'medium',
                height_source: p.height?.source || 'manual',
                weight_kg: p.weightKg ?? null,
                body_type: p.bodyType || null,
                measurements: p.measurements || {},
                privacy: p.privacy || { retainSourcePhoto: false, retainMesh: true },
                version: 1,
            }]).select().single()
            if (error) throw error
            return json({ success: true, profile: rowToProfile(data) }, 201)
        }

        if (action === 'update') {
            const id = String(body.id || '')
            const patch = body.patch || {}
            if (!id) return json({ error: 'id is required' }, 400)
            const { data: existing, error: fetchErr } = await tbl().select('*')
                .eq('id', id).eq('user_id', user.id).maybeSingle()
            if (fetchErr) throw fetchErr
            if (!existing) return json({ error: 'Body profile not found' }, 404)
            const updates: Record<string, unknown> = {}
            if (patch.name !== undefined) updates.name = patch.name
            if (patch.gender !== undefined) {
                if (patch.gender && !GENDERS.includes(patch.gender)) return json({ error: 'invalid gender' }, 400)
                updates.gender = patch.gender
            }
            if (patch.weightKg !== undefined) updates.weight_kg = patch.weightKg
            if (patch.bodyType !== undefined) {
                if (patch.bodyType && !BODY_TYPES.includes(patch.bodyType)) return json({ error: 'invalid bodyType' }, 400)
                updates.body_type = patch.bodyType
            }
            if (patch.status !== undefined) {
                if (!STATUSES.includes(patch.status)) return json({ error: 'invalid status' }, 400)
                updates.status = patch.status
            }
            if (patch.privacy) updates.privacy = patch.privacy
            if (patch.height) {
                if (patch.height.valueCm !== undefined) updates.height_value_cm = patch.height.valueCm
                if (patch.height.confidence) {
                    if (!CONFIDENCES.includes(patch.height.confidence)) return json({ error: 'invalid confidence' }, 400)
                    updates.height_confidence = patch.height.confidence
                }
                if (patch.height.source) {
                    if (!SOURCES.includes(patch.height.source)) return json({ error: 'invalid height.source' }, 400)
                    updates.height_source = patch.height.source
                }
                updates.height_updated_at = new Date().toISOString()
            }
            if (patch.measurements) updates.measurements = { ...(existing.measurements || {}), ...patch.measurements }
            updates.version = (existing.version || 1) + 1
            const { data, error } = await tbl().update(updates)
                .eq('id', id).eq('user_id', user.id).select().single()
            if (error) throw error
            return json({ success: true, profile: rowToProfile(data) })
        }

        if (action === 'activate') {
            const id = String(body.id || '')
            if (!id) return json({ error: 'id is required' }, 400)
            const { data: existing, error: fetchErr } = await tbl().select('id')
                .eq('id', id).eq('user_id', user.id).maybeSingle()
            if (fetchErr) throw fetchErr
            if (!existing) return json({ error: 'Body profile not found' }, 404)
            await tbl().update({ is_active: false }).eq('user_id', user.id).eq('is_active', true)
            const { data, error } = await tbl().update({ is_active: true })
                .eq('id', id).eq('user_id', user.id).select().single()
            if (error) throw error
            return json({ success: true, profile: rowToProfile(data) })
        }

        if (action === 'delete') {
            const id = String(body.id || '')
            if (!id) return json({ error: 'id is required' }, 400)
            const { error } = await tbl().delete().eq('id', id).eq('user_id', user.id)
            if (error) throw error
            return json({ success: true })
        }

        return json({ error: 'Unknown action. Use me|list|create|update|activate|delete.' }, 400)
    } catch (e: any) {
        console.error('body-profiles error:', e?.message || e)
        return json({ error: 'Body profile request failed' }, 500)
    }
})
