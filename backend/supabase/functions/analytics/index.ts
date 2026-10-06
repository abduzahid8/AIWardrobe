import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

/**
 * analytics — Supabase-native replacement for Express POST
 * /api/analytics/events (backend/api/routes/analytics.js).
 *
 * The Express handler only validates + structured-logs events (PostHog
 * forwarding is an unimplemented TODO there), so this port does exactly
 * the same: validate the batch, log it server-side, acknowledge.
 *
 * Contract (kept identical):
 *   Request:  { events: [{name, timestamp?, properties?}...≤200], sessionId? }
 *   Success:  { success: true, received: N }
 *
 * Auth: optional (mirrors optionalAuth) — user id attached when valid.
 */

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }
    const json = (body: unknown, status = 200) =>
        new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })

    const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
    const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!

    let body: any = {}
    try {
        body = await req.json()
    } catch {
        return json({ error: 'events array is required' }, 400)
    }
    const events = body.events
    if (!Array.isArray(events) || events.length === 0) {
        return json({ error: 'events array is required' }, 400)
    }
    if (events.length > 200) {
        return json({ error: 'Maximum 200 events per batch' }, 400)
    }

    let userId = 'anonymous'
    const authHeader = req.headers.get('Authorization') || ''
    if (authHeader.startsWith('Bearer ')) {
        try {
            const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
                global: { headers: { Authorization: authHeader } },
            })
            const { data: { user } } = await userClient.auth.getUser()
            if (user) userId = user.id
        } catch {
            // anonymous
        }
    }

    for (const event of events) {
        if (!event || typeof event.name !== 'string') continue
        console.log(JSON.stringify({
            type: 'analytics',
            name: event.name.slice(0, 120),
            userId,
            sessionId: body.sessionId || event.sessionId,
            timestamp: event.timestamp,
            properties: event.properties || {},
        }))
    }

    return json({ success: true, received: events.length })
})
