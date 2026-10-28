import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { normalizeRevenueCatSnapshot, revenueCatEnvironmentForUser } from '../_shared/revenuecatSnapshot.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
})

// Reconcile Customer Info rather than inventing subscription state from event
// order. The database commits all account changes and the dedupe ledger together.
serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })
  if (req.method !== 'POST') return json({ error: 'Method not allowed' }, 405)
  const secret = Deno.env.get('REVENUECAT_WEBHOOK_SECRET')
  const apiKey = Deno.env.get('REVENUECAT_SECRET_API_KEY') || Deno.env.get('REVENUECAT_API_KEY')
  if (!secret || !apiKey || secret === apiKey || /^(appl_|goog_|test_)/.test(secret)) return json({ error: 'Webhook configuration unavailable' }, 503)
  if (req.headers.get('Authorization') !== `Bearer ${secret}`) return json({ error: 'Unauthorized' }, 401)
  try {
    const text = await req.text()
    if (text.length > 262144) return json({ error: 'Webhook payload too large' }, 413)
    const event = JSON.parse(text)?.event
    if (!event || typeof event.type !== 'string') return json({ error: 'Invalid event' }, 400)
    if (event.type === 'TEST') return json({ success: true, test: true })
    if (typeof event.id !== 'string' || !event.id || event.id.length > 250) return json({ error: 'Event ID required' }, 400)
    const environment = Deno.env.get('REVENUECAT_ENVIRONMENT') || 'PRODUCTION'
    if (!['PRODUCTION', 'SANDBOX'].includes(environment)) return json({ error: 'Invalid webhook environment configuration' }, 503)
    const sandboxUsers = Deno.env.get('REVENUECAT_SANDBOX_USER_IDS') || ''
    if (event.environment && event.environment !== environment && !(event.environment === 'SANDBOX' && sandboxUsers.trim())) return json({ success: true, ignored: 'different_environment' })
    const admin = createClient(Deno.env.get('SUPABASE_URL') || '', Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '')
    let ownerIds: string[]
    if (event.type === 'TRANSFER') {
      if (!Array.isArray(event.transferred_from) || !Array.isArray(event.transferred_to)) return json({ error: 'Invalid transfer identities' }, 400)
      ownerIds = [...event.transferred_from, ...event.transferred_to].filter((value: unknown) => typeof value === 'string' && UUID.test(value))
    } else {
      if (typeof event.app_user_id !== 'string') return json({ error: 'App user ID required' }, 400)
      // A known primary UUID takes precedence; do not redirect a missing primary
      // account to a different alias or to somebody who bought the same product.
      ownerIds = UUID.test(event.app_user_id) ? [event.app_user_id] :
        [event.original_app_user_id, ...(Array.isArray(event.aliases) ? event.aliases : [])]
          .filter((value: unknown) => typeof value === 'string' && UUID.test(value))
    }
    ownerIds = [...new Set(ownerIds.map(id => id.toLowerCase()))].sort()
    if (!ownerIds.length || ownerIds.length > 50) return json({ error: 'Unresolved webhook identity' }, 503)
    const { data: profiles, error: identityError } = await admin.from('profiles').select('id').in('id', ownerIds)
    if (identityError || !profiles?.length || (event.type !== 'TRANSFER' && profiles.length !== 1)) return json({ error: 'Unresolved webhook identity' }, 503)
    if (event.type !== 'TRANSFER' && UUID.test(event.app_user_id) && profiles[0].id !== event.app_user_id.toLowerCase()) return json({ error: 'Unresolved webhook identity' }, 503)
    const accounts = []
    // Bound external work below the webhook delivery deadline. Transfers normally
    // contain two known accounts; one batch is applied only after every fetch succeeds.
    const deadline = AbortSignal.timeout(45000)
    for (const profile of profiles) {
      const ownerEnvironment = revenueCatEnvironmentForUser(profile.id, environment, sandboxUsers)
      if (event.environment && event.environment !== ownerEnvironment) continue
      const response = await fetch(`https://api.revenuecat.com/v1/subscribers/${encodeURIComponent(profile.id)}`, {
        headers: { Authorization: `Bearer ${apiKey}`, Accept: 'application/json' },
        signal: AbortSignal.any([deadline, AbortSignal.timeout(10000)]),
      })
      if (!response.ok) return json({ error: 'Subscription verification unavailable' }, 503)
      accounts.push(normalizeRevenueCatSnapshot(profile.id, await response.json(), ownerEnvironment))
    }
    if (!accounts.length) return json({ success: true, ignored: 'different_environment' })
    const { data, error } = await admin.rpc('apply_revenuecat_snapshots', {
      p_event: { ...event, resolved_user_id: event.type === 'TRANSFER' ? null : profiles[0].id },
      p_accounts: accounts,
    })
    if (error || !data || typeof data.duplicate !== 'boolean') return json({ error: 'Subscription synchronization failed' }, 503)
    return json({ success: true, duplicate: data.duplicate })
  } catch (error) {
    return json({ error: error instanceof SyntaxError ? 'Invalid JSON request' : 'Subscription synchronization unavailable' }, error instanceof SyntaxError ? 400 : 503)
  }
})
