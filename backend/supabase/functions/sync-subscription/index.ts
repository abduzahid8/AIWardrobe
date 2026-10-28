import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { enforceRateLimit } from '../_shared/rateLimit.ts'
import { normalizeRevenueCatSnapshot, revenueCatEnvironmentForUser } from '../_shared/revenuecatSnapshot.ts'
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })
serve(async (req: Request) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })
  if (req.method !== 'POST') return json({ error: 'Method not allowed' }, 405)
  const denied = await enforceRateLimit(req, { feature: 'sync-subscription', limit: 15, headers: corsHeaders, meterPaidWork: false })
  if (denied) return denied
  try {
    const key = Deno.env.get('REVENUECAT_SECRET_API_KEY') || Deno.env.get('REVENUECAT_API_KEY')
    const environment = Deno.env.get('REVENUECAT_ENVIRONMENT') || 'PRODUCTION'
    if (!key || !['PRODUCTION', 'SANDBOX'].includes(environment)) return json({ error: 'Subscription verification unavailable' }, 503)
    const admin = createClient(Deno.env.get('SUPABASE_URL') || '', Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') || '')
    const token = (req.headers.get('Authorization') || '').replace(/^Bearer\s+/i, '').trim()
    const { data: identity, error: authError } = await admin.auth.getUser(token)
    if (authError || !identity?.user) return json({ error: 'Unauthorized' }, 401)
    // The payload grants no authority: always reconcile only the authenticated owner.
    const owner = identity.user.id
    const response = await fetch(`https://api.revenuecat.com/v1/subscribers/${encodeURIComponent(owner)}`, {
      headers: { Authorization: `Bearer ${key}`, Accept: 'application/json' }, signal: AbortSignal.timeout(10000),
    })
    if (!response.ok) return json({ error: 'Subscription verification unavailable' }, 503)
    const ownerEnvironment = revenueCatEnvironmentForUser(owner, environment, Deno.env.get('REVENUECAT_SANDBOX_USER_IDS') || '')
    const account = normalizeRevenueCatSnapshot(owner, await response.json(), ownerEnvironment)
    const { data: applied, error } = await admin.rpc('apply_revenuecat_snapshots', {
      p_event: { id: `customer-sync:${owner}:${account.request_date_ms}`, type: 'CUSTOMER_SYNC' }, p_accounts: [account],
    })
    if (error || !applied || typeof applied.duplicate !== 'boolean') return json({ error: 'Subscription synchronization failed' }, 503)
    const { data: profile, error: profileError } = await admin.from('profiles').select('subscription_tier, subscription_expires_at').eq('id', owner).single()
    if (profileError || !profile) return json({ error: 'Subscription verification unavailable' }, 503)
    return json({ verified: true, tier: profile.subscription_tier, expiryDate: profile.subscription_expires_at })
  } catch { return json({ error: 'Subscription verification unavailable' }, 503) }
})
