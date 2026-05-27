/**
 * RevenueCat Webhook Handler
 * 
 * Receives server-to-server events from RevenueCat for:
 * - Initial purchase
 * - Renewal
 * - Cancellation
 * - Refund
 * - Billing issues
 * 
 * This is the ONLY authorized writer to the subscriptions and payments tables.
 * RLS policies block all client-side writes to these tables.
 */
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// RevenueCat webhook event types
interface RevenueCatEvent {
  event: {
    type: 'INITIAL_PURCHASE' | 'RENEWAL' | 'CANCELLATION' | 'UNCANCELLATION' | 'NON_RENEWING_PURCHASE' | 'REFUND' | 'BILLING_ISSUE' | 'PRODUCT_CHANGE' | 'SUBSCRIPTION_PAUSED' | 'TRANSFER';
    app_user_id: string;
    subscriber_attributes?: Record<string, any>;
    customer_info?: {
      original_app_user_id: string;
      first_seen: string;
      subscriptions: Record<string, any>;
      entitlements: Record<string, any>;
    };
    id?: string;
    transaction_id?: string;
    product_id?: string;
    price?: number;
    currency?: string;
    store?: string;
    trial_start_at_ms?: number;
    trial_end_at_ms?: number;
    cancel_reason?: 'UNSUBSCRIBE' | 'BILLING_ERROR' | 'REFUND' | 'UNKNOWN';
    expiration_at_ms?: number;
    entitlement_id?: string;
    entitlement_ids?: string[];
    presented_offering_id?: string;
    purchased_at_ms?: number;
    event_timestamp_ms?: number;
  }
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Verify RevenueCat API key is configured
    const REVENUECAT_API_KEY = Deno.env.get('REVENUECAT_API_KEY')
    if (!REVENUECAT_API_KEY) {
      console.error('[RevenueCat] REVENUECAT_API_KEY not configured')
      return new Response(
        JSON.stringify({ error: 'Server configuration error' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
      )
    }

    // Verify request authenticity via RevenueCat shared-secret Authorization header.
    // Configure RevenueCat dashboard → Integrations → Webhook → Authorization header
    // to send "Bearer <REVENUECAT_WEBHOOK_SECRET>". The secret value below must match.
    const WEBHOOK_SECRET = Deno.env.get('REVENUECAT_WEBHOOK_SECRET') || REVENUECAT_API_KEY
    const authHeader = req.headers.get('Authorization') || ''
    const expected = `Bearer ${WEBHOOK_SECRET}`
    if (!authHeader || authHeader !== expected) {
      console.error('[RevenueCat] Rejected webhook with invalid/missing Authorization header')
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 401 }
      )
    }

    const payload: RevenueCatEvent = await req.json()
    const event = payload.event

    // Basic payload validation
    if (!event?.type || !event?.app_user_id) {
      return new Response(
        JSON.stringify({ error: 'Invalid payload: missing event type or app_user_id' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
      )
    }

    console.log(`[RevenueCat] Received ${event.type} for user ${event.app_user_id}`)

    // Initialize Supabase with service role key for admin access
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // ── Idempotency ────────────────────────────────────────────────
    // RevenueCat retries failed webhook deliveries. Without a guard, a refund
    // or renewal could be applied multiple times (double-counted payments,
    // duplicated subscription rows). We dedupe on (source, event_id).
    const eventId = event.id
      || (event.transaction_id && event.type
            ? `${event.type}:${event.transaction_id}:${event.event_timestamp_ms ?? ''}`
            : null)
    if (eventId) {
      const { error: insErr } = await supabaseAdmin
        .from('processed_webhooks')
        .insert({ source: 'revenuecat', event_id: eventId })
      if (insErr) {
        // Unique violation = already processed. Acknowledge with 200 so RC stops retrying.
        if ((insErr as any).code === '23505') {
          console.log(`[RevenueCat] Duplicate event ignored: ${eventId}`)
          return new Response(
            JSON.stringify({ success: true, duplicate: true }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
          )
        }
        // Any other error: log and continue (better than dropping the event).
        console.error('[RevenueCat] Idempotency insert error:', insErr)
      }
    }

    // Map RevenueCat user ID to our Supabase user ID.
    // When iapService.identify() is called before purchase, app_user_id is the
    // Supabase user ID. If identify() was never called, RevenueCat uses an
    // anonymous ID like "$RCAnonymousUserId:xxx" which won't match any profile.
    let userId = event.app_user_id

    // Verify the user exists
    let { data: profile, error: profileError } = await supabaseAdmin
      .from('profiles')
      .select('id')
      .eq('id', userId)
      .single()

    // Fallback: if app_user_id is an anonymous RC ID, try to resolve the real
    // user from the original_app_user_id in customer_info, or from an existing
    // subscription row that the client-side fallback may have created.
    if ((profileError || !profile) && userId.startsWith('$RC')) {
      console.warn('[RevenueCat] Anonymous RC user ID detected, attempting resolution:', userId)

      // Try original_app_user_id from customer_info
      const originalId = event.customer_info?.original_app_user_id
      if (originalId && !originalId.startsWith('$RC')) {
        const { data: origProfile } = await supabaseAdmin
          .from('profiles')
          .select('id')
          .eq('id', originalId)
          .single()
        if (origProfile) {
          console.log('[RevenueCat] Resolved user via original_app_user_id:', originalId)
          userId = originalId
          profile = origProfile
          profileError = null
        }
      }

      // Fallback: find the most recent subscription row for this product_id
      // that was created recently (client-side fallback inserts with the real user_id)
      if (!profile && event.product_id) {
        const cutoff = new Date(Date.now() - 10 * 60 * 1000).toISOString() // last 10 min
        const { data: recentSub } = await supabaseAdmin
          .from('subscriptions')
          .select('user_id')
          .eq('product_id', event.product_id)
          .eq('status', 'active')
          .gte('created_at', cutoff)
          .order('created_at', { ascending: false })
          .limit(1)
          .maybeSingle()

        if (recentSub) {
          console.log('[RevenueCat] Resolved user via recent subscription row:', recentSub.user_id)
          userId = recentSub.user_id
          const { data: resolvedProfile } = await supabaseAdmin
            .from('profiles')
            .select('id')
            .eq('id', userId)
            .single()
          if (resolvedProfile) {
            profile = resolvedProfile
            profileError = null
          }
        }
      }
    }

    if (profileError || !profile) {
      console.warn('[RevenueCat] User not found, but returning 200 OK to acknowledge receipt:', userId, { originalAppUserId: event.app_user_id })
      return new Response(
        JSON.stringify({ success: true, message: 'User not found in database, event acknowledged' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
      )
    }

    const productId = event.product_id || ''
    const tier = mapProductToTier(productId)
    const platform = mapStoreToPlatform(event.store)

    switch (event.type) {
      case 'INITIAL_PURCHASE':
      case 'RENEWAL':
        if (tier === 'free') {
          console.error('[RevenueCat] Ignoring purchase event with unmapped product:', productId)
          break
        }
        await handlePurchase(supabaseAdmin, userId, event, tier, platform, event.type === 'INITIAL_PURCHASE')
        break
      case 'CANCELLATION':
        await handleCancellation(supabaseAdmin, userId, event)
        break
      case 'REFUND':
        await handleRefund(supabaseAdmin, userId, event)
        break
      case 'BILLING_ISSUE':
        await handleBillingIssue(supabaseAdmin, userId, event)
        break
      case 'UNCANCELLATION':
        await handleUncancellation(supabaseAdmin, userId, event)
        break
      case 'PRODUCT_CHANGE':
        if (tier === 'free') {
          console.error('[RevenueCat] Ignoring product change event with unmapped product:', productId)
          break
        }
        await handleProductChange(supabaseAdmin, userId, event, tier, platform)
        break
      default:
        console.log(`[RevenueCat] Unhandled event type: ${event.type}`)
    }

    return new Response(
      JSON.stringify({ success: true }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
    )
  } catch (error: any) {
    console.error('[RevenueCat] Webhook error:', error)
    return new Response(
      JSON.stringify({ error: error.message || 'Internal server error' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
    )
  }
})

function mapProductToTier(productId: string): 'premium' | 'vip' | 'free' {
  const id = productId.toLowerCase();
  if (id.includes('vip') || id.includes('yearly')) return 'vip'
  if (id.includes('premium') || id.includes('pro')) return 'premium'
  return 'free'
}

function mapStoreToPlatform(store?: string): 'apple' | 'google' | 'stripe' | 'manual' {
  switch ((store || '').toLowerCase()) {
    case 'app_store': return 'apple'
    case 'play_store': return 'google'
    case 'stripe': return 'stripe'
    default: return 'manual'
  }
}

function toIsoFromMs(timestampMs?: number | null): string | null {
  if (!timestampMs || Number.isNaN(timestampMs)) return null
  return new Date(timestampMs).toISOString()
}

async function handlePurchase(
  supabase: any,
  userId: string,
  event: RevenueCatEvent['event'],
  tier: 'premium' | 'vip',
  platform: 'apple' | 'google' | 'stripe' | 'manual',
  isInitial: boolean
) {
  const expiryDate = toIsoFromMs(event.expiration_at_ms)
  const startDate = toIsoFromMs(event.purchased_at_ms || event.event_timestamp_ms || Date.now()) || new Date().toISOString()
  const endDate = expiryDate || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()

  // For RENEWAL events, update the existing subscription instead of inserting a duplicate
  if (!isInitial) {
    const { data: existing } = await supabase
      .from('subscriptions')
      .select('id')
      .eq('user_id', userId)
      .eq('product_id', event.product_id)
      .in('status', ['active', 'cancelled', 'expired'])
      .order('created_at', { ascending: false })
      .limit(1)

    if (existing && existing.length > 0) {
      const { error: updateError } = await supabase
        .from('subscriptions')
        .update({
          status: 'active',
          end_date: endDate,
          auto_renew: true,
          cancelled_at: null,
        })
        .eq('id', existing[0].id)

      if (updateError) {
        console.error('[RevenueCat] Failed to update subscription on renewal:', updateError)
      }

      // Insert payment record for the renewal
      const { error: paymentError } = await supabase
        .from('payments')
        .insert({
          user_id: userId,
          subscription_id: existing[0].id,
          amount: event.price || 0,
          currency: event.currency || 'USD',
          status: 'completed',
          product_id: event.product_id,
          transaction_id: event.transaction_id,
          platform,
        })

      if (paymentError) {
        console.error('[RevenueCat] Failed to insert renewal payment:', paymentError)
      }

      // Update profile denormalized fields
      await supabase
        .from('profiles')
        .update({
          subscription_tier: tier,
          subscription_expires_at: endDate,
        })
        .eq('id', userId)

      console.log(`[RevenueCat] Renewal recorded for ${userId}`)
      return
    }
    // If no existing subscription found, fall through to insert (edge case)
  }

  // Insert new subscription record (INITIAL_PURCHASE or no existing record)
  const { data: subscription, error: subError } = await supabase
    .from('subscriptions')
    .insert({
      user_id: userId,
      tier,
      status: 'active',
      start_date: startDate,
      end_date: endDate,
      trial_end_date: toIsoFromMs(event.trial_end_at_ms),
      auto_renew: true,
      product_id: event.product_id,
      platform,
      apple_original_transaction_id: platform === 'apple' ? event.transaction_id : null,
      google_purchase_token: platform === 'google' ? event.transaction_id : null,
      stripe_subscription_id: platform === 'stripe' ? event.transaction_id : null,
    })
    .select('id')
    .single()

  if (subError) {
    console.error('[RevenueCat] Failed to insert subscription:', subError)
    return
  }

  // Insert payment record
  const { error: paymentError } = await supabase
    .from('payments')
    .insert({
      user_id: userId,
      subscription_id: subscription.id,
      amount: event.price || 0,
      currency: event.currency || 'USD',
      status: 'completed',
      product_id: event.product_id,
      transaction_id: event.transaction_id,
      platform,
    })

  if (paymentError) {
    console.error('[RevenueCat] Failed to insert payment:', paymentError)
  }

  // Update profile denormalized fields
  await supabase
    .from('profiles')
    .update({
      subscription_tier: tier,
      subscription_expires_at: endDate,
    })
    .eq('id', userId)

  console.log(`[RevenueCat] ${isInitial ? 'Initial purchase' : 'Renewal'} recorded for ${userId}`)
}

async function handleCancellation(supabase: any, userId: string, event: RevenueCatEvent['event']) {
  // Find active subscription and mark as cancelled
  const { data: subscriptions } = await supabase
    .from('subscriptions')
    .select('id, end_date, tier')
    .eq('user_id', userId)
    .eq('product_id', event.product_id)
    .in('status', ['active', 'trial'])
    .order('created_at', { ascending: false })
    .limit(1)

  if (subscriptions && subscriptions.length > 0) {
    const sub = subscriptions[0]
    await supabase
      .from('subscriptions')
      .update({
        status: 'cancelled',
        cancelled_at: new Date().toISOString(),
        auto_renew: false,
      })
      .eq('id', sub.id)

    // Keep the current tier until end_date — user still has access.
    // Only set subscription_expires_at; do NOT downgrade tier yet.
    // The client-side subscriptionStore checks end_date to determine access.
    await supabase
      .from('profiles')
      .update({
        subscription_expires_at: sub.end_date, // Keep expiry date so they keep access until then
      })
      .eq('id', userId)

    console.log(`[RevenueCat] Cancellation recorded for ${userId} — access retained until ${sub.end_date}`)
  }
}

async function handleRefund(supabase: any, userId: string, event: RevenueCatEvent['event']) {
  // Update payment record
  await supabase
    .from('payments')
    .update({ status: 'refunded' })
    .eq('transaction_id', event.transaction_id)

  // Downgrade subscription
  await supabase
    .from('subscriptions')
    .update({ status: 'expired' })
    .eq('user_id', userId)
    .eq('product_id', event.product_id)

  // Update profile
  await supabase
    .from('profiles')
    .update({
      subscription_tier: 'free',
      subscription_expires_at: null,
    })
    .eq('id', userId)

  console.log(`[RevenueCat] Refund recorded for ${userId}`)
}

async function handleBillingIssue(supabase: any, userId: string, event: RevenueCatEvent['event']) {
  // Log billing issue but don't change subscription status immediately
  // RevenueCat will send a CANCELLATION event if the billing issue is not resolved
  console.log(`[RevenueCat] Billing issue for ${userId}:`, event.cancel_reason)
}

async function handleUncancellation(supabase: any, userId: string, event: RevenueCatEvent['event']) {
  // User re-enabled auto-renew
  await supabase
    .from('subscriptions')
    .update({
      status: 'active',
      auto_renew: true,
      cancelled_at: null,
    })
    .eq('user_id', userId)
    .eq('product_id', event.product_id)

  console.log(`[RevenueCat] Uncancellation recorded for ${userId}`)
}

async function handleProductChange(supabase: any, userId: string, event: RevenueCatEvent['event'], tier: 'premium' | 'vip', platform: string) {
  // Handle upgrade/downgrade
  const newTier = mapProductToTier(event.product_id || '')
  const expiryDate = toIsoFromMs(event.expiration_at_ms)

  const { data: subs } = await supabase
    .from('subscriptions')
    .select('id, end_date')
    .eq('user_id', userId)
    .in('status', ['active', 'trial'])
    .order('created_at', { ascending: false })
    .limit(1)

  if (subs && subs.length > 0) {
    await supabase
      .from('subscriptions')
      .update({
        tier: newTier,
        product_id: event.product_id,
        ...(expiryDate ? { end_date: expiryDate } : {}),
      })
      .eq('id', subs[0].id)
  }

  await supabase
    .from('profiles')
    .update({
      subscription_tier: newTier,
      ...(expiryDate ? { subscription_expires_at: expiryDate } : {}),
    })
    .eq('id', userId)

  console.log(`[RevenueCat] Product change recorded for ${userId}: ${tier} -> ${newTier}`)
}
