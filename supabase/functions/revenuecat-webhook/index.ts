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

    const payload: RevenueCatEvent = await req.json()
    const event = payload.event

    console.log(`[RevenueCat] Received ${event.type} for user ${event.app_user_id}`)

    // Initialize Supabase with service role key for admin access
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // Map RevenueCat user ID to our user ID (should be the same if properly configured)
    const userId = event.app_user_id

    // Verify the user exists
    const { data: profile, error: profileError } = await supabaseAdmin
      .from('profiles')
      .select('id')
      .eq('id', userId)
      .single()

    if (profileError || !profile) {
      console.error('[RevenueCat] User not found:', userId)
      return new Response(
        JSON.stringify({ error: 'User not found' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 404 }
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
  if (productId.includes('premium')) return 'premium'
  if (productId.includes('vip')) return 'vip'
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

  // Insert subscription record
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
    .select('id, end_date')
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

    // Update profile if this was the active subscription
    await supabase
      .from('profiles')
      .update({
        subscription_tier: 'free',
        subscription_expires_at: sub.end_date, // Keep expiry date so they keep access until then
      })
      .eq('id', userId)

    console.log(`[RevenueCat] Cancellation recorded for ${userId}`)
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
  
  await supabase
    .from('subscriptions')
    .update({
      tier: newTier,
      product_id: event.product_id,
    })
    .eq('user_id', userId)
    .in('status', ['active', 'trial'])

  await supabase
    .from('profiles')
    .update({ subscription_tier: newTier })
    .eq('id', userId)

  console.log(`[RevenueCat] Product change recorded for ${userId}: ${tier} -> ${newTier}`)
}
