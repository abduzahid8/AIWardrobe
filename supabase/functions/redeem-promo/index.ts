import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(JSON.stringify({ error: 'Missing authorization header' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;

    // User client — identify who is making the request
    const userClient = createClient(supabaseUrl, Deno.env.get('SUPABASE_ANON_KEY')!, {
      global: { headers: { Authorization: authHeader } },
    });

    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) {
      return new Response(JSON.stringify({ error: 'Invalid or expired token' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    const userId = user.id;

    // Parse request body
    const body = await req.json();
    const code = (body.code || '').trim().toUpperCase();

    if (!code) {
      return new Response(JSON.stringify({ error: 'Promo code is required' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Admin client — service role for writes
    const adminClient = createClient(supabaseUrl, supabaseServiceKey);

    // 1. Check if user already redeemed a promo code (one per user)
    const { data: existingRedemption } = await adminClient
      .from('promo_redemptions')
      .select('id, trial_days, redeemed_at')
      .eq('user_id', userId)
      .maybeSingle();

    if (existingRedemption) {
      return new Response(JSON.stringify({
        error: 'You have already redeemed a promo code',
        trial_days: existingRedemption.trial_days,
        redeemed_at: existingRedemption.redeemed_at,
      }), {
        status: 409,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // 2. Look up the promo code
    const { data: promoCode, error: lookupError } = await adminClient
      .from('promo_codes')
      .select('*')
      .eq('code', code)
      .eq('is_active', true)
      .maybeSingle();

    if (!promoCode) {
      return new Response(JSON.stringify({ error: 'Invalid or expired promo code' }), {
        status: 404,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // 3. Check expiry
    if (promoCode.expires_at && new Date(promoCode.expires_at) < new Date()) {
      return new Response(JSON.stringify({ error: 'This promo code has expired' }), {
        status: 410,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // 4. Check usage limit
    if (promoCode.max_uses !== null && promoCode.used_count >= promoCode.max_uses) {
      return new Response(JSON.stringify({ error: 'This promo code has reached its usage limit' }), {
        status: 410,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // 5. All checks passed — redeem the code atomically
    const trialDays = promoCode.trial_days;
    const trialStartDate = new Date().toISOString();

    // Insert redemption record
    const { error: redemptionError } = await adminClient
      .from('promo_redemptions')
      .insert({
        user_id: userId,
        promo_code_id: promoCode.id,
        trial_days: trialDays,
      });

    if (redemptionError) {
      console.error('Failed to insert redemption:', redemptionError);
      return new Response(JSON.stringify({ error: 'Failed to redeem promo code' }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Increment used_count on the promo code
    await adminClient
      .from('promo_codes')
      .update({ used_count: promoCode.used_count + 1 })
      .eq('id', promoCode.id);

    // Set trial_started_at on the user's profile
    await adminClient
      .from('profiles')
      .update({ trial_started_at: trialStartDate } as any)
      .eq('id', userId);

    // Also create a subscription row with status='trial'
    const trialEndDate = new Date();
    trialEndDate.setDate(trialEndDate.getDate() + trialDays);

    await adminClient
      .from('subscriptions')
      .insert({
        user_id: userId,
        tier: 'premium',
        status: 'trial',
        start_date: trialStartDate,
        end_date: trialEndDate.toISOString(),
        trial_end_date: trialEndDate.toISOString(),
        auto_renew: false,
        platform: 'promo',
      });

    return new Response(JSON.stringify({
      success: true,
      trial_days: trialDays,
      trial_started_at: trialStartDate,
      message: `${trialDays}-day free trial activated!`,
    }), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Promo code redemption error:', error);
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});
