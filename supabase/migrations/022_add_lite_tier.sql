-- Add 'lite' tier to CHECK constraints on profiles and subscriptions tables

ALTER TABLE public.profiles
  DROP CONSTRAINT IF EXISTS profiles_subscription_tier_check,
  ADD CONSTRAINT profiles_subscription_tier_check
    CHECK (subscription_tier IN ('free', 'lite', 'premium', 'vip'));

ALTER TABLE public.subscriptions
  DROP CONSTRAINT IF EXISTS subscriptions_tier_check,
  ADD CONSTRAINT subscriptions_tier_check
    CHECK (tier IN ('free', 'lite', 'premium', 'vip'));
