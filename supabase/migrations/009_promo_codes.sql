-- ============================================================
-- AIWardrobe — Promo Codes
-- Users must enter a valid promo code to unlock a 7-day free trial.
-- Without a promo code, they are directed to the paywall.
-- ============================================================

-- ── 1. PROMO CODES ──────────────────────────────────────────
-- Admin-defined codes. Each code grants a trial of N days.
CREATE TABLE IF NOT EXISTS public.promo_codes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code        TEXT UNIQUE NOT NULL,
    description TEXT,
    trial_days  INT NOT NULL DEFAULT 7,
    max_uses    INT,            -- NULL = unlimited
    used_count  INT NOT NULL DEFAULT 0,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    expires_at  TIMESTAMPTZ,    -- NULL = never expires
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_promo_codes_code ON public.promo_codes(code);
CREATE INDEX IF NOT EXISTS idx_promo_codes_active ON public.promo_codes(is_active);

-- Only service_role can write promo_codes. Users can read to validate.
ALTER TABLE public.promo_codes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can read active promo codes" ON public.promo_codes;
CREATE POLICY "Users can read active promo codes"
    ON public.promo_codes FOR SELECT
    USING (is_active = true);

DROP POLICY IF EXISTS "Service role full access promo_codes" ON public.promo_codes;
CREATE POLICY "Service role full access promo_codes"
    ON public.promo_codes FOR ALL
    USING (auth.role() = 'service_role');


-- ── 2. PROMO REDEMPTIONS ────────────────────────────────────
-- Tracks which user redeemed which code (one code per user).
CREATE TABLE IF NOT EXISTS public.promo_redemptions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    promo_code_id   UUID NOT NULL REFERENCES public.promo_codes(id) ON DELETE CASCADE,
    trial_days      INT NOT NULL,
    redeemed_at     TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT uq_promo_redemptions_user UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_promo_redemptions_user ON public.promo_redemptions(user_id);

ALTER TABLE public.promo_redemptions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own redemptions" ON public.promo_redemptions;
CREATE POLICY "Users can view own redemptions"
    ON public.promo_redemptions FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Service role full access promo_redemptions" ON public.promo_redemptions;
CREATE POLICY "Service role full access promo_redemptions"
    ON public.promo_redemptions FOR ALL
    USING (auth.role() = 'service_role');


-- ── 3. UPDATED_AT TRIGGER ───────────────────────────────────

DROP TRIGGER IF EXISTS set_promo_codes_updated_at ON public.promo_codes;
CREATE TRIGGER set_promo_codes_updated_at
    BEFORE UPDATE ON public.promo_codes
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


-- ── 4. SEED: default promo code for early users ─────────────
INSERT INTO public.promo_codes (code, description, trial_days, max_uses, is_active)
VALUES ('AIWARDROBE7', 'Early access 7-day trial', 7, NULL, true)
ON CONFLICT (code) DO NOTHING;
