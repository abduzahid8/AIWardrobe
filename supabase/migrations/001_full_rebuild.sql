-- ============================================================
-- AIWardrobe — Full Supabase Migration
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- ── 1. PROFILES ─────────────────────────────────────────────
-- Source: store/auth.ts, api/routes/auth.js, api/middleware/security.js

CREATE TABLE IF NOT EXISTS public.profiles (
    id              UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email           TEXT UNIQUE NOT NULL,
    username        TEXT UNIQUE NOT NULL DEFAULT '',
    gender          TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')) DEFAULT 'prefer_not_to_say',
    profile_image   TEXT,

    -- Subscription (denormalized for fast reads)
    subscription_tier       TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'premium', 'vip')),
    subscription_expires_at TIMESTAMPTZ,
    trial_started_at        TIMESTAMPTZ,

    -- Account Status
    is_active               BOOLEAN DEFAULT true,
    is_email_verified       BOOLEAN DEFAULT false,

    -- Security (account lockout)
    failed_login_attempts   INT DEFAULT 0,
    locked_until            TIMESTAMPTZ,
    last_failed_login       TIMESTAMPTZ,
    last_login_at           TIMESTAMPTZ,
    last_login_ip           TEXT,

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
CREATE POLICY "Users can insert own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, username, gender, profile_image)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'username', ''),
        NEW.raw_user_meta_data->>'gender',
        NEW.raw_user_meta_data->>'profile_image'
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ── 2. CLOTHING ITEMS ───────────────────────────────────────
-- Source: store/wardrobeSyncService.ts, src/hooks/queries/useWardrobeQuery.ts,
--         screens/WardrobeVideoScreen.tsx, screens/MyClosetScreen.tsx

CREATE TABLE IF NOT EXISTS public.clothing_items (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,

    -- Core fields (from wardrobeSyncService.ts mapRowToItem)
    image_url           TEXT,
    thumbnail_url       TEXT,
    category            TEXT DEFAULT 'top',
    sub_category        TEXT DEFAULT '',
    type                TEXT,                 -- from WardrobeVideoScreen save
    primary_color       TEXT DEFAULT '',
    color               TEXT,                 -- alias used by useWardrobeQuery
    color_hex           TEXT DEFAULT '#000000',
    pattern             TEXT DEFAULT 'solid',
    material            TEXT DEFAULT '',
    brand               TEXT,
    name                TEXT,
    style               TEXT,
    description         TEXT,
    season              TEXT,

    -- Arrays
    seasons             JSONB DEFAULT '[]',
    occasions           JSONB DEFAULT '[]',

    -- Wear tracking
    wear_count          INT DEFAULT 0,
    last_worn_at        TIMESTAMPTZ,
    is_favorite         BOOLEAN DEFAULT false,

    -- Detection metadata
    detection_confidence FLOAT,
    outfit_id           INT,

    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clothing_items_user ON public.clothing_items(user_id);
CREATE INDEX IF NOT EXISTS idx_clothing_items_user_created ON public.clothing_items(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_clothing_items_category ON public.clothing_items(user_id, category);

ALTER TABLE public.clothing_items ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own items" ON public.clothing_items;
CREATE POLICY "Users can view own items"
    ON public.clothing_items FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own items" ON public.clothing_items;
CREATE POLICY "Users can insert own items"
    ON public.clothing_items FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own items" ON public.clothing_items;
CREATE POLICY "Users can update own items"
    ON public.clothing_items FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own items" ON public.clothing_items;
CREATE POLICY "Users can delete own items"
    ON public.clothing_items FOR DELETE
    USING (auth.uid() = user_id);


-- ── 3. SAVED OUTFITS ────────────────────────────────────────
-- Source: screens/NewOutfitScreen.tsx, screens/ProfileScreen.tsx

CREATE TABLE IF NOT EXISTS public.saved_outfits (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    items       JSONB NOT NULL DEFAULT '[]',
    date        TEXT,
    occasion    TEXT,
    season      TEXT DEFAULT 'All',
    name        TEXT,
    caption     TEXT,
    visibility  TEXT DEFAULT 'Everyone',
    is_ootd     BOOLEAN DEFAULT false,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_saved_outfits_user ON public.saved_outfits(user_id);

ALTER TABLE public.saved_outfits ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own outfits" ON public.saved_outfits;
CREATE POLICY "Users can view own outfits"
    ON public.saved_outfits FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own outfits" ON public.saved_outfits;
CREATE POLICY "Users can insert own outfits"
    ON public.saved_outfits FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own outfits" ON public.saved_outfits;
CREATE POLICY "Users can update own outfits"
    ON public.saved_outfits FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own outfits" ON public.saved_outfits;
CREATE POLICY "Users can delete own outfits"
    ON public.saved_outfits FOR DELETE
    USING (auth.uid() = user_id);


-- ── 4. WEAR LOGS ────────────────────────────────────────────
-- Source: store/wardrobeSyncService.ts

CREATE TABLE IF NOT EXISTS public.wear_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    outfit_id           TEXT,
    item_ids            JSONB DEFAULT '[]',
    date                TEXT NOT NULL,
    occasion            TEXT,
    weather_temp        FLOAT,
    weather_condition   TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wear_logs_user ON public.wear_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_wear_logs_date ON public.wear_logs(user_id, date);

ALTER TABLE public.wear_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own wear logs" ON public.wear_logs;
CREATE POLICY "Users can view own wear logs"
    ON public.wear_logs FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own wear logs" ON public.wear_logs;
CREATE POLICY "Users can insert own wear logs"
    ON public.wear_logs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own wear logs" ON public.wear_logs;
CREATE POLICY "Users can update own wear logs"
    ON public.wear_logs FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own wear logs" ON public.wear_logs;
CREATE POLICY "Users can delete own wear logs"
    ON public.wear_logs FOR DELETE
    USING (auth.uid() = user_id);


-- ── 5. SUBSCRIPTIONS ────────────────────────────────────────
-- Source: api/middleware/subscriptionGuard.js, api/routes/subscription.js

CREATE TABLE IF NOT EXISTS public.subscriptions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    tier        TEXT NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'premium', 'vip')),
    status      TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'trial', 'cancelled', 'expired')),
    start_date  TIMESTAMPTZ DEFAULT now(),
    end_date    TIMESTAMPTZ,
    trial_end_date TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    auto_renew  BOOLEAN DEFAULT true,
    product_id  TEXT,
    platform    TEXT,    -- 'ios', 'android', 'web'
    apple_original_transaction_id TEXT,
    google_purchase_token TEXT,
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    last_receipt_data TEXT,
    last_receipt_validated_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON public.subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_active ON public.subscriptions(user_id, status, end_date);

ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;

-- Only service_role (edge functions) can write subscriptions.
-- Users may only VIEW their own subscription rows.
DROP POLICY IF EXISTS "Users can view own subscriptions" ON public.subscriptions;
CREATE POLICY "Users can view own subscriptions"
    ON public.subscriptions FOR SELECT
    USING (auth.uid() = user_id);


-- ── 6. PAYMENTS ─────────────────────────────────────────────
-- Source: api/routes/account.js (data export), api/routes/subscription.js

CREATE TABLE IF NOT EXISTS public.payments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    subscription_id UUID REFERENCES public.subscriptions(id) ON DELETE SET NULL,
    amount          DECIMAL(10,2),
    currency        TEXT DEFAULT 'USD',
    status          TEXT DEFAULT 'completed',
    product_id      TEXT,
    transaction_id  TEXT,
    platform        TEXT,
    receipt_data    TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payments_user ON public.payments(user_id);

ALTER TABLE public.payments ENABLE ROW LEVEL SECURITY;

-- Only service_role (edge functions) can write payments.
-- Users may only VIEW their own payment rows.
DROP POLICY IF EXISTS "Users can view own payments" ON public.payments;
CREATE POLICY "Users can view own payments"
    ON public.payments FOR SELECT
    USING (auth.uid() = user_id);


-- ── 7. STORAGE BUCKETS ──────────────────────────────────────
-- Source: api/routes/ai/wardrobe.js (AIWARDROBE), api/routes/account.js (user_uploads)

INSERT INTO storage.buckets (id, name, public)
VALUES ('AIWARDROBE', 'AIWARDROBE', true)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('user_uploads', 'user_uploads', false)
ON CONFLICT (id) DO NOTHING;

-- Storage policies: authenticated users can upload to their own folder
DROP POLICY IF EXISTS "Users can upload to AIWARDROBE" ON storage.objects;
CREATE POLICY "Users can upload to AIWARDROBE"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'AIWARDROBE'
        AND auth.role() = 'authenticated'
    );

DROP POLICY IF EXISTS "Public read AIWARDROBE" ON storage.objects;
CREATE POLICY "Public read AIWARDROBE"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'AIWARDROBE');

DROP POLICY IF EXISTS "Users can upload own files" ON storage.objects;
CREATE POLICY "Users can upload own files"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'user_uploads'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can read own files" ON storage.objects;
CREATE POLICY "Users can read own files"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'user_uploads'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can delete own files" ON storage.objects;
CREATE POLICY "Users can delete own files"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'user_uploads'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );


-- ── 8. UPDATED_AT TRIGGER ───────────────────────────────────

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_profiles_updated_at ON public.profiles;
CREATE TRIGGER set_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

DROP TRIGGER IF EXISTS set_clothing_items_updated_at ON public.clothing_items;
CREATE TRIGGER set_clothing_items_updated_at
    BEFORE UPDATE ON public.clothing_items
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

DROP TRIGGER IF EXISTS set_saved_outfits_updated_at ON public.saved_outfits;
CREATE TRIGGER set_saved_outfits_updated_at
    BEFORE UPDATE ON public.saved_outfits
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

DROP TRIGGER IF EXISTS set_subscriptions_updated_at ON public.subscriptions;
CREATE TRIGGER set_subscriptions_updated_at
    BEFORE UPDATE ON public.subscriptions
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
