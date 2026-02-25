-- ============================================
-- SUPABASE MIGRATION SCRIPT FOR AIWARDROBE
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. PROFILES (Extends auth.users)
-- ============================================
CREATE TABLE public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')) DEFAULT 'prefer_not_to_say',
    profile_image TEXT,
    
    -- Gmail OAuth tokens (encrypted in application logic if needed, or stored here securely)
    gmail_refresh_token TEXT,
    gmail_access_token TEXT,
    gmail_token_expiry TIMESTAMP WITH TIME ZONE,
    
    -- Security
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    last_failed_login TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    last_login_ip TEXT,
    
    -- Subscription Cache
    subscription_tier TEXT CHECK (subscription_tier IN ('free', 'premium', 'vip')) DEFAULT 'free',
    subscription_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Account Status
    is_active BOOLEAN DEFAULT TRUE,
    is_email_verified BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 2. CLOTHING ITEMS
-- ============================================
CREATE TABLE public.clothing_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    
    type TEXT NOT NULL,
    category TEXT CHECK (category IN ('Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Accessories', 'Other')) DEFAULT 'Other',
    
    -- Visuals
    image_url TEXT DEFAULT 'https://via.placeholder.com/150',
    thumbnail_url TEXT,
    
    -- Details
    color TEXT[],
    style TEXT CHECK (style IN ('Casual', 'Formal', 'Sport', 'Streetwear', 'Beach', 'Elegant', 'Business', 'Other')) DEFAULT 'Casual',
    brand TEXT DEFAULT '',
    material TEXT DEFAULT '',
    pattern TEXT CHECK (pattern IN ('Solid', 'Striped', 'Checkered', 'Floral', 'Printed', 'Other')) DEFAULT 'Solid',
    
    -- Organization
    season TEXT[] DEFAULT '{All Seasons}',
    occasion TEXT[] DEFAULT '{}',
    is_favorite BOOLEAN DEFAULT FALSE,
    is_archived BOOLEAN DEFAULT FALSE,
    notes TEXT DEFAULT '',
    tags TEXT[] DEFAULT '{}',
    
    -- Analytics
    wear_count INTEGER DEFAULT 0,
    last_worn_date TIMESTAMP WITH TIME ZONE,
    price NUMERIC DEFAULT 0,
    currency TEXT DEFAULT 'USD',
    purchase_date TIMESTAMP WITH TIME ZONE,
    purchase_location TEXT DEFAULT '',
    
    -- AI Metadata
    ai_generated BOOLEAN DEFAULT FALSE,
    description TEXT DEFAULT '',
    source_metadata JSONB DEFAULT '{}'::jsonb,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 3. SAVED OUTFITS
-- ============================================
CREATE TABLE public.saved_outfits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    
    date TEXT, -- Keeping as TEXT to match existing frontend logic, can be DATE
    
    -- Items stored as JSONB to replicate existing structure: [{id, type, image, x, y}]
    -- Ideally this should be a many-to-many relationship table, but preserving existing logic for now.
    items JSONB DEFAULT '[]'::jsonb,
    
    caption TEXT DEFAULT '',
    occasion TEXT DEFAULT 'casual',
    visibility TEXT DEFAULT 'Everyone',
    is_ootd BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 4. WEAR LOGS
-- ============================================
CREATE TABLE public.wear_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    
    clothing_item_id UUID REFERENCES public.clothing_items(id) ON DELETE CASCADE NOT NULL,
    outfit_id UUID REFERENCES public.saved_outfits(id) ON DELETE SET NULL,
    
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    occasion TEXT DEFAULT '',
    weather TEXT DEFAULT '',
    temperature NUMERIC,
    notes TEXT DEFAULT '',
    photo_url TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 5. SUBSCRIPTIONS
-- ============================================
CREATE TABLE public.subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    
    tier TEXT CHECK (tier IN ('free', 'premium', 'vip')) NOT NULL,
    status TEXT CHECK (status IN ('active', 'cancelled', 'expired', 'pending', 'trial')) NOT NULL,
    platform TEXT CHECK (platform IN ('apple', 'google', 'stripe', 'manual')) NOT NULL,
    
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    trial_end_date TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    
    auto_renew BOOLEAN DEFAULT TRUE,
    
    -- Platform IDs
    apple_original_transaction_id TEXT,
    google_purchase_token TEXT,
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    
    price NUMERIC NOT NULL,
    currency TEXT DEFAULT 'USD',
    product_id TEXT NOT NULL,
    
    last_receipt_data TEXT,
    last_receipt_validated_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 6. PAYMENTS
-- ============================================
CREATE TABLE public.payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    subscription_id UUID REFERENCES public.subscriptions(id) ON DELETE SET NULL,
    
    amount NUMERIC NOT NULL,
    currency TEXT DEFAULT 'USD',
    
    status TEXT CHECK (status IN ('pending', 'completed', 'failed', 'refunded', 'disputed')) NOT NULL,
    type TEXT CHECK (type IN ('subscription', 'one_time', 'refund', 'upgrade', 'renewal')) NOT NULL,
    platform TEXT CHECK (platform IN ('apple', 'google', 'stripe')) NOT NULL,
    
    -- Platform Transaction IDs
    transaction_id TEXT, -- Generic for Apple Transaction ID / Stripe Charge ID etc.
    apple_original_transaction_id TEXT,
    stripe_payment_intent_id TEXT,
    google_order_id TEXT,
    
    receipt_data TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 7. ENABLE ROW LEVEL SECURITY (RLS)
-- ============================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clothing_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.saved_outfits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.wear_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.payments ENABLE ROW LEVEL SECURITY;

-- ============================================
-- 8. CREATE RLS POLICIES
-- ============================================

-- Profiles
CREATE POLICY "Users can view own profile" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- Clothing Items
CREATE POLICY "Users can view own items" ON public.clothing_items FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own items" ON public.clothing_items FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own items" ON public.clothing_items FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own items" ON public.clothing_items FOR DELETE USING (auth.uid() = user_id);

-- Saved Outfits
CREATE POLICY "Users can view own outfits" ON public.saved_outfits FOR SELECT USING (auth.uid() = user_id); 
-- Note: Logic for 'Public' outfits would require an additional policy:
CREATE POLICY "Users can view public outfits" ON public.saved_outfits FOR SELECT USING (visibility = 'Public');

CREATE POLICY "Users can insert own outfits" ON public.saved_outfits FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own outfits" ON public.saved_outfits FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own outfits" ON public.saved_outfits FOR DELETE USING (auth.uid() = user_id);

-- Wear Logs
CREATE POLICY "Users can view own wear logs" ON public.wear_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own wear logs" ON public.wear_logs FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own wear logs" ON public.wear_logs FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own wear logs" ON public.wear_logs FOR DELETE USING (auth.uid() = user_id);

-- Subscriptions
CREATE POLICY "Users can view own subscriptions" ON public.subscriptions FOR SELECT USING (auth.uid() = user_id);

-- Payments
CREATE POLICY "Users can view own payments" ON public.payments FOR SELECT USING (auth.uid() = user_id);

-- ============================================
-- 9. FUNCTIONS & TRIGGERS
-- ============================================

-- Handle new user signup -> Create Profile
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, username, gender, profile_image)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'username', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'gender', 'prefer_not_to_say'),
        NEW.raw_user_meta_data->>'profile_image'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();
CREATE TRIGGER update_clothing_updated_at BEFORE UPDATE ON public.clothing_items FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();
CREATE TRIGGER update_outfits_updated_at BEFORE UPDATE ON public.saved_outfits FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();
CREATE TRIGGER update_wear_logs_updated_at BEFORE UPDATE ON public.wear_logs FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- ============================================
-- 10. STORAGE BUCKETS (Script cannot create buckets, but here is policy)
-- ============================================
-- You must manually create a bucket named 'user_uploads' in Supabase Dashboard.

-- Policy to allow authenticated uploads
-- user_uploads bucket:
-- INSERT: (bucket_id = 'user_uploads' AND auth.uid() = owner)
-- SELECT: (bucket_id = 'user_uploads')
