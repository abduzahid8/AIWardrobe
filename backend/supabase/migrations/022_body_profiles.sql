/**
 * Supabase migration 022 — body_profiles table for the body-fit try-on
 * feature (see docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md, Month 1).
 *
 * The table is intentionally forward-compatible:
 *   - `measurements` is JSONB so Month 3 can add SAM-derived zones
 *     (chest, waist, hips, etc.) without schema changes.
 *   - `mesh` is JSONB for SAM 3D Body output URLs (Month 3).
 *   - `privacy` is JSONB for retainSourcePhoto / retainMesh toggles.
 *   - `version` is bumped on every PATCH so Month 4 can show
 *     measurement-source history.
 *
 * One user → many profiles, with exactly one marked `is_active = true`.
 * The api/routes/bodyProfiles.js handler enforces that invariant by
 * deactivating all rows before activating a new one.
 */

-- 1. Table ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.body_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,

    -- Display
    name TEXT,

    -- Lifecycle
    status TEXT CHECK (status IN ('draft', 'analyzing', 'ready', 'failed')) DEFAULT 'draft' NOT NULL,
    is_active BOOLEAN DEFAULT FALSE NOT NULL,
    version INTEGER DEFAULT 1 NOT NULL,

    -- Identity
    gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),

    -- Height: high-confidence anchor
    height_value_cm NUMERIC(5, 1) NOT NULL,
    height_confidence TEXT CHECK (height_confidence IN ('low', 'medium', 'high')) DEFAULT 'medium' NOT NULL,
    height_source TEXT CHECK (height_source IN ('manual', 'apple_measure', 'photo_sam_3d_body', 'arkit_height', 'hybrid')) DEFAULT 'manual' NOT NULL,
    height_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Body shape
    weight_kg NUMERIC(5, 1),
    body_type TEXT CHECK (body_type IN ('ectomorph', 'average', 'mesomorph', 'endomorph', 'hourglass', 'pear')),

    -- Per-zone measurements (any subset). JSONB for forward-compat.
    -- Shape: { shoulderWidth?: { valueCm, confidence, source, updatedAt }, chest?: {...}, ... }
    measurements JSONB DEFAULT '{}'::jsonb NOT NULL,

    -- Mesh: populated by SAM 3D Body in Month 3
    mesh JSONB,

    -- Privacy toggles
    privacy JSONB DEFAULT '{"retainSourcePhoto": false, "retainMesh": true}'::jsonb NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- 2. One-active-per-user invariant ───────────────────────────────────────────
-- A partial unique index makes "only one active profile per user" a DB-level
-- guarantee (instead of relying on the API handler to enforce it).
CREATE UNIQUE INDEX IF NOT EXISTS body_profiles_one_active_per_user
    ON public.body_profiles (user_id)
    WHERE is_active = TRUE;

-- 3. Standard lookups ────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS body_profiles_user_id_idx ON public.body_profiles (user_id);
CREATE INDEX IF NOT EXISTS body_profiles_user_updated_idx ON public.body_profiles (user_id, updated_at DESC);

-- 4. updated_at trigger (matches handle_updated_at() used by other tables) ──
DROP TRIGGER IF EXISTS update_body_profiles_updated_at ON public.body_profiles;
CREATE TRIGGER update_body_profiles_updated_at
    BEFORE UPDATE ON public.body_profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- 5. RLS ─────────────────────────────────────────────────────────────────────
ALTER TABLE public.body_profiles ENABLE ROW LEVEL SECURITY;

-- A user can only see/modify their own body profiles.
DROP POLICY IF EXISTS "Users can view own body profiles" ON public.body_profiles;
CREATE POLICY "Users can view own body profiles" ON public.body_profiles
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own body profiles" ON public.body_profiles;
CREATE POLICY "Users can insert own body profiles" ON public.body_profiles
    FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own body profiles" ON public.body_profiles;
CREATE POLICY "Users can update own body profiles" ON public.body_profiles
    FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own body profiles" ON public.body_profiles;
CREATE POLICY "Users can delete own body profiles" ON public.body_profiles
    FOR DELETE USING (auth.uid() = user_id);

-- 6. Account deletion — already covered by ON DELETE CASCADE on user_id,
--    but include a defensive comment in case the FK is ever relaxed:
--    When a profile row is deleted, all related `body_analyses` rows
--    (added in migration 023) will cascade-delete automatically.
