-- ============================================================
-- 011_admin_panel.sql
-- Admin role for shop catalog management
-- ============================================================
--
-- Adds is_admin flag to profiles and grants admin users
-- (identified by email or flag) full CRUD on shop_catalog
-- and upload access to the shop-catalog storage bucket.
-- ============================================================

-- ── 1. ADD is_admin COLUMN ────────────────────────────────────
ALTER TABLE public.profiles
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

-- Mark the designated admin email as admin
UPDATE public.profiles
    SET is_admin = TRUE
    WHERE email = 'info@aiwardrobe.club';

-- ── 2. HELPER FUNCTION: is_admin check ────────────────────────
-- Used in RLS policies so both the flag AND the hardcoded email
-- qualify as admin (belt-and-suspenders approach).
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT EXISTS (
        SELECT 1
          FROM public.profiles
         WHERE id = auth.uid()
           AND (is_admin = TRUE OR email = 'info@aiwardrobe.club')
    );
$$;

-- ── 3. SHOP_CATALOG RLS — admin write access ──────────────────
-- Drop the old service_role-only policy and replace with one that
-- also allows admin users to insert/update/delete.

-- Remove old restrictive write policy (if it exists)
DROP POLICY IF EXISTS shop_catalog_service_write ON public.shop_catalog;

-- Admin + service_role can write
CREATE POLICY shop_catalog_admin_write
    ON public.shop_catalog
    FOR ALL
    USING (
        auth.role() = 'service_role'
        OR public.is_admin()
    )
    WITH CHECK (
        auth.role() = 'service_role'
        OR public.is_admin()
    );

-- ── 4. STORAGE — admin upload to shop-catalog bucket ──────────
-- Drop old service_role-only policy
DROP POLICY IF EXISTS shop_catalog_storage_service_write ON storage.objects;

-- Admin + service_role can upload/delete
CREATE POLICY shop_catalog_storage_admin_write
    ON storage.objects
    FOR ALL
    USING (
        bucket_id = 'shop-catalog'
        AND (auth.role() = 'service_role' OR public.is_admin())
    )
    WITH CHECK (
        bucket_id = 'shop-catalog'
        AND (auth.role() = 'service_role' OR public.is_admin())
    );

-- ── 5. GRANT SELECT on shop_catalog to anon ───────────────────
-- (Already exists from 005, but ensure it's present)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'shop_catalog'
           AND policyname = 'shop_catalog_public_read'
    ) THEN
        CREATE POLICY shop_catalog_public_read ON public.shop_catalog
            FOR SELECT USING (is_active = TRUE);
    END IF;
END $$;

-- ── 6. INDEX on is_admin for fast lookups ─────────────────────
CREATE INDEX IF NOT EXISTS profiles_is_admin_idx ON public.profiles (is_admin) WHERE is_admin = TRUE;
