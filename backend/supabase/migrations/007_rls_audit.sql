-- ============================================================
-- 007_rls_audit.sql
-- Row-Level Security Audit + Drift Guard
-- ============================================================
--
-- Goal: make it impossible to accidentally ship a public table
-- without RLS enabled. This migration:
--
-- 1. Re-asserts RLS is ENABLED on every user-facing table (idempotent).
-- 2. Adds a verification block that aborts the migration if any
--    table in the `public` schema lacks RLS.
-- 3. Confirms the expected baseline policies exist for per-user tables.
--
-- Run it after all prior migrations. Re-running is safe.
-- ============================================================

-- ---- 1. Re-enable RLS everywhere (idempotent) ----------------
ALTER TABLE public.profiles         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clothing_items   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.saved_outfits    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.wear_logs        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.payments         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.app_config       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.shop_catalog     ENABLE ROW LEVEL SECURITY;

-- ---- 2. Drift guard: fail if ANY public table lacks RLS ------
DO $$
DECLARE
    unprotected text;
BEGIN
    SELECT string_agg(schemaname || '.' || tablename, ', ')
      INTO unprotected
      FROM pg_tables
     WHERE schemaname = 'public'
       AND rowsecurity = false
       -- internal migration helpers can be excluded here if needed
       AND tablename NOT IN ('schema_migrations');

    IF unprotected IS NOT NULL THEN
        RAISE EXCEPTION
            'RLS AUDIT FAILED. The following tables do not have RLS enabled: %',
            unprotected;
    END IF;
END $$;

-- ---- 3. Baseline policy assertions ---------------------------
-- Every per-user table must have at least one policy that scopes
-- rows to auth.uid(). Insert them if missing; leave existing ones.

-- profiles: users read/update their own row
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'profiles'
           AND policyname = 'profiles_self_select'
    ) THEN
        CREATE POLICY profiles_self_select ON public.profiles
            FOR SELECT USING (auth.uid() = id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'profiles'
           AND policyname = 'profiles_self_update'
    ) THEN
        CREATE POLICY profiles_self_update ON public.profiles
            FOR UPDATE USING (auth.uid() = id) WITH CHECK (auth.uid() = id);
    END IF;
END $$;

-- clothing_items: users manage only their own items
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'clothing_items'
           AND policyname = 'clothing_items_owner_all'
    ) THEN
        CREATE POLICY clothing_items_owner_all ON public.clothing_items
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- saved_outfits: same pattern
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'saved_outfits'
           AND policyname = 'saved_outfits_owner_all'
    ) THEN
        CREATE POLICY saved_outfits_owner_all ON public.saved_outfits
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- wear_logs
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'wear_logs'
           AND policyname = 'wear_logs_owner_all'
    ) THEN
        CREATE POLICY wear_logs_owner_all ON public.wear_logs
            FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
    END IF;
END $$;

-- subscriptions + payments: read-only for the owner; mutations
-- happen exclusively through service_role via webhooks.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'subscriptions'
           AND policyname = 'subscriptions_owner_select'
    ) THEN
        CREATE POLICY subscriptions_owner_select ON public.subscriptions
            FOR SELECT USING (auth.uid() = user_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'payments'
           AND policyname = 'payments_owner_select'
    ) THEN
        CREATE POLICY payments_owner_select ON public.payments
            FOR SELECT USING (auth.uid() = user_id);
    END IF;
END $$;

-- app_config: secrets table. NO anon / authenticated access; only
-- service_role (Edge Functions) may read. We deny everything at the
-- policy level for extra safety.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'app_config'
           AND policyname = 'app_config_deny_all'
    ) THEN
        CREATE POLICY app_config_deny_all ON public.app_config
            FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
    END IF;
END $$;

-- shop_catalog: public read, write restricted to service_role.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'shop_catalog'
           AND policyname = 'shop_catalog_public_read'
    ) THEN
        CREATE POLICY shop_catalog_public_read ON public.shop_catalog
            FOR SELECT USING (true);
    END IF;
END $$;

-- ---- 4. Final verification -----------------------------------
DO $$
DECLARE
    tbl            text;
    policy_count   integer;
    per_user_tables text[] := ARRAY[
        'profiles', 'clothing_items', 'saved_outfits',
        'wear_logs', 'subscriptions', 'payments'
    ];
BEGIN
    FOREACH tbl IN ARRAY per_user_tables
    LOOP
        SELECT count(*) INTO policy_count
          FROM pg_policies
         WHERE schemaname = 'public' AND tablename = tbl;

        IF policy_count = 0 THEN
            RAISE EXCEPTION
                'RLS AUDIT FAILED. Table public.% has RLS enabled but no policies.',
                tbl;
        END IF;
    END LOOP;
END $$;
