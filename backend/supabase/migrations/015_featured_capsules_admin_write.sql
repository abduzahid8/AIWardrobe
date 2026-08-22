-- ============================================================
-- 015_featured_capsules_admin_write.sql
-- Allow admin users to edit featured capsules from app
-- ============================================================

-- Keep public read policy as-is, but replace service-only write with
-- admin+service write to match the admin panel behavior.
DROP POLICY IF EXISTS "featured_capsules_service_write" ON public.featured_capsules;
DROP POLICY IF EXISTS "featured_capsules_admin_write" ON public.featured_capsules;

CREATE POLICY "featured_capsules_admin_write"
    ON public.featured_capsules
    FOR ALL
    USING (
        auth.role() = 'service_role'
        OR public.is_admin()
    )
    WITH CHECK (
        auth.role() = 'service_role'
        OR public.is_admin()
    );
