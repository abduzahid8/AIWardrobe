-- ============================================================
-- AIWardrobe — Featured Capsules (Inspiration / Shop tab)
-- Admin-editable content driving the `Featured Capsules` row.
-- Run in: Supabase Dashboard → SQL Editor
-- ============================================================

-- ── 1. TABLE ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.featured_capsules (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT        NOT NULL,
    subtitle    TEXT,
    image_url   TEXT        NOT NULL,
    link_url    TEXT,
    sort_order  INTEGER     NOT NULL DEFAULT 0,
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── 2. INDEXES ───────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS featured_capsules_sort_order_idx ON public.featured_capsules (sort_order);
CREATE INDEX IF NOT EXISTS featured_capsules_is_active_idx  ON public.featured_capsules (is_active);

-- ── 3. ROW-LEVEL SECURITY ────────────────────────────────────
ALTER TABLE public.featured_capsules ENABLE ROW LEVEL SECURITY;

-- Anyone can read active capsules (including unauthenticated visitors)
DROP POLICY IF EXISTS "featured_capsules_public_read" ON public.featured_capsules;
CREATE POLICY "featured_capsules_public_read"
    ON public.featured_capsules
    FOR SELECT
    USING (is_active = TRUE);

-- Only service-role (admin / edge functions) can write
DROP POLICY IF EXISTS "featured_capsules_service_write" ON public.featured_capsules;
CREATE POLICY "featured_capsules_service_write"
    ON public.featured_capsules
    FOR ALL
    USING (auth.role() = 'service_role');

-- ── 4. updated_at TRIGGER ────────────────────────────────────
CREATE OR REPLACE FUNCTION public.featured_capsules_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS featured_capsules_updated_at ON public.featured_capsules;
CREATE TRIGGER featured_capsules_updated_at
    BEFORE UPDATE ON public.featured_capsules
    FOR EACH ROW
    EXECUTE FUNCTION public.featured_capsules_set_updated_at();

-- ── 5. SEED (initial content) ────────────────────────────────
-- Matches the previously hardcoded Featured Capsules. Admins can
-- edit/replace these rows from the Supabase dashboard.
INSERT INTO public.featured_capsules (title, image_url, sort_order)
VALUES
    ('Winter Dressing Guide',   'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=600&q=80', 1),
    ('The Cozy Edit',           'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=600&q=80', 2),
    ('Capsule Wardrobe Picks',  'https://images.unsplash.com/photo-1555069519-127aadedf1ee?w=600&q=80', 3)
ON CONFLICT DO NOTHING;
